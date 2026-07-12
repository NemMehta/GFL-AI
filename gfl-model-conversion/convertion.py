





# it's working conversion model into mlcore
import os
import uuid
import logging
import shutil
import threading
import zipfile  # To zip the .mlpackage folder
import requests
from flask import Flask, jsonify, request
from azure.storage.blob import ContainerClient, BlobClient
from ultralytics import YOLO
import traceback
import subprocess
from flask import Flask, jsonify, request
import threading
from azure.storage.blob import BlobClient, ContainerClient
from azure.ai.ml import MLClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.ml.entities import CommandJob
import os
import logging
from datetime import datetime
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, Input
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, ComputeTarget, Datastore
import os
from azureml.core.authentication import ServicePrincipalAuthentication


# -------------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------------
LOG_FILE = "/home/site/wwwroot/conversion.log"
os.makedirs("/home/site/wwwroot", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("conversion")


def log_info(msg):
    logger.info(msg)


def log_error(msg, e=None):
    if e:
        tb = traceback.format_exc()
        logger.error(f"{msg}: {e}\nTRACEBACK:\n{tb}")
    else:
        logger.error(msg)


def log_step(step):
    logger.info(f"========== {step} ==========")


# -------------------------------------------------------------------------
# FLASK APP
# -------------------------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------------------------
# AZURE SETTINGS
# -------------------------------------------------------------------------
AZURE_CONFIG = {
    "ConnectionString": "https://gflstorageblob.blob.core.windows.net",
    "Container": "dotnetbackend-container",
    "SasToken": "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2026-01-31T02:09:33Z&st=2025-12-02T17:54:33Z&spr=https&sig=YBF1lZtYZ2oSmNZoFXe6Tssp7G0FMbpqiM3telTANqM%3D"
}

AZ_BASE = AZURE_CONFIG["ConnectionString"]
AZ_CONTAINER = AZURE_CONFIG["Container"]
AZ_SAS = AZURE_CONFIG["SasToken"]

TEMP_DIR = "/home/site/wwwroot/temp_models"
os.makedirs(TEMP_DIR, exist_ok=True)

COREML_FOLDER = "mlcore"
COREML_PREVIOUS = "mlcore_previous"
TFLITE_FOLDER = "tflite"
TFLITE_PREVIOUS = "tflite_previous"

# -------------------------------------------------------------------------
# GLOBAL STATUS
# -------------------------------------------------------------------------
EXPORT_STATUS = {
    "state": "idle",       # idle / running / done / error
    "iosUrl": None,
    "androidUrl": None,
    "error": None,
    "modelId": None,  # Added modelId here to track the request
}

export_lock = threading.Lock()


# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------
def append_sas(url):
    return f"{url}?{AZ_SAS}"


def cleanup_temp():
    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass
    os.makedirs(TEMP_DIR, exist_ok=True)


def get_container_client():
    container_url = f"{AZ_BASE}/{AZ_CONTAINER}?{AZ_SAS}"
    return ContainerClient.from_container_url(container_url)


def get_latest_model_blob():
    cc = get_container_client()
    blobs = list(cc.list_blobs(name_starts_with="models/"))
    pt_files = [b for b in blobs if b.name.endswith(".pt")]

    if not pt_files:
        raise Exception("❌ No .pt model found in /models/")

    latest = max(pt_files, key=lambda b: b.last_modified)
    return latest.name


def download_model(blob_name):
    clean_url = f"{AZ_BASE}/{AZ_CONTAINER}/{blob_name}"
    sas_url = append_sas(clean_url)
    local_file = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.pt")

    blob = BlobClient.from_blob_url(sas_url)
    data = blob.download_blob().readall()

    with open(local_file, "wb") as f:
        f.write(data)

    return local_file


def move_old_models(folder, previous_folder):
    try:
        cc = get_container_client()
        blobs = list(cc.list_blobs(name_starts_with=f"{folder}/"))

        for blob in blobs:
            old_path = blob.name
            relative = old_path[len(folder) + 1:]
            new_path = f"{previous_folder}/{relative}"

            src_clean = f"{AZ_BASE}/{AZ_CONTAINER}/{old_path}"
            src_sas = append_sas(src_clean)

            dst_clean = f"{AZ_BASE}/{AZ_CONTAINER}/{new_path}"
            dst_sas = append_sas(dst_clean)

            BlobClient.from_blob_url(dst_sas).start_copy_from_url(src_sas)
            cc.delete_blob(old_path)

        return True

    except Exception as e:
        log_error("Rotation failed", e)
        return False


def upload_mlpackage_zip(local_mlpackage_dir, azure_folder):
    # Create a zip file from the mlpackage folder
    zip_filename = f"{local_mlpackage_dir}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(local_mlpackage_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_mlpackage_dir)
                zipf.write(local_path, rel_path)

    # Upload the zip file
    zip_file_name = os.path.basename(zip_filename)
    blob_path = f"{azure_folder}/{zip_file_name}"
    clean_url = f"{AZ_BASE}/{AZ_CONTAINER}/{blob_path}"
    sas_url = append_sas(clean_url)

    blob = BlobClient.from_blob_url(sas_url)
    with open(zip_filename, "rb") as f:
        blob.upload_blob(f, overwrite=True)

    os.remove(zip_filename)  # Clean up the local zip file


# -------------------------------------------------------------------------
# BACKGROUND CONVERSION WORKER
# -------------------------------------------------------------------------

def clean_url(url_with_sas):
    # This function will strip the SAS token from the URL and return only the base URL
    base_url = url_with_sas.split("?")[0]
    return base_url



def background_worker(model_id):
    global EXPORT_STATUS

    try:
        # ----------------------------------------------------
        # CLEAN TEMP AND DOWNLOAD PT MODEL
        # ----------------------------------------------------
        cleanup_temp()
        blob_name = get_latest_model_blob()
        local_pt = download_model(blob_name)

        EXPORT_STATUS.update({
            "state": "running",
            "modelId": model_id
        })

        # Load YOLO Model
        model = YOLO(local_pt)

        # ----------------------------------------------------
        # 1. EXPORT TO COREML (.mlpackage)
        # ----------------------------------------------------
        coreml_dir = model.export(
            format="coreml",
            imgsz=640,
            device="cpu",
            optimize=False,
            save_dir=TEMP_DIR
        )

        # If YOLO returns list, take first
        if isinstance(coreml_dir, list):
            coreml_dir = coreml_dir[0]

        # Move existing mlcore files to previous folder
        move_old_models(COREML_FOLDER, COREML_PREVIOUS)

        # Upload zipped .mlpackage
        upload_mlpackage_zip(coreml_dir, COREML_FOLDER)

        # Construct cleaned iOS URL (no SAS token)
        ios_url = f"{AZ_BASE}/{AZ_CONTAINER}/{COREML_FOLDER}/{os.path.basename(coreml_dir)}.zip"
        ios_url = clean_url(ios_url)

        # ----------------------------------------------------
        # 2. CALL TFLITE API (GET, NO INPUT)
        # ----------------------------------------------------
        # tflite_api = "https://model-conversion.azurewebsites.net/export-mobile-tflite"
        tflite_api = "http://108.181.169.74:8000/convert"

        try:
            tflite_response = requests.get(tflite_api, timeout=1500)

            if tflite_response.status_code == 200:
                tflite_json = tflite_response.json()

                if tflite_json.get("status") != "success":
                    raise Exception("TFLite API returned non-success status")

                # Extract blob path
                tflite_blob_path = tflite_json.get("tflite_blob_path")
                if not tflite_blob_path:
                    raise Exception("Missing tflite_blob_path in TFLite API response")

                # Convert tflite path → full Azure Blob URL
                android_url = f"{AZ_BASE}/{AZ_CONTAINER}/{tflite_blob_path}"
                android_url = clean_url(android_url)

            else:
                raise Exception(
                    f"TFLite API error {tflite_response.status_code}: {tflite_response.text}"
                )

        except Exception as e:
            EXPORT_STATUS.update({
                "state": "error",
                "modelId": model_id,
                "iosUrl": ios_url,
                "androidUrl": None,
                "error": f"TFLite conversion failed: {str(e)}"
            })
            # export_lock.release()
            return

        # ----------------------------------------------------
        # SUCCESS — BOTH COREML + TFLITE DONE
        # ----------------------------------------------------
        EXPORT_STATUS.update({
            "state": "done",
            "modelId": model_id,
            "iosUrl": ios_url,
            "androidUrl": android_url,
            "error": None
        })

        # ----------------------------------------------------
        # 3. NOTIFY YOUR ADMIN API
        # ----------------------------------------------------
        external_api_url = "https://gfladmin-stage.azurewebsites.net/api/GflModel/update-gfl-model"

        payload = {
            "modelId": model_id,
            "iosUrl": ios_url,
            "androidUrl": android_url
        }

        response = requests.post(external_api_url, json=payload)

        if response.status_code == 200:
            log_info(f"Successfully notified external API with modelId {model_id}")
        else:
            log_error(f"Failed notifying external API {response.status_code}: {response.text}")

    except Exception as e:
        # Catch unexpected errors
        EXPORT_STATUS.update({
            "state": "error",
            "modelId": model_id,
            "iosUrl": None,
            "androidUrl": None,
            "error": str(e)
        })

    finally:
        cleanup_temp()
        export_lock.release()




# -------------------------------------------------------------------------
# API ROUTES
# -------------------------------------------------------------------------
@app.route("/export-mobile-models", methods=["POST"])
def export_mobile_models():
    # Get the modelId from the request body
    data = request.get_json()
    model_id = data.get("modelId")

    if not model_id:
        return jsonify({"error": "modelId is required"}), 400

    # if not export_lock.acquire(blocking=False):
    #     return jsonify({"error": "A conversion is already running"}), 409

    # If a conversion is already in progress, return a friendly status instead of an error
    if not export_lock.acquire(blocking=False):
        return jsonify({
            "status": "A model conversion is in progress.",
            "modelId": EXPORT_STATUS.get("modelId"),
            "state": EXPORT_STATUS.get("state"),
            # "iosUrl": EXPORT_STATUS.get("iosUrl"),
            # "androidUrl": EXPORT_STATUS.get("androidUrl"),
            "error": EXPORT_STATUS.get("error")
        }), 200


    EXPORT_STATUS.update({
        "state": "running",
        "iosUrl": None,
        "androidUrl": None,
        "error": None,
        "modelId": model_id
    })

    threading.Thread(target=background_worker, args=(model_id,), daemon=True).start()

    return jsonify({"status": "started", "modelId": model_id}), 202


@app.route("/export-mobile-status", methods=["GET"])
def export_status():
    return jsonify(EXPORT_STATUS), 200


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200






# -------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)



    