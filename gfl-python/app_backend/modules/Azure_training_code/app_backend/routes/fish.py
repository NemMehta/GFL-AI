

from flask import Blueprint, request, jsonify, current_app, url_for, send_file
# from app_backend.services.detect import detect_fish, detect_fish_species, get_top_bbox_yolo_format
# from app_backend.services.measure import measure_fish, measure_fish_v2
# from app_backend.database.db import is_duplicate, insert_fish, is_duplicate_v2  # Import DB functions
import base64
import cv2
import numpy as np
import uuid
import os
import time
from datetime import datetime

# from PIL import Image
import logging
from werkzeug.exceptions import RequestEntityTooLarge
from app_backend.config import IMG_SIZE, DB_FOLDER, DATASET_ROOT, BASE_PUBLIC_URL, DB_FOLDER_Unknown_Fish, LOCAL_PREDICTION_MODELS_DIR, PREDICTED_OUTPUT_DIR
from app_backend.services.uniqueness import compare_two_images
from app_backend.utils.helpers import ensure_db_folder
from app_backend.services.fish_sqlite import save_image_file, BASE_DIR 
from app_backend.database.db import DB_PATH_DATA_COLLECTION, insert_unknown_fish
import sqlite3
from pathlib import Path
import json
import base64
from io import BytesIO
import requests
import asyncio
import threading
from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import requests
# from PIL import Image
import io

logger = logging.getLogger(__name__)

# For multimodel training

# from pathlib import Path
# from flask import current_app
# from app_backend.config import DATASET_ROOT, BASE_PUBLIC_URL
# Fish Annotation (builder)
from app_backend.modules.fish_annotation.src.pipeline import process_from_api, group_records_for_json
from app_backend.modules.fish_annotation.src.cli import parse_args
from app_backend.modules.fish_annotation.src.config import CFG

# Fish Training (pipeline + inference)
# from app_backend.modules.fish_training.parse_dataset import task1_run
# from app_backend.modules.fish_training.group_species import task2_run
# from app_backend.modules.fish_training.filter_dataset import generate_filtered_group_dataset
# from app_backend.modules.fish_training.train_models import train_all_groups
# from app_backend.modules.fish_training.multi_model_predict import predict_best_multithread, preload_models
# from app_backend.modules.fish_training.augment_pipeline import build_augmented_dataset
# from app_backend.database.db import insert_predicted
from app_backend.config import LOCAL_PREDICTION_MODELS_DIR, AZURE_ML_WORKSPACE, AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP

import json
from flask import request

from pathlib import Path
import os
from azure.storage.blob import BlobServiceClient
from ultralytics import YOLO

from redis import Redis
from rq import Queue
from rq.job import Job
# from app_backend.modules.fish_training.jobs import run_training_pipeline
# from rq import Worker, Queue
# from rq.connections import Connection
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, Input
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, ComputeTarget, Datastore
import os
from azureml.core.authentication import ServicePrincipalAuthentication

fish_bp = Blueprint("fish_bp", __name__)


from azure.storage.blob import BlobServiceClient

# -------------------- Azure Config --------------------
AZURE_CONFIG = {
    # "ConnectionString": "https://gflstorageaccount.blob.core.windows.net/dotnetbackend-container",
    # "Container": "dotnetbackend-container",
    # # "SasToken": "sp=rcwd&st=2025-09-08T09:08:53Z&se=2026-09-08T17:23:53Z&spr=https&sv=2024-11-04&sr=c&sig=%2FPkcD%2FANv4T6EtPEAjrNw43HQ9Q6nTLM8%2BCkhxq6kw8%3D"
    # "SasToken": "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-12-31T15:38:34Z&st=2025-09-17T07:23:34Z&spr=https&sig=IOG7m4n1ZfNb8SnCOBBURLvSPYKR0qiGjToBH8AmUzs%3D"


    "ConnectionString": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container",
    "Container": "dotnetbackend-container",
    # "SasUrl": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container?sp=r&st=2025-10-14T12:00:0…,
    "SasToken": "sp=rcwl&st=2025-10-15T15:17:31Z&se=2026-10-15T23:32:31Z&spr=https&sv=2024-11-04&sr=c&sig=Cu7aKTRa3s7Thflh3pr1mGm%2BG8TflhSrdYHD%2FXgxJ80%3D"

}

# redis_conn = Redis()
# q = Queue(connection=redis_conn)



# # BASE_UPLOAD_DIR = Path("uploads")
# # BASE_UPLOAD_DIR.mkdir(exist_ok=True)
# preload_models()


 
def _coerce_bool(v):
    if isinstance(v, bool): return v
    if v is None: return False
    return str(v).strip().lower() in {"1","true","yes","y","on"}


def _maybe_json(x):
    if x is None: return None
    if isinstance(x, (dict, list)): return x
    try:
        return json.loads(x)
    except Exception:
        return None


def _parse_payload_post():
    """
    Accept:
      - application/json: {"records":[...], ...}  OR a bare list [...]
      - multipart/form-data or x-www-form-urlencoded:
          * 'records' field as JSON text (recommended)
          * OR any single field containing a JSON object with 'records', or a JSON list

    Returns normalized dict: {
      "records": [...],
      "species_map": {..} | None,
      "json_only": bool,
      "val_fraction": float | None,
      "base_url": str | None
    }
    """
    payload = None

    # 1) Proper JSON body
    if request.is_json:
        payload = request.get_json(silent=True)

    # 2) Raw body (sometimes clients mis-set content-type)
    if payload is None:
        raw = request.get_data(cache=False, as_text=True)
        if raw and raw.strip():
            try:
                payload = json.loads(raw)
            except Exception:
                payload = None

    # 3) Form data
    if payload is None:
        # Prefer explicit 'records' if present
        if "records" in request.form:
            payload = {
                "records": _maybe_json(request.form.get("records")),
                "species_map": _maybe_json(request.form.get("species_map")),
                "json_only": request.form.get("json_only"),
                "val_fraction": request.form.get("val_fraction"),
                "base_url": request.form.get("base_url"),
            }
        else:
            # Fallback: try ANY form value that parses to JSON
            for k, v in request.form.items():
                cand = _maybe_json(v)
                if isinstance(cand, dict) and "records" in cand:
                    payload = cand
                    break
                if isinstance(cand, list):
                    payload = {"records": cand}
                    break

    # Normalize
    if isinstance(payload, list):
        payload = {"records": payload}
    if not isinstance(payload, dict):
        payload = {}

    rec = payload.get("records")
    if isinstance(rec, str):
        rec = _maybe_json(rec)
    payload["records"] = rec if isinstance(rec, list) else []

    sm = payload.get("species_map")
    if isinstance(sm, str):
        sm = _maybe_json(sm)
    payload["species_map"] = sm if isinstance(sm, dict) else None

    payload["json_only"]    = _coerce_bool(payload.get("json_only"))
    # val_fraction handling
    try:
        raw_val = payload.get("val_fraction")
        if raw_val is None or raw_val == "":
            payload["val_fraction"] = CFG.VAL_FRACTION
        else:
            payload["val_fraction"] = float(raw_val)
    except Exception:
        payload["val_fraction"] = CFG.VAL_FRACTION


    return payload


def normalize_species_name(species_name: str) -> str:
    """
    Replace ':' with ' (' and append ')' at the end if needed.
    Example: 'Red Drum: Sciaenops ocellatus' -> 'Red Drum (Sciaenops ocellatus)'
    """
    if ":" in species_name:
        parts = species_name.split(":", 1)
        return f"{parts[0].strip()} ({parts[1].strip()})"
    return species_name.strip()


def upload_file_to_blob(species: str, is_handheld: bool, file) -> str:
    """
    Upload file to Azure Blob Storage inside Annotated/ folder and return blob URL.
    """
    try:
        container_name = AZURE_CONFIG["container"]

        # # Folder naming: Annotated/{species}/Hand-Held or Not-Hand-Held
        # hand_held_folder = "Hand-Held" if is_handheld else "Not-Hand-Held"
        # extension = os.path.splitext(file.filename)[1]
        # blob_name = f"Un-Annotated/{species}/{hand_held_folder}/{uuid.uuid4()}{extension}"

        species_folder = normalize_species_name(species)
        hand_held_folder = "Hand-Held" if is_handheld else "Not-Hand-Held"
        extension = os.path.splitext(file.filename)[1]
        blob_name = f"Annotated/{species_folder}/{hand_held_folder}/{uuid.uuid4()}{extension}"


        # # Blob client
        # blob_service_client = BlobServiceClient(account_url=AZURE_CONFIG["connection_string"],
        #                                         credential=AZURE_CONFIG["sas_token"])
        # blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # blob_client.upload_blob(file.read(), overwrite=True)

        # return blob_client.url
    

        blob_service_client = BlobServiceClient(
            account_url=AZURE_CONFIG["connection_string"],
            credential=AZURE_CONFIG["sas_token"]
        )
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Upload
        blob_client.upload_blob(file.read(), overwrite=True)

        # 👇 Return *clean URL* (no SAS token)
        return f"{AZURE_CONFIG['connection_string']}/{container_name}/{blob_name}"



    except Exception as ex:
        raise RuntimeError(f"Azure upload failed: {str(ex)}")


import asyncio

# This is final working version which sends the response to external API
@fish_bp.route("/build_dataset_from_api", methods=["POST"])
def build_dataset_from_api():
    try:
        payload = _parse_payload_post()
        print("DEBUG payload:", payload)

        images = payload.get("images") or []
        if not isinstance(images, list) or not images:
            return jsonify({"status": "error", "message": "Missing or invalid 'images' (list)"}), 400

        defaults = parse_args()

        # ✅ run async code inside sync route
        grouped = asyncio.run(process_from_api(
            payload=payload,
            yolo_model_path=defaults.yolo_model,
            dataset_root=defaults.dataset_root,
            base_dir=defaults.base_dir,
        ))

        return jsonify(grouped), 200

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        logging.exception("Error in dataset build (API payload)")
        return jsonify({"status": "error", "message": str(e)}), 500



@fish_bp.route("/dataset/<path:filename>")
def serve_dataset_file(filename):
    file_path = CFG.DATASET_ROOT / filename
    if file_path.exists():
        return send_file(file_path)
    return "File not found", 404


@fish_bp.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    logging.warning("Upload failed: file too large.")
    return jsonify({
        "status": False,
        "message": "Uploaded image is too large.",
        "body": None
    }), 413




# -------------------------
# Routes
# -------------------------

import re
# ---- Helper function to sanitize folder names for Windows ----
def sanitize_filename(name: str) -> str:
    # Replace invalid Windows filename characters with underscore
    return re.sub(r'[\\/*?:"<>|]', "_", name)


# ---- Species ----
@fish_bp.route("/species", methods=["GET", "POST"])
def species():
    if request.method == "POST":
        data = request.json
        species_name = data.get("species_name")
        if not species_name:
            return jsonify({"error": "species_name is required"}), 400

        try:
            conn = sqlite3.connect(DB_PATH_DATA_COLLECTION)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO species (species_name) VALUES (?)", (species_name,)
            )
            conn.commit()
            species_id = cursor.lastrowid
            conn.close()
            return jsonify({"message": f"Species '{species_name}' added", "id": species_id}), 201
        except sqlite3.IntegrityError:
            return jsonify({"error": f"Species '{species_name}' already exists"}), 400

    else:  # GET method
        conn = sqlite3.connect(DB_PATH_DATA_COLLECTION)
        cursor = conn.cursor()
        cursor.execute("SELECT id, species_name FROM species")
        rows = cursor.fetchall()
        conn.close()
        return jsonify([{"id": r[0], "species_name": r[1]} for r in rows])





training_lock = threading.Lock()
TRAINING_STATUS = {"state": "idle", "message": "No training running."}



#------------------------------------------------------------------------------


# @fish_bp.route("/run_pipeline", methods=["POST"])
# def run_pipeline():
#     """Run the full training pipeline via Azure ML."""
#     global TRAINING_STATUS

#     data = request.get_json(silent=True) or {}
#     species_name = data.get("speciesName")
#     project_id = data.get("projectId")

#     # --- Basic validation ---
#     if not species_name:
#         return jsonify({"status": "error", "message": "speciesName is required"}), 400
#     if not project_id:
#         return jsonify({"status": "error", "message": "projectId is required"}), 400

#     if TRAINING_STATUS.get("state") == "running":
#         return jsonify({
#             "status": "busy",
#             "message": "Training is already running. Please wait for it to complete."
#         }), 429

#     # --- Background worker for non-blocking response ---
#     def background_job(species_name: str, project_id: int):
#         global TRAINING_STATUS
#         lock_acquired = training_lock.acquire(blocking=False)

#         if not lock_acquired:
#             print("⚠️ Another training job is already running — skipping new one.")
#             logging.info("⚠️ Another training job is already running — skipping new one.")
#             return

#         try:
#             TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}
#             print(f"🚀 Submitting Azure ML training for project {project_id}, species: {species_name}")
#             logging.info(f"🚀 Submitting Azure ML training for project {project_id}, species: {species_name}")

#             # --- Authenticate to Azure ML ---
#             credential = DefaultAzureCredential()
#             ml_client = MLClient(
#                 credential=credential,
#                 subscription_id=AZURE_SUBSCRIPTION_ID,
#                 resource_group_name=AZURE_RESOURCE_GROUP,
#                 workspace_name=AZURE_ML_WORKSPACE,
#             )

#             # compute_name = "GflTrainCompute"
#             compute_name = "trainingComputev2"
            
#             # --- Ensure compute exists and is running ---

#             # --- Ensure compute exists and is running ---
#             try:
#                 compute = ml_client.compute.get(compute_name)
#                 state = compute.properties.get("provisioning_state", None)
#                 compute_type = compute.properties.get("compute_type", "Unknown")
                
#                 print(f"ℹ️ Compute '{compute_name}' type: {compute_type}, state: {state}")
#                 logging.info(f"ℹ️ Compute '{compute_name}' type: {compute_type}, state: {state}")

#                 # --- Only try to start if needed ---
#                 if compute_type.lower() == "computeinstance":
#                     # Only start if it's stopped
#                     if state and state.lower() == "stopped":
#                         print(f"⚙️ Starting compute instance '{compute_name}'...")
#                         logging.info(f"⚙️ Starting compute instance '{compute_name}'...")
#                         poller = ml_client.compute.begin_start(compute_name)
#                         poller.wait()  # Wait for it to start
#                         print(f"✅ Compute instance '{compute_name}' started successfully.")
#                         logging.info(f"✅ Compute instance '{compute_name}' started successfully.")
#                     else:
#                         print(f"✅ Compute instance '{compute_name}' already running (state={state}).")
#                         logging.info(f"✅ Compute instance '{compute_name}' already running (state={state}).")

#                 else:
#                     # It's a cluster — clusters don't need to be manually started
#                     print(f"✅ Compute cluster '{compute_name}' is available and ready.")
#                     logging.info(f"✅ Compute cluster '{compute_name}' is available and ready.")

#             except Exception as e:
#                 print(f"❌ Error fetching compute target '{compute_name}': {e}")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"Compute target '{compute_name}' not found or failed to start in Azure ML workspace."
#                 }
#                 training_lock.release()
#                 return



#             # try:
#             #     compute = ml_client.compute.get(compute_name)
#             #     state = compute.properties.get("provisioning_state", None)
#             #     print(f"ℹ️ Compute '{compute_name}' current state: {state}")
#             #     logging.info(f"ℹ️ Compute '{compute_name}' current state: {state}")
#             #     ml_client.compute.begin_start(compute_name).wait(120)
#             #     print(f"✅ Compute '{compute.provisioning_state}' is starting...")

#             #     # if compute.provisioning_state == 'Stopped':
#             #     #     print(f"⚠️ Compute '{compute_name}' is stopped. Starting it now...")
#             #     #     ml_client.compute.begin_start(compute_name)
#             #     #     print(f"✅ Compute '{compute_name}' is starting...")
                
#             #     if state in ["Stopped", "Deallocated", "Unknown", "Stopping"]:
#             #         print(f"⚠️ Compute '{compute_name}' is not running. Starting now...")
#             #         logging.info(f"⚠️ Compute '{compute_name}' is not running. Starting now...")
#             #         poller = ml_client.compute.begin_start(compute_name)
#             #         poller.wait()  # Wait for start to complete
#             #         print(f"✅ Compute '{compute_name}' has started successfully.")
#             #     else:
#             #         logging.info(f"✅ Compute '{compute_name}' is already running.")


#             # except Exception as e:
#             #     print(f"❌ Error fetching compute target '{compute_name}': {e}")
                
                
#             #     TRAINING_STATUS = {
#             #         "state": "failed",
#             #         "message": f"Compute target '{compute_name}' not found or failed to start in Azure ML workspace."
#             #     }
#             #     training_lock.release()
#             #     return

#             # --- Access datastore dynamically ---
#             try:
#                 logging.info("🔍 Retrieving workspace datastore...")
#                 # Connect using AzureML v1 to get blob path
#                 # ✅ Fetch datastore directly via MLClient
#                 datastore = ml_client.datastores.get("workspaceblobstore")
#                 print(f"✅ Datastore retrieved: {datastore.name}")
#                 logging.info(f"✅ Datastore retrieved: {datastore.name}")
                
#                 # Construct URI safely
#                 source_path = "UI/JobSubmission/10-23-2025_102845_UTC"
#                 datastore_uri = f"azureml://datastores/{datastore.name}/paths/{source_path}/"
#                 print(f"📁 Using datastore path: {datastore_uri}")
#                 logging.info(f"📁 Using datastore path: {datastore_uri}")

#                 logging.info(f"✅ Datastore retrieved: {datastore.name}")

#                 # Define source directory path inside blob
#                 source_path = "UI/JobSubmission/10-23-2025_102845_UTC"
#                 datastore_uri = f"azureml://datastores/{datastore.name}/paths/{source_path}/"
#                 logging.info(f"📁 Using datastore path: {datastore_uri}")

#             except Exception as ds_error:
#                 logging.info(f"❌ Datastore access failed: {ds_error}")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"Datastore access failed: {ds_error}"
#                 }
#                 training_lock.release()
#                 return

#             # --- Define and submit the job ---
#             job = command(
#                 code=datastore_uri,  # ✅ now uses datastore URI dynamically
#                 command=(
#                     f"pip install -r requirements.txt && "
#                     f"python train_pipeline_entry.py --species '{species_name}' --project_id {project_id}"
#                 ),
#                 environment=(
#                     "azureml://locations/centralus/workspaces/"
#                     "38466c87-eabd-4f91-96d1-2d8c68b55e3c/"
#                     "environments/Train_env2/versions/3"
#                 ),
#                 compute=compute_name,
#                 experiment_name="Default",
#                 display_name=f"fish_train_{species_name}_{project_id}",
#                 description="YOLO fish species training job"
#             )

#             submitted_job = ml_client.jobs.create_or_update(job)

#             TRAINING_STATUS = {
#                 "state": "submitted",
#                 "message": f"✅ Azure ML job submitted: {submitted_job.name}",
#                 "job_name": submitted_job.name
#             }
#             logging.info(f"✅ Submitted Azure ML job: {submitted_job.name}")

#         except Exception as e:
#             TRAINING_STATUS = {
#                 "state": "failed",
#                 "message": f"❌ Azure ML job submission failed: {str(e)}"
#             }
#             logging.info(f"❌ Azure ML submission failed: {e}")

#         # finally:
#         #     if lock_acquired:
#         #         training_lock.release()
                
#         finally:
#             # Always release safely
#             if training_lock.locked():
#                 training_lock.release()


#     # --- Run background thread ---
#     thread = threading.Thread(target=background_job, args=(species_name, project_id), daemon=True)
#     thread.start()

#     # Immediate HTTP response
#     return jsonify({
#         "status": "started",
#         "message": f"Training job submitted for project {project_id}, species {species_name}.",
#         "check_status_at": "/pipeline_status"
#     }), 202



from pathlib import Path
import tempfile

TEMP_DATA_DIR = Path(tempfile.gettempdir()) / "gfl_training"



def download_species_yaml():
    """
    Downloads data.yaml from blob storage and saves it locally as species_data.yaml.
    Returns the local file path.
    """
    blob_url = "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container/augmentation/training_data/dataset.yaml"

    local_yaml_path = TEMP_DATA_DIR / "species_data.yaml"
    local_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"⬇ Downloading species YAML from: {blob_url}")
        response = requests.get(blob_url, timeout=30)

        if response.status_code != 200:
            raise Exception(f"Failed to download YAML: HTTP {response.status_code}")

        with open(local_yaml_path, "wb") as f:
            f.write(response.content)

        logger.info(f"📁 Saved species YAML to: {local_yaml_path}")
        return local_yaml_path

    except Exception as e:
        logger.error(f"❌ Failed to download species YAML: {e}")
        raise


def load_species_indexes_from_yaml():

    try:
        yaml_path = download_species_yaml()

        import yaml
        with open(yaml_path, "r") as f:
            yaml_content = yaml.safe_load(f)

        names = yaml_content.get("names")
        if not names or not isinstance(names, list):
            raise ValueError("names list not found in YAML")

        species_indexes = [
            {
                "id": index,
                "speciesName": species_name
            }
            for index, species_name in enumerate(names)
        ]

        logger.info(f"✅ Loaded {len(species_indexes)} species from YAML (names)")
        return species_indexes

    except Exception as e:
        logger.exception("❌ Failed loading species indexes from YAML")
        raise


@fish_bp.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    """Run the full multi-species training pipeline inside Azure ML (ONE job)."""
    global TRAINING_STATUS

    data = request.get_json(silent=True) or []

    if not data or not isinstance(data, list):
        return jsonify({
            "status": "error",
            "message": "Input must be a list of species objects"
        }), 400

    # Validate entries
    for entry in data:
        if not entry.get("speciesName") or not entry.get("projectId"):
            return jsonify({
                "status": "error",
                "message": "Each entry must contain speciesName and projectId"
            }), 400

    # Prevent double-run
    if TRAINING_STATUS["state"] == "running":
        return jsonify({
            "status": "busy",
            "message": "Training pipeline already running"
        }), 429

    # -----------------------------------------------------
    # Background Thread
    # -----------------------------------------------------
    def background_job(entries: list):
        global TRAINING_STATUS

        if not training_lock.acquire(blocking=False):
            print("⚠ Another training job already running — skipping.")
            return

        try:
            TRAINING_STATUS = {
                "state": "running",
                "message": "Training started..."
            }

            # -----------------------------------------
            # STEP A — Extract species → projects mapping
            # -----------------------------------------
            # 1. Load species from YAML (class_map)
            yaml_species = load_species_indexes_from_yaml()
            yaml_species_names = {s["speciesName"] for s in yaml_species}

            species_list = []
            species_to_projects = {}

            for e in entries:
                sp = e["speciesName"]
                pid = int(e["projectId"])

                # 2. Validate species against YAML
                if sp not in yaml_species_names:
                    raise ValueError(f"Invalid species '{sp}' — not found in species YAML")

                if sp not in species_list:
                    species_list.append(sp)

                species_to_projects.setdefault(sp, [])
                if pid not in species_to_projects[sp]:
                    species_to_projects[sp].append(pid)

            print(f"🐟 Species selected: {species_list}")
            print(f"📦 Species → Projects: {species_to_projects}")

            # Save mapping for Azure ML (train_pipeline_entry.py will read these)
            train_species_json = json.dumps(species_list)
            species_project_map_json = json.dumps(species_to_projects)

            # -----------------------------------------------------
            # SUBMIT SINGLE AZURE ML JOB
            # -----------------------------------------------------
            try:
                credential = DefaultAzureCredential()
                ml_client = MLClient(
                    credential=credential,
                    subscription_id=AZURE_SUBSCRIPTION_ID,
                    resource_group_name=AZURE_RESOURCE_GROUP,
                    workspace_name=AZURE_ML_WORKSPACE,
                )

                compute_name = "GFL-GPU-Max" 
                # compute_name = "GFL-GPU" 
                #compute_name = "gflTrainCluster"
                print("Using compute:", compute_name)

                # Optional: ensure compute exists
                compute = ml_client.compute.get(compute_name)
                print(f"ℹ️ Compute '{compute_name}' state:", compute.provisioning_state)

            except Exception as e:
                TRAINING_STATUS = {
                    "state": "failed",
                    "message": f"Azure ML setup failed: {str(e)}"
                }
                return

            # Data asset (your code base)
            # data_asset = ml_client.data.get("training_pipeline", version="2")
            code_input = Input(
                type="uri_folder",
                # path=data_asset.path
                path="azureml://subscriptions/417ecdbc-9b50-421a-82c2-c45f29686df1/"
                     "resourcegroups/GFL-App-Rg/workspaces/GFL-MLAI/"
                     "datastores/workspaceblobstore/paths/UI/JobSubmission/11-25-2025_120606_UTC/"
            )

            # -----------------------------------------------------
            # Build Azure ML Job (ONE job — includes ALL species)
            # -----------------------------------------------------
            job = command(
                code=None,
                inputs={"code_dir": code_input},
                command=(
                    "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && "
                    "pip install -r ${{inputs.code_dir}}/requirements.txt && "
                    "python ${{inputs.code_dir}}/train_pipeline_entry.py"
                ),
                environment="gflTrainEnv:5",
                compute=compute_name,
                experiment_name="GFLTrainingGlobalModel",
                display_name=f"fish_global_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description="Global YOLO model training for multiple species",
                environment_variables={
                    "TRAIN_SPECIES": train_species_json,
                    "SPECIES_PROJECT_MAP": species_project_map_json,
                    "BLOB_BASE_URL": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container",
                    "BLOB_CONTAINER": "dotnetbackend-container",
                    "BLOB_SAS_TOKEN": "sp=rcwl&st=2025-10-15T15:17:31Z&se=2026-10-15T23:32:31Z&spr=https&sv=2024-11-04&sr=c&sig=Cu7aKTRa3s7Thflh3pr1mGm%2BG8TflhSrdYHD%2FXgxJ80%3D"
                }
            )

            submitted_job = ml_client.jobs.create_or_update(job)

            TRAINING_STATUS = {
                "state": "submitted",
                "message": f"Azure ML job submitted: {submitted_job.name}",
                "job_name": submitted_job.name,
                "species": species_list
            }

            print(f"🎉 Azure job submitted: {submitted_job.name}")

        except Exception as e:
            logger.exception(f"❌ Training pipeline failed: {e}")
            TRAINING_STATUS = {
                "state": "failed",
                "message": f"Training failed: {str(e)}"
            }

        finally:
            training_lock.release()

    # -----------------------------------------------------
    # Start background thread
    # -----------------------------------------------------
    thread = threading.Thread(target=background_job, args=(data,), daemon=True)
    thread.start()

    return jsonify({
        "status": "started",
        "message": f"Training pipeline started for {len(data)} species. Check /pipeline_status.",
    }), 202







# Old work
# @fish_bp.route("/run_pipeline", methods=["POST"])
# def run_pipeline():
#     """Run the full multi-species training pipeline inside Azure ML (ONE job)."""
#     global TRAINING_STATUS

#     data = request.get_json(silent=True) or []

#     if not data or not isinstance(data, list):
#         return jsonify({
#             "status": "error",
#             "message": "Input must be a list of species objects"
#         }), 400

#     # Validate entries
#     for entry in data:
#         if not entry.get("speciesName") or not entry.get("projectId"):
#             return jsonify({
#                 "status": "error",
#                 "message": "Each entry must contain speciesName and projectId"
#             }), 400

#     # Prevent double-run
#     if TRAINING_STATUS["state"] == "running":
#         return jsonify({
#             "status": "busy",
#             "message": "Training pipeline already running"
#         }), 429

#     # -----------------------------------------------------
#     # Background Thread
#     # -----------------------------------------------------
#     def background_job(entries: list):
#         global TRAINING_STATUS

#         if not training_lock.acquire(blocking=False):
#             print("⚠ Another training job already running — skipping.")
#             return

#         try:
#             TRAINING_STATUS = {
#                 "state": "running",
#                 "message": "Training started..."
#             }

#             # -----------------------------------------
#             # STEP A — Extract species → projects mapping
#             # -----------------------------------------
#             species_list = []
#             species_to_projects = {}

#             for e in entries:
#                 sp = e["speciesName"]
#                 pid = int(e["projectId"])

#                 if sp not in species_list:
#                     species_list.append(sp)

#                 species_to_projects.setdefault(sp, [])
#                 if pid not in species_to_projects[sp]:
#                     species_to_projects[sp].append(pid)

#             print(f"🐟 Species selected: {species_list}")
#             print(f"📦 Species → Projects: {species_to_projects}")

#             # Save mapping for Azure ML (train_pipeline_entry.py will read these)
#             train_species_json = json.dumps(species_list)
#             species_project_map_json = json.dumps(species_to_projects)

#             # -----------------------------------------------------
#             # SUBMIT SINGLE AZURE ML JOB
#             # -----------------------------------------------------
#             try:
#                 credential = DefaultAzureCredential()
#                 ml_client = MLClient(
#                     credential=credential,
#                     subscription_id=AZURE_SUBSCRIPTION_ID,
#                     resource_group_name=AZURE_RESOURCE_GROUP,
#                     workspace_name=AZURE_ML_WORKSPACE,
#                 )

#                 compute_name = "gflTrainCluster"
#                 print("Using compute:", compute_name)

#                 # Optional: ensure compute exists
#                 compute = ml_client.compute.get(compute_name)
#                 print(f"ℹ️ Compute '{compute_name}' state:", compute.provisioning_state)

#             except Exception as e:
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"Azure ML setup failed: {str(e)}"
#                 }
#                 return

#             # Data asset (your code base)
#             data_asset = ml_client.data.get("training_pipeline", version="2")
#             code_input = Input(
#                 type="uri_folder",
#                 path=data_asset.path
#             )

#             # -----------------------------------------------------
#             # Build Azure ML Job (ONE job — includes ALL species)
#             # -----------------------------------------------------
#             job = command(
#                 code=None,
#                 inputs={"code_dir": code_input},
#                 command=(
#                     "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && "
#                     "pip install -r ${{inputs.code_dir}}/requirements.txt && "
#                     "python ${{inputs.code_dir}}/train_pipeline_entry.py"
#                 ),
#                 environment="gflTrainEnv:5",
#                 compute=compute_name,
#                 experiment_name="GFLTrainingGlobalModel",
#                 display_name=f"fish_global_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
#                 description="Global YOLO model training for multiple species",
#                 environment_variables={
#                     "TRAIN_SPECIES": train_species_json,
#                     "SPECIES_PROJECT_MAP": species_project_map_json,
#                     "BLOB_BASE_URL": "https://gflstorageaccount.blob.core.windows.net/dotnetbackend-container",
#                     "BLOB_CONTAINER": "dotnetbackend-container",
#                     "BLOB_SAS_TOKEN": "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-12-31T15:38:34Z&st=2025-09-17T07:23:34Z&spr=https&sig=IOG7m4n1ZfNb8SnCOBBURLvSPYKR0qiGjToBH8AmUzs%3D"
#                 }
#             )

#             submitted_job = ml_client.jobs.create_or_update(job)

#             TRAINING_STATUS = {
#                 "state": "submitted",
#                 "message": f"Azure ML job submitted: {submitted_job.name}",
#                 "job_name": submitted_job.name,
#                 "species": species_list
#             }

#             print(f"🎉 Azure job submitted: {submitted_job.name}")

#         except Exception as e:
#             logger.exception(f"❌ Training pipeline failed: {e}")
#             TRAINING_STATUS = {
#                 "state": "failed",
#                 "message": f"Training failed: {str(e)}"
#             }

#         finally:
#             training_lock.release()

#     # -----------------------------------------------------
#     # Start background thread
#     # -----------------------------------------------------
#     thread = threading.Thread(target=background_job, args=(data,), daemon=True)
#     thread.start()

#     return jsonify({
#         "status": "started",
#         "message": f"Training pipeline started for {len(data)} species. Check /pipeline_status.",
#     }), 202







# # Old work
# @fish_bp.route("/run_pipeline", methods=["POST"])
# def run_pipeline():
#     """Run the full training pipeline via Azure ML."""
#     global TRAINING_STATUS

#     data = request.get_json(silent=True) or {}
#     species_name = data.get("speciesName")
#     project_id = data.get("projectId")

#     # --- Basic validation ---
#     if not species_name:
#         return jsonify({"status": "error", "message": "speciesName is required"}), 400
#     # # Allow "all" as a special keyword
#     # if species_name.lower() == "all":
#     #     print("🌍 Running full multi-species training pipeline.")
#     if not project_id:
#         return jsonify({"status": "error", "message": "projectId is required"}), 400

#     if TRAINING_STATUS.get("state") == "running":
#         return jsonify({
#             "status": "busy",
#             "message": "Training is already running. Please wait for it to complete."
#         }), 429

#     # --- Background worker for non-blocking response ---
#     def background_job(species_name: str, project_id: int):
#         global TRAINING_STATUS
#         lock_acquired = training_lock.acquire(blocking=False)

#         if not lock_acquired:
#             print("⚠️ Another training job is already running — skipping new one.")
#             logging.info("⚠️ Another training job is already running — skipping new one.")
#             return

#         try:
#             TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}
#             print(f"🚀 Submitting Azure ML training for project {project_id}, species: {species_name}")
#             logging.info(f"🚀 Submitting Azure ML training for project {project_id}, species: {species_name}")

#             # --- Authenticate to Azure ML ---
#             credential = DefaultAzureCredential()
#             ml_client = MLClient(
#                 credential=credential,
#                 subscription_id=AZURE_SUBSCRIPTION_ID,
#                 resource_group_name=AZURE_RESOURCE_GROUP,
#                 workspace_name=AZURE_ML_WORKSPACE,
#             )

#             # compute_name = "GflTrainCompute"
#             # compute_name = "gflTrainComputeV1"
#             compute_name = "gflTrainCluster"
#             print("my new Compute cluster is: ", compute_name)
#             logging.info("my new Compute cluster is: ", compute_name)
            
#             # --- Ensure compute exists and is running ---
#             try:
#                 compute = ml_client.compute.get(compute_name)
#                 state = compute.properties.get("provisioning_state", None)
#                 print(f"ℹ️ Compute '{compute_name}' current state: {state}")
#                 logging.info(f"ℹ️ Compute '{compute_name}' current state: {state}")
#                 #ml_client.compute.begin_start(compute_name).wait()
#                 print(f"✅ Compute '{compute.provisioning_state}' is starting...")

#                 # if compute.provisioning_state == 'Stopped':
#                 #     print(f"⚠️ Compute '{compute_name}' is stopped. Starting it now...")
#                 #     ml_client.compute.begin_start(compute_name)
#                 #     print(f"✅ Compute '{compute_name}' is starting...")
                
#                 if state in ["Stopped", "Deallocated", "Unknown", "Stopping", "None"]:
#                     print(f"⚠️ Compute '{compute_name}' is not running. Starting now...")
#                     logging.info(f"⚠️ Compute '{compute_name}' is not running. Starting now...")
#                     #poller = ml_client.compute.begin_start(compute_name)
#                     #poller.wait()  # Wait for start to complete
#                     print(f"✅ Compute '{compute_name}' has started successfully.")
#                 else:
#                     logging.info(f"✅ Compute '{compute_name}' is already running.")


#             except Exception as e:
#                 print(f"❌ Error fetching compute target '{compute_name}': {e}")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"Compute target '{compute_name}' not found or failed to start in Azure ML workspace."
#                 }
#                 training_lock.release()
#                 return

#             # --- Access datastore dynamically ---
#             try:
#                 logging.info("🔍 Retrieving workspace datastore...")
#                 # Connect using AzureML v1 to get blob path
#                 # ✅ Fetch datastore directly via MLClient
#                 datastore = ml_client.datastores.get("workspaceblobstore")
#                 print(f"✅ Datastore retrieved: {datastore.name}")
#                 logging.info(f"✅ Datastore retrieved: {datastore.name}")
                
#                 # Construct URI safely
#                 source_path = "UI/JobSubmission/10-23-2025_102845_UTC"
#                 datastore_uri = f"azureml://subscriptions/417ecdbc-9b50-421a-82c2-c45f29686df1/resourcegroups/GFL-App-Rg/workspaces/GFL-MLAI/datastores/workspaceblobstore/paths/{source_path}/"
#                 print(f"📁 Using datastore path: {datastore_uri}")
#                 logging.info(f"📁 Using datastore path: {datastore_uri}")

#                 logging.info(f"✅ Datastore retrieved: {datastore.name}")

#                 # Define source directory path inside blob
#                 # source_path = "UI/JobSubmission/10-23-2025_102845_UTC"
#                 # datastore_uri = f"azureml://datastores/{datastore.name}/paths/{source_path}/"
#                 # logging.info(f"📁 Using datastore path: {datastore_uri}")

#             except Exception as ds_error:
#                 logging.info(f"❌ Datastore access failed: {ds_error}")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"Datastore access failed: {ds_error}"
#                 }
#                 training_lock.release()
#                 return
#             # Get the data asset for the registered code (as you have already done)
#             data_asset = ml_client.data.get("training_pipeline", version="2")
#             print("Data assets path code folder",data_asset.path)
#             code_input = Input(
#                             type="uri_folder",
#                             path="azureml://subscriptions/417ecdbc-9b50-421a-82c2-c45f29686df1/"
#                                  "resourcegroups/GFL-App-Rg/workspaces/GFL-MLAI/"
#                                  "datastores/workspaceblobstore/paths/UI/JobSubmission/10-23-2025_102845_UTC/"
#                         )


#             job = command(
#                 code=None,
#                 inputs={"code_dir": code_input},
#                 command=(
#                     "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 &&"
#                     "pip install -r ${{inputs.code_dir}}/requirements.txt && "
#                     "python ${{inputs.code_dir}}/train_pipeline_entry.py "
#                     f"--species '{species_name}' --project_id {project_id}"
#                 ),
#                 # ✅ Correct environment format
#                 environment="gflTrainEnv:5",
#                 compute=compute_name,
#                 experiment_name="GFLTrainingModelV1",
#                 # experiment_name = "gflTrainCluster",
#                 display_name=f"fish_train_{species_name}_{project_id}",
#                 description="YOLO fish species training job",
#                 environment_variables={   # ✅ Add these lines
#                     "BLOB_BASE_URL": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container",
#                     "BLOB_CONTAINER": "dotnetbackend-container",
#                     "BLOB_SAS_TOKEN": "sp=rcwl&st=2025-10-15T15:17:31Z&se=2026-10-15T23:32:31Z&spr=https&sv=2024-11-04&sr=c&sig=Cu7aKTRa3s7Thflh3pr1mGm%2BG8TflhSrdYHD%2FXgxJ80%3D"
#                 }
#             )



#             submitted_job = ml_client.jobs.create_or_update(job)
            
#             TRAINING_STATUS = {
#                 "state": "submitted",
#                 "message": f"✅ Azure ML job submitted: {submitted_job.name}",
#                 "job_name": submitted_job.name
#             }
#             logging.info(f"✅ Submitted Azure ML job: {submitted_job.name}")

#         except Exception as e:
#             TRAINING_STATUS = {
#                 "state": "failed",
#                 "message": f"❌ Azure ML job submission failed: {str(e)}"
#             }
#             logging.info(f"❌ Azure ML submission failed: {e}")

#         # finally:
#         #     if lock_acquired:
#         #         training_lock.release()
                
#         finally:
#             # Always release safely
#             if training_lock.locked():
#                 training_lock.release()


#     # --- Run background thread ---
#     thread = threading.Thread(target=background_job, args=(species_name, project_id), daemon=True)
#     thread.start()

#     # Immediate HTTP response
#     return jsonify({
#         "status": "started",
#         "message": f"Training job submitted for project {project_id}, species {species_name}.",
#         "check_status_at": "/pipeline_status"
#     }), 202



@fish_bp.route("/pipeline_status", methods=["GET"])
def pipeline_status():
    """Check current training pipeline status."""
    return jsonify(TRAINING_STATUS), 200






















# # Old work on 21/11/2025
# from flask import Blueprint, request, jsonify, current_app, url_for, send_file
# # from app_backend.services.detect import detect_fish, detect_fish_species, get_top_bbox_yolo_format
# # from app_backend.services.measure import measure_fish, measure_fish_v2
# # from app_backend.database.db import is_duplicate, insert_fish, is_duplicate_v2  # Import DB functions
# import base64
# import cv2
# import numpy as np
# import uuid
# import os
# import time
# from datetime import datetime
# # from PIL import Image
# import logging
# logger = logging.getLogger(__name__)
# from werkzeug.exceptions import RequestEntityTooLarge
# from app_backend.config import IMG_SIZE, DB_FOLDER, DATASET_ROOT, BASE_PUBLIC_URL, DB_FOLDER_Unknown_Fish, LOCAL_PREDICTION_MODELS_DIR, PREDICTED_OUTPUT_DIR
# from app_backend.services.uniqueness import compare_two_images
# from app_backend.utils.helpers import ensure_db_folder
# from app_backend.services.fish_sqlite import save_image_file, BASE_DIR 
# from app_backend.database.db import DB_PATH_DATA_COLLECTION, insert_unknown_fish
# import sqlite3
# from pathlib import Path
# import json
# import base64
# from io import BytesIO
# import requests
# import asyncio
# import threading
# from flask import Flask, request, jsonify
# from ultralytics import YOLO
# import os
# import requests
# from datetime import datetime
# # from PIL import Image
# import io


# # For multimodel training

# from pathlib import Path
# from flask import current_app
# from app_backend.config import DATASET_ROOT, BASE_PUBLIC_URL
# # Fish Annotation (builder)
# from app_backend.modules.fish_annotation.src.pipeline import process_from_api, group_records_for_json
# from app_backend.modules.fish_annotation.src.cli import parse_args
# from app_backend.modules.fish_annotation.src.config import CFG

# # Fish Training (pipeline + inference)
# from app_backend.modules.fish_training.parse_dataset import task1_run
# from app_backend.modules.fish_training.group_species import task2_run
# from app_backend.modules.fish_training.filter_dataset import generate_filtered_group_dataset
# from app_backend.modules.fish_training.train_models import train_all_groups
# from app_backend.modules.fish_training.multi_model_predict import predict_best_multithread, preload_models
# from app_backend.modules.fish_training.augment_pipeline import build_augmented_dataset_multi
# from app_backend.database.db import insert_predicted
# from app_backend.config import LOCAL_PREDICTION_MODELS_DIR, AZURE_ML_WORKSPACE, AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP

# import json
# from flask import request

# from pathlib import Path
# import os
# from azure.storage.blob import BlobServiceClient
# from ultralytics import YOLO

# from redis import Redis
# from rq import Queue
# from rq.job import Job
# # from app_backend.modules.fish_training.jobs import run_training_pipeline
# # from rq import Worker, Queue
# # from rq.connections import Connection
# from azure.identity import DefaultAzureCredential
# from azure.ai.ml import MLClient, command, Input
# from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, ComputeTarget, Datastore
# import datetime
# import os
# from azureml.core.authentication import ServicePrincipalAuthentication

# fish_bp = Blueprint("fish_bp", __name__)


# from azure.storage.blob import BlobServiceClient

# # -------------------- Azure Config --------------------
# AZURE_CONFIG = {
#     # "ConnectionString": "https://gflstorageaccount.blob.core.windows.net/dotnetbackend-container",
#     # "Container": "dotnetbackend-container",
#     # # "SasToken": "sp=rcwd&st=2025-09-08T09:08:53Z&se=2026-09-08T17:23:53Z&spr=https&sv=2024-11-04&sr=c&sig=%2FPkcD%2FANv4T6EtPEAjrNw43HQ9Q6nTLM8%2BCkhxq6kw8%3D"
#     # "SasToken": "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-12-31T15:38:34Z&st=2025-09-17T07:23:34Z&spr=https&sig=IOG7m4n1ZfNb8SnCOBBURLvSPYKR0qiGjToBH8AmUzs%3D"


#     "ConnectionString": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container",
#     "Container": "dotnetbackend-container",
#     # "SasUrl": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container?sp=r&st=2025-10-14T12:00:0…,
#     "SasToken": "sp=rcwl&st=2025-10-15T15:17:31Z&se=2026-10-15T23:32:31Z&spr=https&sv=2024-11-04&sr=c&sig=Cu7aKTRa3s7Thflh3pr1mGm%2BG8TflhSrdYHD%2FXgxJ80%3D"

# }

# # redis_conn = Redis()
# # q = Queue(connection=redis_conn)



# # # BASE_UPLOAD_DIR = Path("uploads")
# # # BASE_UPLOAD_DIR.mkdir(exist_ok=True)
# # preload_models()


 
# def _coerce_bool(v):
#     if isinstance(v, bool): return v
#     if v is None: return False
#     return str(v).strip().lower() in {"1","true","yes","y","on"}


# def _maybe_json(x):
#     if x is None: return None
#     if isinstance(x, (dict, list)): return x
#     try:
#         return json.loads(x)
#     except Exception:
#         return None


# def _parse_payload_post():
#     """
#     Accept:
#       - application/json: {"records":[...], ...}  OR a bare list [...]
#       - multipart/form-data or x-www-form-urlencoded:
#           * 'records' field as JSON text (recommended)
#           * OR any single field containing a JSON object with 'records', or a JSON list

#     Returns normalized dict: {
#       "records": [...],
#       "species_map": {..} | None,
#       "json_only": bool,
#       "val_fraction": float | None,
#       "base_url": str | None
#     }
#     """
#     payload = None

#     # 1) Proper JSON body
#     if request.is_json:
#         payload = request.get_json(silent=True)

#     # 2) Raw body (sometimes clients mis-set content-type)
#     if payload is None:
#         raw = request.get_data(cache=False, as_text=True)
#         if raw and raw.strip():
#             try:
#                 payload = json.loads(raw)
#             except Exception:
#                 payload = None

#     # 3) Form data
#     if payload is None:
#         # Prefer explicit 'records' if present
#         if "records" in request.form:
#             payload = {
#                 "records": _maybe_json(request.form.get("records")),
#                 "species_map": _maybe_json(request.form.get("species_map")),
#                 "json_only": request.form.get("json_only"),
#                 "val_fraction": request.form.get("val_fraction"),
#                 "base_url": request.form.get("base_url"),
#             }
#         else:
#             # Fallback: try ANY form value that parses to JSON
#             for k, v in request.form.items():
#                 cand = _maybe_json(v)
#                 if isinstance(cand, dict) and "records" in cand:
#                     payload = cand
#                     break
#                 if isinstance(cand, list):
#                     payload = {"records": cand}
#                     break

#     # Normalize
#     if isinstance(payload, list):
#         payload = {"records": payload}
#     if not isinstance(payload, dict):
#         payload = {}

#     rec = payload.get("records")
#     if isinstance(rec, str):
#         rec = _maybe_json(rec)
#     payload["records"] = rec if isinstance(rec, list) else []

#     sm = payload.get("species_map")
#     if isinstance(sm, str):
#         sm = _maybe_json(sm)
#     payload["species_map"] = sm if isinstance(sm, dict) else None

#     payload["json_only"]    = _coerce_bool(payload.get("json_only"))
#     # val_fraction handling
#     try:
#         raw_val = payload.get("val_fraction")
#         if raw_val is None or raw_val == "":
#             payload["val_fraction"] = CFG.VAL_FRACTION
#         else:
#             payload["val_fraction"] = float(raw_val)
#     except Exception:
#         payload["val_fraction"] = CFG.VAL_FRACTION


#     return payload


# def normalize_species_name(species_name: str) -> str:
#     """
#     Replace ':' with ' (' and append ')' at the end if needed.
#     Example: 'Red Drum: Sciaenops ocellatus' -> 'Red Drum (Sciaenops ocellatus)'
#     """
#     if ":" in species_name:
#         parts = species_name.split(":", 1)
#         return f"{parts[0].strip()} ({parts[1].strip()})"
#     return species_name.strip()


# def upload_file_to_blob(species: str, is_handheld: bool, file) -> str:
#     """
#     Upload file to Azure Blob Storage inside Annotated/ folder and return blob URL.
#     """
#     try:
#         container_name = AZURE_CONFIG["container"]

#         # # Folder naming: Annotated/{species}/Hand-Held or Not-Hand-Held
#         # hand_held_folder = "Hand-Held" if is_handheld else "Not-Hand-Held"
#         # extension = os.path.splitext(file.filename)[1]
#         # blob_name = f"Un-Annotated/{species}/{hand_held_folder}/{uuid.uuid4()}{extension}"

#         species_folder = normalize_species_name(species)
#         hand_held_folder = "Hand-Held" if is_handheld else "Not-Hand-Held"
#         extension = os.path.splitext(file.filename)[1]
#         blob_name = f"Annotated/{species_folder}/{hand_held_folder}/{uuid.uuid4()}{extension}"


#         # # Blob client
#         # blob_service_client = BlobServiceClient(account_url=AZURE_CONFIG["connection_string"],
#         #                                         credential=AZURE_CONFIG["sas_token"])
#         # blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

#         # blob_client.upload_blob(file.read(), overwrite=True)

#         # return blob_client.url
    

#         blob_service_client = BlobServiceClient(
#             account_url=AZURE_CONFIG["connection_string"],
#             credential=AZURE_CONFIG["sas_token"]
#         )
#         blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

#         # Upload
#         blob_client.upload_blob(file.read(), overwrite=True)

#         # 👇 Return *clean URL* (no SAS token)
#         return f"{AZURE_CONFIG['connection_string']}/{container_name}/{blob_name}"



#     except Exception as ex:
#         raise RuntimeError(f"Azure upload failed: {str(ex)}")


# import asyncio

# # This is final working version which sends the response to external API
# @fish_bp.route("/build_dataset_from_api", methods=["POST"])
# def build_dataset_from_api():
#     try:
#         payload = _parse_payload_post()
#         print("DEBUG payload:", payload)

#         images = payload.get("images") or []
#         if not isinstance(images, list) or not images:
#             return jsonify({"status": "error", "message": "Missing or invalid 'images' (list)"}), 400

#         defaults = parse_args()

#         # ✅ run async code inside sync route
#         grouped = asyncio.run(process_from_api(
#             payload=payload,
#             yolo_model_path=defaults.yolo_model,
#             dataset_root=defaults.dataset_root,
#             base_dir=defaults.base_dir,
#         ))

#         return jsonify(grouped), 200

#     except ValueError as ve:
#         return jsonify({"status": "error", "message": str(ve)}), 400
#     except Exception as e:
#         logging.exception("Error in dataset build (API payload)")
#         return jsonify({"status": "error", "message": str(e)}), 500



# @fish_bp.route("/dataset/<path:filename>")
# def serve_dataset_file(filename):
#     file_path = CFG.DATASET_ROOT / filename
#     if file_path.exists():
#         return send_file(file_path)
#     return "File not found", 404


# @fish_bp.errorhandler(RequestEntityTooLarge)
# def handle_large_file(e):
#     logging.warning("Upload failed: file too large.")
#     return jsonify({
#         "status": False,
#         "message": "Uploaded image is too large.",
#         "body": None
#     }), 413




# # -------------------------
# # Routes
# # -------------------------

# import re
# # ---- Helper function to sanitize folder names for Windows ----
# def sanitize_filename(name: str) -> str:
#     # Replace invalid Windows filename characters with underscore
#     return re.sub(r'[\\/*?:"<>|]', "_", name)


# # ---- Species ----
# @fish_bp.route("/species", methods=["GET", "POST"])
# def species():
#     if request.method == "POST":
#         data = request.json
#         species_name = data.get("species_name")
#         if not species_name:
#             return jsonify({"error": "species_name is required"}), 400

#         try:
#             conn = sqlite3.connect(DB_PATH_DATA_COLLECTION)
#             cursor = conn.cursor()
#             cursor.execute(
#                 "INSERT INTO species (species_name) VALUES (?)", (species_name,)
#             )
#             conn.commit()
#             species_id = cursor.lastrowid
#             conn.close()
#             return jsonify({"message": f"Species '{species_name}' added", "id": species_id}), 201
#         except sqlite3.IntegrityError:
#             return jsonify({"error": f"Species '{species_name}' already exists"}), 400

#     else:  # GET method
#         conn = sqlite3.connect(DB_PATH_DATA_COLLECTION)
#         cursor = conn.cursor()
#         cursor.execute("SELECT id, species_name FROM species")
#         rows = cursor.fetchall()
#         conn.close()
#         return jsonify([{"id": r[0], "species_name": r[1]} for r in rows])





# training_lock = threading.Lock()
# TRAINING_STATUS = {"state": "idle", "message": "No training running."}



# # @fish_bp.route("/run_pipeline", methods=["POST"])
# # def run_pipeline():
# #     """Run the full training pipeline via Azure ML (v1 SDK)."""
# #     global TRAINING_STATUS

# #     data = request.get_json(silent=True) or {}
# #     species_name = data.get("speciesName")
# #     project_id = data.get("projectId")

# #     # --- Validation ---
# #     if not species_name:
# #         return jsonify({"status": "error", "message": "speciesName is required"}), 400
# #     if not project_id:
# #         return jsonify({"status": "error", "message": "projectId is required"}), 400

# #     if TRAINING_STATUS.get("state") == "running":
# #         return jsonify({
# #             "status": "busy",
# #             "message": "Training is already running. Please wait for it to complete."
# #         }), 429
        



# #     def background_job(species_name: str, project_id: int):
# #         """Background Azure ML job submission using v1 SDK with Service Principal authentication."""
# #         global TRAINING_STATUS

# #         if not training_lock.acquire(blocking=False):
# #             logging.info("⚠️ Another training job is already running — skipping new one.")
# #             return

# #         try:
# #             TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}
# #             logging.info(f"🚀 Starting Azure ML training for project {project_id}, species: {species_name}")

# #             # --- Authenticate using Service Principal ---
# #             try:
# #                 sp_auth = ServicePrincipalAuthentication(
# #                     tenant_id=os.environ["AZURE_TENANT_ID"],
# #                     service_principal_id=os.environ["AZURE_CLIENT_ID"],
# #                     service_principal_password=os.environ["AZURE_CLIENT_SECRET"]
# #                 )
# #                 logging.info("✅ Authenticated successfully with Service Principal.")
# #             except KeyError as e:
# #                 missing_var = str(e)
# #                 logging.error(f"❌ Missing environment variable: {missing_var}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Missing environment variable: {missing_var}"
# #                 }
# #                 return

# #             # --- Connect to Azure ML Workspace ---
# #             try:
# #                 ws = Workspace(
# #                     subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
# #                     resource_group=os.environ["AZURE_RESOURCE_GROUP"],
# #                     workspace_name=os.environ["AZURE_ML_WORKSPACE"],
# #                     auth=sp_auth
# #                 )
# #                 logging.info(f"✅ Connected to Azure ML Workspace: {ws.name}")
# #             except Exception as ws_error:
# #                 logging.error(f"❌ Failed to connect to Azure ML Workspace: {ws_error}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Workspace connection failed: {ws_error}"
# #                 }
# #                 return

# #             # --- Get compute target ---
# #             try:
# #                 compute_target = ComputeTarget(workspace=ws, name="GflTrainCompute")
# #                 logging.info("✅ Compute target 'GflTrainCompute' found.")
# #             except Exception as compute_error:
# #                 logging.error(f"❌ Failed to locate compute target: {compute_error}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Compute target not found: {compute_error}"
# #                 }
# #                 return

# #             # --- Get environment ---
# #             try:
# #                 env = ws.environments["gflTrainEnv"]
# #                 logging.info("✅ Environment 'gflTrainEnv' loaded successfully.")
# #             except KeyError:
# #                 logging.error("❌ Environment 'gflTrainEnv' not found in workspace.")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": "Environment 'gflTrainEnv' not found in workspace."
# #                 }
# #                 return

# #             # --- Access datastore ---
# #             try:
# #                 logging.info("🔍 Retrieving workspace datastore...")
# #                 datastore = Datastore.get(ws, "workspaceblobstore")
# #                 logging.info(f"✅ Datastore retrieved: {datastore.name}")
# #             except Exception as ds_error:
# #                 logging.error(f"❌ Failed to access datastore: {ds_error}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Datastore access failed: {ds_error}"
# #                 }
# #                 return

# #             # --- Define job source and command ---
# #             source_directory = "UI/JobSubmission/10-23-2025_102845_UTC"
# #             logging.info(f"📁 Source directory: {source_directory}")

# #             command = [
# #                 "python", "train_pipeline_entry.py",
# #                 "--species", species_name,
# #                 "--project_id", str(project_id)
# #             ]
# #             logging.info(f"⚙️ Training command: {command}")

# #             # --- Create ScriptRunConfig ---
# #             try:
# #                 config = ScriptRunConfig(
# #                     source_directory=source_directory,
# #                     command=command,
# #                     compute_target=compute_target,
# #                     environment=env,
# #                 )
# #                 logging.info("✅ ScriptRunConfig created successfully.")
# #             except Exception as config_error:
# #                 logging.error(f"❌ Failed to create ScriptRunConfig: {config_error}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"ScriptRunConfig creation failed: {config_error}"
# #                 }
# #                 return

# #             # --- Submit experiment ---
# #             try:
# #                 exp = Experiment(ws, "GFLTrainingModelV2")
# #                 logging.info("🚀 Submitting job to Azure ML experiment...")
# #                 run = exp.submit(config)
# #                 run_url = run.get_portal_url()
# #                 logging.info(f"✅ Job submitted successfully. Portal: {run_url}")

# #                 TRAINING_STATUS = {
# #                     "state": "submitted",
# #                     "message": "Azure ML job submitted successfully.",
# #                     "run_id": run.id,
# #                     "portal_url": run_url
# #                 }

# #             except Exception as submit_error:
# #                 logging.error(f"❌ Experiment submission failed: {submit_error}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Experiment submission failed: {submit_error}"
# #                 }

# #         except Exception as e:
# #             logging.exception("❌ Unhandled error during Azure ML job submission.")
# #             TRAINING_STATUS = {"state": "failed", "message": str(e)}

# #         finally:
# #             training_lock.release()


# #     # --- Launch background job ---
# #     thread = threading.Thread(target=background_job, args=(species_name, project_id), daemon=True)
# #     thread.start()

# #     return jsonify({
# #         "status": "started",
# #         "message": f"Training job submitted for project {project_id}, species {species_name}.",
# #         "check_status_at": "/pipeline_status"
# #     }), 202


# # @fish_bp.route("/run_pipeline", methods=["POST"])
# # def run_pipeline():
# #     """Run the full training pipeline via Azure ML."""
# #     global TRAINING_STATUS

# #     data = request.get_json(silent=True) or {}
# #     species_name = data.get("speciesName")
# #     project_id = data.get("projectId")

# #     # --- Basic validation ---
# #     if not species_name:
# #         return jsonify({"status": "error", "message": "speciesName is required"}), 400
# #     if not project_id:
# #         return jsonify({"status": "error", "message": "projectId is required"}), 400

# #     if TRAINING_STATUS.get("state") == "running":
# #         return jsonify({
# #             "status": "busy",
# #             "message": "Training is already running. Please wait for it to complete."
# #         }), 429

# #     # --- Background worker for non-blocking response ---
# #     def background_job(species_name: str, project_id: int):
# #         global TRAINING_STATUS
# #         lock_acquired = training_lock.acquire(blocking=False)

# #         if not lock_acquired:
# #             print("⚠️ Another training job is already running — skipping new one.")
# #             logging.info("⚠️ Another training job is already running — skipping new one.")
# #             return

# #         try:
# #             TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}
# #             print(f"🚀 Submitting Azure ML training for project {project_id}, species: {species_name}")
# #             logging.info(f"🚀 Submitting Azure ML training for project {project_id}, species: {species_name}")

# #             # --- Authenticate to Azure ML ---
# #             credential = DefaultAzureCredential()
# #             ml_client = MLClient(
# #                 credential=credential,
# #                 subscription_id=AZURE_SUBSCRIPTION_ID,
# #                 resource_group_name=AZURE_RESOURCE_GROUP,
# #                 workspace_name=AZURE_ML_WORKSPACE,
# #             )

# #             # compute_name = "GflTrainCompute"
# #             # compute_name = "gflTrainComputeV1"
# #             compute_name = "gflTrainCluster"
# #             print("my new Compute cluster is: ", compute_name)
# #             logging.info("my new Compute cluster is: ", compute_name)
            
# #             # --- Ensure compute exists and is running ---
# #             try:
# #                 compute = ml_client.compute.get(compute_name)
# #                 state = compute.properties.get("provisioning_state", None)
# #                 print(f"ℹ️ Compute '{compute_name}' current state: {state}")
# #                 logging.info(f"ℹ️ Compute '{compute_name}' current state: {state}")
# #                 #ml_client.compute.begin_start(compute_name).wait()
# #                 print(f"✅ Compute '{compute.provisioning_state}' is starting...")

# #                 # if compute.provisioning_state == 'Stopped':
# #                 #     print(f"⚠️ Compute '{compute_name}' is stopped. Starting it now...")
# #                 #     ml_client.compute.begin_start(compute_name)
# #                 #     print(f"✅ Compute '{compute_name}' is starting...")
                
# #                 if state in ["Stopped", "Deallocated", "Unknown", "Stopping", "None"]:
# #                     print(f"⚠️ Compute '{compute_name}' is not running. Starting now...")
# #                     logging.info(f"⚠️ Compute '{compute_name}' is not running. Starting now...")
# #                     #poller = ml_client.compute.begin_start(compute_name)
# #                     #poller.wait()  # Wait for start to complete
# #                     print(f"✅ Compute '{compute_name}' has started successfully.")
# #                 else:
# #                     logging.info(f"✅ Compute '{compute_name}' is already running.")


# #             except Exception as e:
# #                 print(f"❌ Error fetching compute target '{compute_name}': {e}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Compute target '{compute_name}' not found or failed to start in Azure ML workspace."
# #                 }
# #                 training_lock.release()
# #                 return

# #             # --- Access datastore dynamically ---
# #             try:
# #                 logging.info("🔍 Retrieving workspace datastore...")
# #                 # Connect using AzureML v1 to get blob path
# #                 # ✅ Fetch datastore directly via MLClient
# #                 datastore = ml_client.datastores.get("workspaceblobstore")
# #                 print(f"✅ Datastore retrieved: {datastore.name}")
# #                 logging.info(f"✅ Datastore retrieved: {datastore.name}")
                
# #                 # Construct URI safely
# #                 source_path = "UI/JobSubmission/10-23-2025_102845_UTC"
# #                 datastore_uri = f"azureml://subscriptions/417ecdbc-9b50-421a-82c2-c45f29686df1/resourcegroups/GFL-App-Rg/workspaces/GFL-MLAI/datastores/workspaceblobstore/paths/{source_path}/"
# #                 print(f"📁 Using datastore path: {datastore_uri}")
# #                 logging.info(f"📁 Using datastore path: {datastore_uri}")

# #                 logging.info(f"✅ Datastore retrieved: {datastore.name}")

# #                 # Define source directory path inside blob
# #                 # source_path = "UI/JobSubmission/10-23-2025_102845_UTC"
# #                 # datastore_uri = f"azureml://datastores/{datastore.name}/paths/{source_path}/"
# #                 # logging.info(f"📁 Using datastore path: {datastore_uri}")

# #             except Exception as ds_error:
# #                 logging.info(f"❌ Datastore access failed: {ds_error}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Datastore access failed: {ds_error}"
# #                 }
# #                 training_lock.release()
# #                 return
# #             # Get the data asset for the registered code (as you have already done)
# #             data_asset = ml_client.data.get("training_pipeline", version="2")
# #             print("Data assets path code folder",data_asset.path)
# #             code_input = Input(
# #                             type="uri_folder",
# #                             path="azureml://subscriptions/417ecdbc-9b50-421a-82c2-c45f29686df1/"
# #                                  "resourcegroups/GFL-App-Rg/workspaces/GFL-MLAI/"
# #                                  "datastores/workspaceblobstore/paths/UI/JobSubmission/10-23-2025_102845_UTC/"
# #                         )

# #             # --- Define and submit the job ---
# #             # job = command(
# #             #     # code= ("azureml://subscriptions/417ecdbc-9b50-421a-82c2-c45f29686df1/"
# #             #     #         "resourcegroups/GFL-App-Rg/workspaces/GFL-MLAI/"
# #             #     #         "datastores/workspaceblobstore/paths/UI/JobSubmission/10-23-2025_102845_UTC/"
# #             #     # ),
# #             #     code=".",  # 🔹 Don't upload anything from App Service
# #             #     inputs={"code_dir": code_input},  # 🔹 Mount your blob folder into the compute
# #             #     command=(
# #             #         f"pip install -r requirements.txt && "
# #             #         f"python train_pipeline_entry.py --species '{species_name}' --project_id {project_id}"
# #             #     ),
# #             #     environment="gflTrainEnv:5",

# #             #     # environment=(
# #             #     #     "azureml://locations/eastus2/workspaces/"
# #             #     #     "bbd3a4dc-14ed-4039-80c7-7d0d2df1bd36/"
# #             #     #     "environments/gflTrainEnv/versions/5"
# #             #     # ),
# #             #     compute=compute_name,
# #             #     experiment_name="GFLTrainingModelV1",
# #             #     display_name=f"fish_train_{species_name}_{project_id}",
# #             #     description="YOLO fish species training job"
# #             # )
            

# #             job = command(
# #                 code=None,
# #                 inputs={"code_dir": code_input},
# #                 command=(
# #                     "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 &&"
# #                     "pip install -r ${{inputs.code_dir}}/requirements.txt && "
# #                     "python ${{inputs.code_dir}}/train_pipeline_entry.py "
# #                     f"--species '{species_name}' --project_id {project_id}"
# #                 ),
# #                 # ✅ Correct environment format
# #                 environment="gflTrainEnv:5",
# #                 compute=compute_name,
# #                 experiment_name="GFLTrainingModelV1",
# #                 # experiment_name = "gflTrainCluster",
# #                 display_name=f"fish_train_{species_name}_{project_id}",
# #                 description="YOLO fish species training job"
# #             )



# #             submitted_job = ml_client.jobs.create_or_update(job)
            
# #             TRAINING_STATUS = {
# #                 "state": "submitted",
# #                 "message": f"✅ Azure ML job submitted: {submitted_job.name}",
# #                 "job_name": submitted_job.name
# #             }
# #             logging.info(f"✅ Submitted Azure ML job: {submitted_job.name}")

# #         except Exception as e:
# #             TRAINING_STATUS = {
# #                 "state": "failed",
# #                 "message": f"❌ Azure ML job submission failed: {str(e)}"
# #             }
# #             logging.info(f"❌ Azure ML submission failed: {e}")

# #         # finally:
# #         #     if lock_acquired:
# #         #         training_lock.release()
                
# #         finally:
# #             # Always release safely
# #             if training_lock.locked():
# #                 training_lock.release()


# #     # --- Run background thread ---
# #     thread = threading.Thread(target=background_job, args=(species_name, project_id), daemon=True)
# #     thread.start()

# #     # Immediate HTTP response
# #     return jsonify({
# #         "status": "started",
# #         "message": f"Training job submitted for project {project_id}, species {species_name}.",
# #         "check_status_at": "/pipeline_status"
# #     }), 202


# # # Old work
# # @fish_bp.route("/run_pipeline", methods=["POST"])
# # def run_pipeline():
# #     """Run the full training pipeline via Azure ML."""
# #     global TRAINING_STATUS

# #     data = request.get_json(silent=True) or {}
# #     species_name = data.get("speciesName")
# #     project_id = data.get("projectId")

# #     # --- Basic validation ---
# #     if not species_name:
# #         return jsonify({"status": "error", "message": "speciesName is required"}), 400
# #     if not project_id:
# #         return jsonify({"status": "error", "message": "projectId is required"}), 400

# #     if TRAINING_STATUS.get("state") == "running":
# #         return jsonify({
# #             "status": "busy",
# #             "message": "Training is already running. Please wait for it to complete."
# #         }), 429

# #     # --- Background worker for non-blocking response ---
# #     def background_job(species_name: str, project_id: int):
# #         global TRAINING_STATUS
# #         lock_acquired = training_lock.acquire(blocking=False)

# #         if not lock_acquired:
# #             print("⚠️ Another training job is already running — skipping new one.")
# #             logging.info("⚠️ Another training job is already running — skipping new one.")
# #             return

# #         try:
# #             TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}
# #             print(f"🚀 Submitting Azure ML training for project {project_id}, species: {species_name}")
# #             logging.info(f"🚀 Submitting Azure ML training for project {project_id}, species: {species_name}")

# #             # --- Authenticate to Azure ML ---
# #             credential = DefaultAzureCredential()
# #             ml_client = MLClient(
# #                 credential=credential,
# #                 subscription_id=AZURE_SUBSCRIPTION_ID,
# #                 resource_group_name=AZURE_RESOURCE_GROUP,
# #                 workspace_name=AZURE_ML_WORKSPACE,
# #             )

# #             # compute_name = "GflTrainCompute"
# #             # compute_name = "gflTrainComputeV1"
# #             compute_name = "gflTrainCluster"
# #             print("my new Compute cluster is: ", compute_name)
# #             logging.info("my new Compute cluster is: ", compute_name)
            
# #             # --- Ensure compute exists and is running ---
# #             try:
# #                 compute = ml_client.compute.get(compute_name)
# #                 state = compute.properties.get("provisioning_state", None)
# #                 print(f"ℹ️ Compute '{compute_name}' current state: {state}")
# #                 logging.info(f"ℹ️ Compute '{compute_name}' current state: {state}")
# #                 #ml_client.compute.begin_start(compute_name).wait()
# #                 print(f"✅ Compute '{compute.provisioning_state}' is starting...")

# #                 # if compute.provisioning_state == 'Stopped':
# #                 #     print(f"⚠️ Compute '{compute_name}' is stopped. Starting it now...")
# #                 #     ml_client.compute.begin_start(compute_name)
# #                 #     print(f"✅ Compute '{compute_name}' is starting...")
                
# #                 if state in ["Stopped", "Deallocated", "Unknown", "Stopping", "None"]:
# #                     print(f"⚠️ Compute '{compute_name}' is not running. Starting now...")
# #                     logging.info(f"⚠️ Compute '{compute_name}' is not running. Starting now...")
# #                     #poller = ml_client.compute.begin_start(compute_name)
# #                     #poller.wait()  # Wait for start to complete
# #                     print(f"✅ Compute '{compute_name}' has started successfully.")
# #                 else:
# #                     logging.info(f"✅ Compute '{compute_name}' is already running.")


# #             except Exception as e:
# #                 print(f"❌ Error fetching compute target '{compute_name}': {e}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Compute target '{compute_name}' not found or failed to start in Azure ML workspace."
# #                 }
# #                 training_lock.release()
# #                 return

# #             # --- Access datastore dynamically ---
# #             try:
# #                 logging.info("🔍 Retrieving workspace datastore...")
# #                 # Connect using AzureML v1 to get blob path
# #                 # ✅ Fetch datastore directly via MLClient
# #                 datastore = ml_client.datastores.get("workspaceblobstore")
# #                 print(f"✅ Datastore retrieved: {datastore.name}")
# #                 logging.info(f"✅ Datastore retrieved: {datastore.name}")
                
# #                 # Construct URI safely
# #                 source_path = "UI/JobSubmission/10-23-2025_102845_UTC"
# #                 datastore_uri = f"azureml://subscriptions/417ecdbc-9b50-421a-82c2-c45f29686df1/resourcegroups/GFL-App-Rg/workspaces/GFL-MLAI/datastores/workspaceblobstore/paths/{source_path}/"
# #                 print(f"📁 Using datastore path: {datastore_uri}")
# #                 logging.info(f"📁 Using datastore path: {datastore_uri}")

# #                 logging.info(f"✅ Datastore retrieved: {datastore.name}")

# #                 # Define source directory path inside blob
# #                 # source_path = "UI/JobSubmission/10-23-2025_102845_UTC"
# #                 # datastore_uri = f"azureml://datastores/{datastore.name}/paths/{source_path}/"
# #                 # logging.info(f"📁 Using datastore path: {datastore_uri}")

# #             except Exception as ds_error:
# #                 logging.info(f"❌ Datastore access failed: {ds_error}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Datastore access failed: {ds_error}"
# #                 }
# #                 training_lock.release()
# #                 return
# #             # Get the data asset for the registered code (as you have already done)
# #             data_asset = ml_client.data.get("training_pipeline", version="2")
# #             print("Data assets path code folder",data_asset.path)
# #             code_input = Input(
# #                             type="uri_folder",
# #                             path="azureml://subscriptions/417ecdbc-9b50-421a-82c2-c45f29686df1/"
# #                                  "resourcegroups/GFL-App-Rg/workspaces/GFL-MLAI/"
# #                                  "datastores/workspaceblobstore/paths/UI/JobSubmission/10-23-2025_102845_UTC/"
# #                         )

# #             # --- Define and submit the job ---
# #             # job = command(
# #             #     # code= ("azureml://subscriptions/417ecdbc-9b50-421a-82c2-c45f29686df1/"
# #             #     #         "resourcegroups/GFL-App-Rg/workspaces/GFL-MLAI/"
# #             #     #         "datastores/workspaceblobstore/paths/UI/JobSubmission/10-23-2025_102845_UTC/"
# #             #     # ),
# #             #     code=".",  # 🔹 Don't upload anything from App Service
# #             #     inputs={"code_dir": code_input},  # 🔹 Mount your blob folder into the compute
# #             #     command=(
# #             #         f"pip install -r requirements.txt && "
# #             #         f"python train_pipeline_entry.py --species '{species_name}' --project_id {project_id}"
# #             #     ),
# #             #     environment="gflTrainEnv:5",

# #             #     # environment=(
# #             #     #     "azureml://locations/eastus2/workspaces/"
# #             #     #     "bbd3a4dc-14ed-4039-80c7-7d0d2df1bd36/"
# #             #     #     "environments/gflTrainEnv/versions/5"
# #             #     # ),
# #             #     compute=compute_name,
# #             #     experiment_name="GFLTrainingModelV1",
# #             #     display_name=f"fish_train_{species_name}_{project_id}",
# #             #     description="YOLO fish species training job"
# #             # )
            

# #             job = command(
# #                 code=None,
# #                 inputs={"code_dir": code_input},
# #                 command=(
# #                     "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 &&"
# #                     "pip install -r ${{inputs.code_dir}}/requirements.txt && "
# #                     "python ${{inputs.code_dir}}/train_pipeline_entry.py "
# #                     f"--species '{species_name}' --project_id {project_id}"
# #                 ),
# #                 # ✅ Correct environment format
# #                 environment="gflTrainEnv:5",
# #                 compute=compute_name,
# #                 experiment_name="GFLTrainingModelV1",
# #                 # experiment_name = "gflTrainCluster",
# #                 display_name=f"fish_train_{species_name}_{project_id}",
# #                 description="YOLO fish species training job",
# #                 environment_variables={   # ✅ Add these lines
# #                     "BLOB_BASE_URL": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container",
# #                     "BLOB_CONTAINER": "dotnetbackend-container",
# #                     "BLOB_SAS_TOKEN": "sp=rcwl&st=2025-10-15T15:17:31Z&se=2026-10-15T23:32:31Z&spr=https&sv=2024-11-04&sr=c&sig=Cu7aKTRa3s7Thflh3pr1mGm%2BG8TflhSrdYHD%2FXgxJ80%3D"
# #                 }
# #             )



# #             submitted_job = ml_client.jobs.create_or_update(job)
            
# #             TRAINING_STATUS = {
# #                 "state": "submitted",
# #                 "message": f"✅ Azure ML job submitted: {submitted_job.name}",
# #                 "job_name": submitted_job.name
# #             }
# #             logging.info(f"✅ Submitted Azure ML job: {submitted_job.name}")

# #         except Exception as e:
# #             TRAINING_STATUS = {
# #                 "state": "failed",
# #                 "message": f"❌ Azure ML job submission failed: {str(e)}"
# #             }
# #             logging.info(f"❌ Azure ML submission failed: {e}")

# #         # finally:
# #         #     if lock_acquired:
# #         #         training_lock.release()
                
# #         finally:
# #             # Always release safely
# #             if training_lock.locked():
# #                 training_lock.release()


# #     # --- Run background thread ---
# #     thread = threading.Thread(target=background_job, args=(species_name, project_id), daemon=True)
# #     thread.start()

# #     # Immediate HTTP response
# #     return jsonify({
# #         "status": "started",
# #         "message": f"Training job submitted for project {project_id}, species {species_name}.",
# #         "check_status_at": "/pipeline_status"
# #     }), 202



# # @fish_bp.route("/run_pipeline", methods=["POST"])
# # def run_pipeline():
# #     """Run the full training pipeline in the background."""
# #     global TRAINING_STATUS

# #     data = request.get_json(silent=True) or {}

# #     if not data:
# #         return jsonify({"status": "error", "message": "Input data is required"}), 400

# #     # Normalize input into a list
# #     # If frontend sends a single object → wrap into list
# #     if isinstance(data, dict):
# #         data = [data]

# #     # Validate each entry
# #     for entry in data:
# #         species_name = entry.get("speciesName")
# #         project_id = entry.get("projectId")
# #         if not species_name or not project_id:
# #             return jsonify({
# #                 "status": "error",
# #                 "message": "Both speciesName and projectId are required for each entry"
# #             }), 400

# #     # Prevent simultaneous runs
# #     if TRAINING_STATUS["state"] == "running":
# #         return jsonify({
# #             "status": "busy",
# #             "message": "Training is already running. Please wait for it to complete."
# #         }), 429

# #     # ------------------ Background Job ------------------
# #     def background_job(entries: list):
# #         global TRAINING_STATUS
# #         if not training_lock.acquire(blocking=False):
# #             print("⚠️ Another training job is already running — skipping new one.")
# #             return

# #         try:
# #             TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}

# #             for entry in entries:
# #                 species_name = entry["speciesName"]
# #                 project_id = entry["projectId"]

# #                 print(f"🚀 Background training started for project {project_id}, species: {species_name}")

# #                 os.environ["PROJECT_ID"] = str(project_id)

# #                 # Run training sequence
# #                 asyncio.run(build_augmented_dataset(species_name=species_name))
# #                 task1_run()
# #                 task2_run()
# #                 generate_filtered_group_dataset()
# #                 train_all_groups()

# #                 print(f"✅ Training completed for project {project_id}, species {species_name}")

# #             TRAINING_STATUS = {"state": "completed",
# #                                "message": "✅ Training completed successfully."}

# #         except Exception as e:
# #             TRAINING_STATUS = {
# #                 "state": "failed",
# #                 "message": f"❌ Training pipeline failed: {str(e)}"
# #             }
# #             print(f"❌ Training failed: {e}")

# #         finally:
# #             training_lock.release()

# #     # Launch thread
# #     thread = threading.Thread(target=background_job, args=(data,), daemon=True)
# #     thread.start()

# #     return jsonify({
# #         "status": "started",
# #         "message": f"Training pipeline started for {len(data)} species.",
# #         "check_status_at": "/pipeline_status"
# #     }), 202

# @fish_bp.route("/run_pipeline", methods=["POST"])
# def run_pipeline():
#     """Run the full multi-species training pipeline in the background."""
#     global TRAINING_STATUS

#     data = request.get_json(silent=True) or {}

#     if not data:
#         return jsonify({"status": "error", "message": "Input data is required"}), 400

#     # Normalize input into a list
#     if isinstance(data, dict):
#         data = [data]

#     # Validate each entry
#     for entry in data:
#         species_name = entry.get("speciesName")
#         project_id = entry.get("projectId")
#         if not species_name or project_id is None:
#             return (
#                 jsonify(
#                     {
#                         "status": "error",
#                         "message": "Both speciesName and projectId are required for each entry",
#                     }
#                 ),
#                 400,
#             )

#     # Optionally enforce same projectId for all species
#     project_ids = {str(e["projectId"]) for e in data}
#     if len(project_ids) > 1:
#         return (
#             jsonify(
#                 {
#                     "status": "error",
#                     "message": "All species in one run must belong to the same projectId",
#                 }
#             ),
#             400,
#         )

#     project_id = project_ids.pop()

#     # Prevent simultaneous runs
#     if TRAINING_STATUS.get("state") == "running":
#         return (
#             jsonify(
#                 {
#                     "status": "busy",
#                     "message": "Training is already running. Please wait for it to complete.",
#                 }
#             ),
#             429,
#         )

#     def background_job(entries: list):
#         global TRAINING_STATUS

#         if not training_lock.acquire(blocking=False):
#             logger.warning("⚠️ Another training job is already running — skipping new one.")
#             return

#         try:
#             TRAINING_STATUS = {
#                 "state": "running",
#                 "message": "Training in progress...",
#                 "results": [],
#             }

#             logger.info(
#                 f"🚀 Background training started for project {project_id} with {len(entries)} species"
#             )

#             # Set project ID once for the whole run (or better: pass it to tasks explicitly)
#             os.environ["PROJECT_ID"] = str(project_id)

#             # 1) Build multi-species dataset (one pass)
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             loop.run_until_complete(build_augmented_dataset_multi(entries))
#             loop.close()

#             # 2) Run your downstream training stages ONCE on the combined dataset
#             task1_run()
#             task2_run()
#             generate_filtered_group_dataset()
#             train_all_groups()

#             logger.info(
#                 f"✅ Training completed for project {project_id} with {len(entries)} species"
#             )

#             TRAINING_STATUS = {
#                 "state": "completed",
#                 "message": "✅ Training completed successfully.",
#             }

#         except Exception as e:
#             TRAINING_STATUS = {
#                 "state": "failed",
#                 "message": f"❌ Training pipeline failed: {str(e)}",
#             }
#             logger.exception(f"❌ Training failed: {e}")

#         finally:
#             training_lock.release()

#     # Launch thread
#     thread = threading.Thread(target=background_job, args=(data,), daemon=True)
#     thread.start()

#     return (
#         jsonify(
#             {
#                 "status": "started",
#                 "message": f"Training pipeline started for {len(data)} species.",
#                 "check_status_at": "/pipeline_status",
#             }
#         ),
#         202,
#     )

# @fish_bp.route("/pipeline_status", methods=["GET"])
# def pipeline_status():
#     """Check current training pipeline status."""
#     return jsonify(TRAINING_STATUS), 200



























# from flask import Blueprint, request, jsonify, current_app, url_for, send_file
# # from app_backend.services.detect import detect_fish, detect_fish_species, get_top_bbox_yolo_format
# # from app_backend.services.measure import measure_fish, measure_fish_v2
# # from app_backend.database.db import is_duplicate, insert_fish, is_duplicate_v2  # Import DB functions
# import base64
# import cv2
# import numpy as np
# import uuid
# import os
# import time
# from datetime import datetime
# # from PIL import Image
# import logging
# from werkzeug.exceptions import RequestEntityTooLarge
# from app_backend.config import IMG_SIZE, DB_FOLDER, DATASET_ROOT, BASE_PUBLIC_URL, DB_FOLDER_Unknown_Fish, LOCAL_PREDICTION_MODELS_DIR, PREDICTED_OUTPUT_DIR
# from app_backend.services.uniqueness import compare_two_images
# from app_backend.utils.helpers import ensure_db_folder
# from app_backend.services.fish_sqlite import save_image_file, BASE_DIR 
# from app_backend.database.db import DB_PATH_DATA_COLLECTION, insert_unknown_fish
# import sqlite3
# from pathlib import Path
# import json
# import base64
# from io import BytesIO
# import requests
# import asyncio
# import threading
# from flask import Flask, request, jsonify
# from ultralytics import YOLO
# import os
# import requests
# from datetime import datetime
# # from PIL import Image
# import io

# # For multimodel training

# # from pathlib import Path
# # from flask import current_app
# # from app_backend.config import DATASET_ROOT, BASE_PUBLIC_URL
# # Fish Annotation (builder)
# # from app_backend.modules.fish_annotation.src.pipeline import process_from_api, group_records_for_json
# # from app_backend.modules.fish_annotation.src.cli import parse_args
# # from app_backend.modules.fish_annotation.src.config import CFG

# # Fish Training (pipeline + inference)
# # from app_backend.modules.fish_training.parse_dataset import task1_run
# # from app_backend.modules.fish_training.group_species import task2_run
# # from app_backend.modules.fish_training.filter_dataset import generate_filtered_group_dataset
# # from app_backend.modules.fish_training.train_models import train_all_groups
# # from app_backend.modules.fish_training.multi_model_predict import predict_best_multithread, preload_models
# # from app_backend.modules.fish_training.augment_pipeline import build_augmented_dataset
# # from app_backend.database.db import insert_predicted
# from app_backend.config import LOCAL_PREDICTION_MODELS_DIR, AZURE_ML_WORKSPACE, AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP

# import json
# from flask import request

# from pathlib import Path
# import os
# from azure.storage.blob import BlobServiceClient
# from ultralytics import YOLO

# from redis import Redis
# from rq import Queue
# from rq.job import Job
# # from app_backend.modules.fish_training.jobs import run_training_pipeline
# # from rq import Worker, Queue
# # from rq.connections import Connection
# from azure.identity import DefaultAzureCredential
# from azure.ai.ml import MLClient, command
# import os
# # from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, ComputeTarget, Datastore
# from azureml.core import Workspace, Experiment, Datastore, ScriptRunConfig, Environment, ComputeTarget
# from azureml.core.authentication import ServicePrincipalAuthentication

# fish_bp = Blueprint("fish_bp", __name__)


# from azure.storage.blob import BlobServiceClient

# # -------------------- Azure Config --------------------
# AZURE_CONFIG = {
#     # "ConnectionString": "https://gflstorageaccount.blob.core.windows.net/dotnetbackend-container",
#     # "Container": "dotnetbackend-container",
#     # # "SasToken": "sp=rcwd&st=2025-09-08T09:08:53Z&se=2026-09-08T17:23:53Z&spr=https&sv=2024-11-04&sr=c&sig=%2FPkcD%2FANv4T6EtPEAjrNw43HQ9Q6nTLM8%2BCkhxq6kw8%3D"
#     # "SasToken": "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-12-31T15:38:34Z&st=2025-09-17T07:23:34Z&spr=https&sig=IOG7m4n1ZfNb8SnCOBBURLvSPYKR0qiGjToBH8AmUzs%3D"


#     "ConnectionString": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container",
#     "Container": "dotnetbackend-container",
#     # "SasUrl": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container?sp=r&st=2025-10-14T12:00:0…,
#     "SasToken": "sp=rcwl&st=2025-10-15T15:17:31Z&se=2026-10-15T23:32:31Z&spr=https&sv=2024-11-04&sr=c&sig=Cu7aKTRa3s7Thflh3pr1mGm%2BG8TflhSrdYHD%2FXgxJ80%3D"

# }

# # redis_conn = Redis()
# # q = Queue(connection=redis_conn)



# # # BASE_UPLOAD_DIR = Path("uploads")
# # # BASE_UPLOAD_DIR.mkdir(exist_ok=True)
# # preload_models()


 
# def _coerce_bool(v):
#     if isinstance(v, bool): return v
#     if v is None: return False
#     return str(v).strip().lower() in {"1","true","yes","y","on"}


# def _maybe_json(x):
#     if x is None: return None
#     if isinstance(x, (dict, list)): return x
#     try:
#         return json.loads(x)
#     except Exception:
#         return None


# def _parse_payload_post():
#     """
#     Accept:
#       - application/json: {"records":[...], ...}  OR a bare list [...]
#       - multipart/form-data or x-www-form-urlencoded:
#           * 'records' field as JSON text (recommended)
#           * OR any single field containing a JSON object with 'records', or a JSON list

#     Returns normalized dict: {
#       "records": [...],
#       "species_map": {..} | None,
#       "json_only": bool,
#       "val_fraction": float | None,
#       "base_url": str | None
#     }
#     """
#     payload = None

#     # 1) Proper JSON body
#     if request.is_json:
#         payload = request.get_json(silent=True)

#     # 2) Raw body (sometimes clients mis-set content-type)
#     if payload is None:
#         raw = request.get_data(cache=False, as_text=True)
#         if raw and raw.strip():
#             try:
#                 payload = json.loads(raw)
#             except Exception:
#                 payload = None

#     # 3) Form data
#     if payload is None:
#         # Prefer explicit 'records' if present
#         if "records" in request.form:
#             payload = {
#                 "records": _maybe_json(request.form.get("records")),
#                 "species_map": _maybe_json(request.form.get("species_map")),
#                 "json_only": request.form.get("json_only"),
#                 "val_fraction": request.form.get("val_fraction"),
#                 "base_url": request.form.get("base_url"),
#             }
#         else:
#             # Fallback: try ANY form value that parses to JSON
#             for k, v in request.form.items():
#                 cand = _maybe_json(v)
#                 if isinstance(cand, dict) and "records" in cand:
#                     payload = cand
#                     break
#                 if isinstance(cand, list):
#                     payload = {"records": cand}
#                     break

#     # Normalize
#     if isinstance(payload, list):
#         payload = {"records": payload}
#     if not isinstance(payload, dict):
#         payload = {}

#     rec = payload.get("records")
#     if isinstance(rec, str):
#         rec = _maybe_json(rec)
#     payload["records"] = rec if isinstance(rec, list) else []

#     sm = payload.get("species_map")
#     if isinstance(sm, str):
#         sm = _maybe_json(sm)
#     payload["species_map"] = sm if isinstance(sm, dict) else None

#     payload["json_only"]    = _coerce_bool(payload.get("json_only"))
#     # val_fraction handling
#     try:
#         raw_val = payload.get("val_fraction")
#         if raw_val is None or raw_val == "":
#             payload["val_fraction"] = CFG.VAL_FRACTION
#         else:
#             payload["val_fraction"] = float(raw_val)
#     except Exception:
#         payload["val_fraction"] = CFG.VAL_FRACTION


#     return payload


# def normalize_species_name(species_name: str) -> str:
#     """
#     Replace ':' with ' (' and append ')' at the end if needed.
#     Example: 'Red Drum: Sciaenops ocellatus' -> 'Red Drum (Sciaenops ocellatus)'
#     """
#     if ":" in species_name:
#         parts = species_name.split(":", 1)
#         return f"{parts[0].strip()} ({parts[1].strip()})"
#     return species_name.strip()


# def upload_file_to_blob(species: str, is_handheld: bool, file) -> str:
#     """
#     Upload file to Azure Blob Storage inside Annotated/ folder and return blob URL.
#     """
#     try:
#         container_name = AZURE_CONFIG["container"]

#         # # Folder naming: Annotated/{species}/Hand-Held or Not-Hand-Held
#         # hand_held_folder = "Hand-Held" if is_handheld else "Not-Hand-Held"
#         # extension = os.path.splitext(file.filename)[1]
#         # blob_name = f"Un-Annotated/{species}/{hand_held_folder}/{uuid.uuid4()}{extension}"

#         species_folder = normalize_species_name(species)
#         hand_held_folder = "Hand-Held" if is_handheld else "Not-Hand-Held"
#         extension = os.path.splitext(file.filename)[1]
#         blob_name = f"Annotated/{species_folder}/{hand_held_folder}/{uuid.uuid4()}{extension}"


#         # # Blob client
#         # blob_service_client = BlobServiceClient(account_url=AZURE_CONFIG["connection_string"],
#         #                                         credential=AZURE_CONFIG["sas_token"])
#         # blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

#         # blob_client.upload_blob(file.read(), overwrite=True)

#         # return blob_client.url
    

#         blob_service_client = BlobServiceClient(
#             account_url=AZURE_CONFIG["connection_string"],
#             credential=AZURE_CONFIG["sas_token"]
#         )
#         blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

#         # Upload
#         blob_client.upload_blob(file.read(), overwrite=True)

#         # 👇 Return *clean URL* (no SAS token)
#         return f"{AZURE_CONFIG['connection_string']}/{container_name}/{blob_name}"



#     except Exception as ex:
#         raise RuntimeError(f"Azure upload failed: {str(ex)}")


# import asyncio

# # # This is final working version which sends the response to external API
# # @fish_bp.route("/build_dataset_from_api", methods=["POST"])
# # def build_dataset_from_api():
# #     try:
# #         payload = _parse_payload_post()
# #         print("DEBUG payload:", payload)

# #         images = payload.get("images") or []
# #         if not isinstance(images, list) or not images:
# #             return jsonify({"status": "error", "message": "Missing or invalid 'images' (list)"}), 400

# #         defaults = parse_args()

# #         # ✅ run async code inside sync route
# #         grouped = asyncio.run(process_from_api(
# #             payload=payload,
# #             yolo_model_path=defaults.yolo_model,
# #             dataset_root=defaults.dataset_root,
# #             base_dir=defaults.base_dir,
# #         ))

# #         return jsonify(grouped), 200

# #     except ValueError as ve:
# #         return jsonify({"status": "error", "message": str(ve)}), 400
# #     except Exception as e:
# #         logging.exception("Error in dataset build (API payload)")
# #         return jsonify({"status": "error", "message": str(e)}), 500



# @fish_bp.route("/dataset/<path:filename>")
# def serve_dataset_file(filename):
#     file_path = CFG.DATASET_ROOT / filename
#     if file_path.exists():
#         return send_file(file_path)
#     return "File not found", 404


# @fish_bp.errorhandler(RequestEntityTooLarge)
# def handle_large_file(e):
#     logging.warning("Upload failed: file too large.")
#     return jsonify({
#         "status": False,
#         "message": "Uploaded image is too large.",
#         "body": None
#     }), 413


# # @fish_bp.route("/predict", methods=["POST"])
# # def predict():
# #     try:
# #         start_time = time.time()
# #         print("Timer start")
# #         if 'image' not in request.files:
# #             logging.warning("Request does not contain 'image' key.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "No image file provided.",
# #                 "body": None
# #             }), 400
        
# #         image_file = request.files['image']
# #         if image_file.filename == '':
# #             logging.warning("Image filename is empty.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "No selected file.",
# #                 "body": None
# #             }), 400
        
# #         # Extract extra form fields
# #         try:
# #             latitude = float(request.form.get("latitude"))
# #             longitude = float(request.form.get("longitude"))
# #             timestamp_str = request.form.get("timestamp")  # Expected format: 'YYYY-MM-DD HH:MM:SS'
# #             datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")  # Validate format
# #         except Exception as e:
# #             return jsonify({
# #                 "status": False,
# #                 "message": "Missing or invalid geolocation/timestamp.",
# #                 "body": None
# #             }), 400

# #         image_bytes = image_file.read()  # Read only once
# #         if len(image_bytes) == 0:
# #             logging.warning("Uploaded file is empty.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "Uploaded file is empty.",
# #                 "body": None
# #             }), 400
        
# #         logging.info("Image file received successfully.")

# #         # Save original image to 'saved_images' folder
# #         try:
# #             saved_folder = os.path.join(current_app.root_path, 'saved_images')
# #             os.makedirs(saved_folder, exist_ok=True)
# #             original_filename = f"{uuid.uuid4().hex}.jpg"
# #             original_image_path = os.path.join(saved_folder, original_filename)
# #         except Exception as err:
# #             logging.exception("Failed to prepare saved_images directory or path. Please check server permissions.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "Server Permission Error: Unable to prepare image saving directory.",
# #                 "body": None
# #             }), 500

# #         # Decode image to OpenCV format
        
# #         np_img = np.frombuffer(image_bytes, np.uint8)
# #         img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
# #         if img is None or not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
# #             logging.error("Uploaded file is not a valid 3-channel image.")
# #             raise ValueError("Uploaded file is not a valid 3-channel color image.")

# #         # Save the original image
# #         try:
# #             cv2.imwrite(original_image_path, img.copy())
# #             logging.info(f"Original image saved to {original_image_path}")
# #         except Exception as err:
# #             logging.exception("Failed to save original uploaded image.", str(err))
# #             return jsonify({
# #                 "status": False,
# #                 "message": f"Image save failed: {str(err)}",
# #                 "body": None
# #             }), 500

# #         logging.info(f"Image decoded and saved. Time: {time.time() - start_time:.2f}s")
# #         print("before detection")
        
# #         # Step 1: Detect fish (binary detection)
# #         try:
# #             fish_conf, fish_label, fish_box = detect_fish(img.copy())
# #             logging.info("Fish detection confidence: %.3f", fish_conf)
# #             print(f"Fish detection confidence: {fish_conf:.3f}")
# #         except ValueError as e:
# #             logging.warning(str(e))
# #             return jsonify({
# #                 "status": False,
# #                 "message": str(e),
# #                 "body": None
# #             }), 200


# #         # Step 2: Try species detection
# #         specie_detected = True
# #         try:
# #             specie_conf, specie_name, box = detect_fish_species(img.copy())
# #             logging.info("Species detection confidence: %.3f", specie_conf)
# #             print(f"Species detection confidence: {specie_conf:.3f}")
# #         except ValueError:
# #             logging.warning("Species not detected, falling back to fish detection box.")
# #             specie_detected = False
# #             specie_name = fish_label
# #             box = fish_box
# #             ###### extra ######
# #             # logging.warning(str(e))
# #             # return jsonify({
# #             #     "status": False,
# #             #     "message": str(e),
# #             #     "body": None
# #             # }), 200
# #             ##################
# #         print("Fish detected: ", time.time() - start_time)

# #         # Step 3: Measurement
# #         try:
# #             length, annotated_img, cropped_image = measure_fish(img, box, specie_name)
# #             logging.info("Fish measurement successful. Length: %.2f in", length)
# #             print(f"Fish measurement successful. Length: {length:.2f} in")
# #         except ValueError:
# #             logging.warning("Fish detected but measurement failed.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "Fish detected but measurement failed",
# #                 "body": None
# #             }), 500

# #         print("Length measured: ", time.time() - start_time)
        
# #         # Ensure DB folder exists before any image save
# #         ensure_db_folder()

# #         if specie_detected:
# #             # Check duplicate by business logic
# #             is_dup, dup_message, reference_image_path = is_duplicate(specie_name, length, latitude, longitude, timestamp_str)
# #             logging.info(f"{dup_message}")
# #             print(dup_message)
# #             result = {
# #                 "similar": False,
# #                 "distance": 0.0
# #             }
# #             #############################################
# #             if is_dup:
# #                 # If business logic says duplicate, do image-based uniqueness check

# #                 # Save cropped image temporarily
# #                 unique_id = str(uuid.uuid4())[:8]
# #                 img_filename = f"{specie_name}_{unique_id}.jpg"
# #                 temp_img_path = os.path.join(DB_FOLDER, img_filename)
# #                 # Convert BGR to RGB and save using PIL
# #                 rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
# #                 Image.fromarray(rgb_image).save(temp_img_path)

# #                 # Compare new cropped image with the existing reference
# #                 result = compare_two_images(temp_img_path, reference_image_path)
# #                 print("unique AI result:", result)
# #                 if result["similar"]:
# #                     # Truly duplicate → delete temp image and return
# #                     os.remove(temp_img_path)
# #                     logging.warning(f"Duplicate fish confirmed by image comparison. Distance: {result['distance']:.4f}")
                    
# #                 # Else: Unique by image — keep image and insert into DB
# #                 else:
# #                     insert_fish(specie_name, length, latitude, longitude, timestamp_str, temp_img_path)
# #                     logging.info("Fish data inserted successfully after image uniqueness check.")
# #                     # is_dup = False

# #             else:
# #                 # Not duplicate by business → insert directly (no image comparison)
# #                 unique_id = str(uuid.uuid4())[:8]
# #                 img_filename = f"{specie_name}_{unique_id}.jpg"
# #                 img_path = os.path.join(DB_FOLDER, img_filename)
# #                 # Convert BGR to RGB and save using PIL
# #                 rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
# #                 Image.fromarray(rgb_image).save(img_path)

# #                 insert_fish(specie_name, length, latitude, longitude, timestamp_str, img_path)
# #                 logging.info("Fish data inserted directly (business logic: unique).")

# #         # else:
# #         #     # No uniqueness check — just measurement result
# #         #     is_dup = False
# #         #     result = {"similar": False, "distance": 0.0}


# #         else:
# #             # Unknown species → store in unknown_fish table
# #             is_dup = False
# #             result = {"similar": False, "distance": 0.0}
# #             unique_id = str(uuid.uuid4())[:8]
# #             img_filename = f"Unknown_{unique_id}.jpg"
# #             img_path = os.path.join(DB_FOLDER_Unknown_Fish, img_filename)
# #             rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
# #             Image.fromarray(rgb_image).save(img_path)
# #             insert_unknown_fish("Unknown", length, latitude, longitude, timestamp_str, img_path)
# #             logging.info("Unknown fish inserted into unknown_fish table.")



# #         # Save annotated image to static folder
# #         # Save annotated image to <static>/predicted/<uuid>.jpg
# #         try:
# #             filename = f"{uuid.uuid4().hex}.jpg"
# #             save_path = os.path.join(current_app.static_folder, 'predicted', filename)
# #             os.makedirs(os.path.dirname(save_path), exist_ok=True)
# #             cv2.imwrite(save_path, annotated_img)
# #             logging.info("Annotated image saved at %s", save_path)
# #         except Exception as err:
# #             logging.exception("Failed to save annotated image: %s", err)
# #             return jsonify({
# #                 "status": False,
# #                 "message": f"Image save failed: {str(err)}",
# #                 "body": None
# #             }), 500


# #         # Build full URL using base (host) + static path
# #         image_url = f"{request.host_url}static/predicted/{filename}"
# #         base_url  = request.host_url  # <-- store this explicitly
# #         logging.info(f"Image URL: {image_url}")
# #         server_time = time.time() - start_time
# #         logging.info("Request completed in %.2fs", server_time)
# #         print(f"Request completed in {server_time:.2f}s")

# #         insert_predicted(
# #             species=specie_name,
# #             length_in=float(length),
# #             uniqueness=not is_dup,
# #             ai_uniqueness=not result["similar"],
# #             ai_uniqueness_distance=float(result["distance"]),
# #             server_time=float(server_time),
# #             image_path=save_path,         # local path you wrote with cv2.imwrite
# #             image_url=image_url,          # f"{request.host_url}static/predicted/{filename}"
# #             # base_url=request.host_url,    # store base URL
# #             fish_confidence=fish_conf,            # from detect_fish(...)
# #             # species_confidence=specie_conf if specie_detected else None
# #         )


# #         # --- response ---
# #         return jsonify({
# #             "status": True,
# #             "message": "Fish detected and measured successfully.",
# #             "body": {
# #                 "specie_name": specie_name,
# #                 "length_of_fish": f"{length:.2f} in",
# #                 "uniqueness": not is_dup,
# #                 "AI_uniqueness": not result["similar"],
# #                 "AI_uniqueness_distance": result["distance"],
# #                 "server_time": server_time,
# #                 # "fish_confidence": fish_conf,
# #                 # "species_confidence": specie_conf if specie_detected else None,
# #                 "image": image_url
# #             }
# #         }), 200

# #     except Exception as e:
# #         logging.exception("Unexpected server error in /predict endpoint.", e)
# #         return jsonify({
# #             "status": False,
# #             "message": f"Unexpected server error: {str(e)}",
# #             "body": None
# #         }), 500


# # #predictv2 with only siamese check with 24hrs
# # @fish_bp.route("/predict_v2", methods=["POST"])
# # def predict_v2():
# #     try:
# #         start_time = time.time()
# #         print("Timer start")
# #         if 'image' not in request.files:
# #             logging.warning("Request does not contain 'image' key.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "No image file provided.",
# #                 "body": None
# #             }), 400
       
# #         image_file = request.files['image']
# #         if image_file.filename == '':
# #             logging.warning("Image filename is empty.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "No selected file.",
# #                 "body": None
# #             }), 400
       
# #         # Extract extra form fields
# #         try:
# #             latitude = float(request.form.get("latitude"))
# #             longitude = float(request.form.get("longitude"))
# #             timestamp_str = request.form.get("timestamp")  # Expected format: 'YYYY-MM-DD HH:MM:SS'
# #             datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")  # Validate format
# #         except Exception as e:
# #             return jsonify({
# #                 "status": False,
# #                 "message": "Missing or invalid geolocation/timestamp.",
# #                 "body": None
# #             }), 400
 
# #         image_bytes = image_file.read()  # Read only once
# #         if len(image_bytes) == 0:
# #             logging.warning("Uploaded file is empty.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "Uploaded file is empty.",
# #                 "body": None
# #             }), 400
       
# #         logging.info("Image file received successfully.")
 
# #         # Save original image to 'saved_images' folder
# #         try:
# #             saved_folder = os.path.join(current_app.root_path, 'saved_images')
# #             os.makedirs(saved_folder, exist_ok=True)
# #             original_filename = f"{uuid.uuid4().hex}.jpg"
# #             original_image_path = os.path.join(saved_folder, original_filename)
# #         except Exception as err:
# #             logging.exception("Failed to prepare saved_images directory or path. Please check server permissions.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "Server Permission Error: Unable to prepare image saving directory.",
# #                 "body": None
# #             }), 500
 
# #         # Decode image to OpenCV format
# #         np_img = np.frombuffer(image_bytes, np.uint8)
# #         img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
# #         if img is None or not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
# #             logging.error("Uploaded file is not a valid 3-channel image.")
# #             raise ValueError("Uploaded file is not a valid 3-channel color image.")
 
# #         # Save the original image
# #         try:
# #             cv2.imwrite(original_image_path, img.copy())
# #             logging.info(f"Original image saved to {original_image_path}")
# #         except Exception as err:
# #             logging.exception("Failed to save original uploaded image.", str(err))
# #             return jsonify({
# #                 "status": False,
# #                 "message": f"Image save failed: {str(err)}",
# #                 "body": None
# #             }), 500
 
# #         logging.info(f"Image decoded and saved. Time: {time.time() - start_time:.2f}s")
# #         print("before detection")
       
# #         # Step 1: Detect fish (binary detection)
# #         try:
# #             _, fish_label, fish_box = detect_fish(img.copy())
# #             logging.info(f"Fish detected: {fish_label}")
# #         except ValueError as e:
# #             logging.warning(str(e))
# #             return jsonify({
# #                 "status": False,
# #                 "message": str(e),
# #                 "body": None
# #             }), 200
       
# #         # Step 2: Try species detection
# #         specie_detected = True
# #         try:
# #             _, specie_name, box = detect_fish_species(img.copy())
# #             logging.info(f"Fish species detected: {specie_name}")
# #         except ValueError as e:
# #             logging.warning("Species not detected, falling back to fish detection box.")
# #             specie_detected = False
# #             specie_name = fish_label
# #             box = fish_box
# #         print("Fish detected: ", time.time() - start_time)
 
# #         # Step 3: Measurement
# #         try:
# #             length, annotated_img, cropped_image = measure_fish_v2(img, box, specie_name)
# #             print("============================================")
# #             print("calling V2 measurement function")
# #             print("============================================")
# #             logging.info(f"Fish measurement successful. Length: {length:.2f} in")
# #             print(f"Fish measurement successful. Length: {length:.2f} in")
# #         except ValueError:
# #             logging.warning("Fish detected but measurement failed.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "Fish detected but measurement failed",
# #                 "body": None
# #             }), 500
 
# #         print("Length measured: ", time.time() - start_time)
       
# #         # Ensure DB folder exists before any image save
# #         ensure_db_folder()
 
# #         if specie_detected:
# #             # ✅ Minimal change: call simplified is_duplicate
# #             is_dup, dup_message, reference_image_path = is_duplicate_v2(timestamp_str)
# #             logging.info(f"{dup_message}")
# #             print(dup_message)
# #             result = {
# #                 "similar": False,
# #                 "distance": 0.0
# #             }
# #             if is_dup:
# #                 # If business logic says duplicate, do image-based uniqueness check
# #                 unique_id = str(uuid.uuid4())[:8]
# #                 img_filename = f"{specie_name}_{unique_id}.jpg"
# #                 temp_img_path = os.path.join(DB_FOLDER, img_filename)
# #                 rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
# #                 Image.fromarray(rgb_image).save(temp_img_path)
 
# #                 result = compare_two_images(temp_img_path, reference_image_path)
# #                 print("unique AI result:", result)
# #                 if result["similar"]:
# #                     os.remove(temp_img_path)
# #                     logging.warning(f"Duplicate fish confirmed by image comparison. Distance: {result['distance']:.4f}")
# #                 else:
# #                     insert_fish(specie_name, length, latitude, longitude, timestamp_str, temp_img_path)
# #                     logging.info("Fish data inserted successfully after image uniqueness check.")
# #             else:
# #                 unique_id = str(uuid.uuid4())[:8]
# #                 img_filename = f"{specie_name}_{unique_id}.jpg"
# #                 img_path = os.path.join(DB_FOLDER, img_filename)
# #                 rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
# #                 Image.fromarray(rgb_image).save(img_path)
# #                 insert_fish(specie_name, length, latitude, longitude, timestamp_str, img_path)
# #                 logging.info("Fish data inserted directly (business logic: unique).")
 
# #         else:
# #             is_dup = False
# #             result = {"similar": False, "distance": 0.0}
# #             unique_id = str(uuid.uuid4())[:8]
# #             img_filename = f"Unknown_{unique_id}.jpg"
# #             img_path = os.path.join(DB_FOLDER, img_filename)
# #             rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
# #             Image.fromarray(rgb_image).save(img_path)
# #             insert_unknown_fish("Unknown", length, latitude, longitude, timestamp_str, img_path)
# #             logging.info("Unknown fish inserted into unknown_fish table.")
 
# #         # Save annotated image to static folder
# #         try:
# #             filename = f"{uuid.uuid4().hex}.jpg"
# #             save_path = os.path.join(current_app.static_folder, 'predicted', filename)
# #             os.makedirs(os.path.dirname(save_path), exist_ok=True)
# #             cv2.imwrite(save_path, annotated_img)
# #             logging.info(f"Annotated image saved at {save_path}")
# #         except Exception as err:
# #             logging.exception("Failed to save annotated image.", str(err))
# #             return jsonify({
# #                 "status": False,
# #                 "message": f"Image save failed: {str(err)}",
# #                 "body": None
# #             }), 500
 
# #         image_url = f"{request.host_url}static/predicted/{filename}"
# #         logging.info(f"Image URL: {image_url}")
# #         logging.info(f"Request completed in {time.time() - start_time:.2f}s")
# #         print(f"Request completed in {time.time() - start_time:.2f}s")
# #         server_time = time.time() - start_time
# #         return jsonify({
# #             "status": True,
# #             "message": "Fish detected and measured successfully.",
# #             "body": {
# #                 "specie_name": specie_name,
# #                 "length_of_fish": f"{length:.2f} in",
# #                 "uniqueness": not is_dup,
# #                 "AI_uniqueness": not result["similar"],
# #                 "AI_uniqueness_distance": result["distance"],
# #                 "server_time": server_time,
# #                 "image": image_url
# #             }
# #         }), 200
 
# #     except Exception as e:
# #         logging.exception("Unexpected server error in /predict endpoint.", e)
# #         return jsonify({
# #             "status": False,
# #             "message": f"Unexpected server error: {str(e)}",
# #             "body": None
# #         }), 500



# # @fish_bp.route("/predict_bbox", methods=["POST"])
# # def predict_bbox():
# #     try:
# #         if 'image' not in request.files:
# #             logging.warning("Request does not contain 'image' key.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "No image file provided.",
# #                 "body": None
# #             }), 400
        
# #         image_file = request.files['image']
# #         if image_file.filename == '':
# #             logging.warning("Image filename is empty.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "No selected file.",
# #                 "body": None
# #             }), 400

# #         image_bytes = image_file.read()  # Read only once
# #         if len(image_bytes) == 0:
# #             logging.warning("Uploaded file is empty.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "Uploaded file is empty.",
# #                 "body": None
# #             }), 400
        
# #         logging.info("Image file received successfully.")
# #         print("Image file received successfully.")
# #         # Decode image to OpenCV format
        
# #         np_img = np.frombuffer(image_bytes, np.uint8)
# #         img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
# #         if img is None or not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
# #             logging.error("Uploaded file is not a valid 3-channel image.")
# #             raise ValueError("Uploaded file is not a valid 3-channel color image.")

# #         bbox, confidence = get_top_bbox_yolo_format(img.copy())
# #         # Convert all bbox values to Python floats
# #         bbox = [int(v) for v in bbox]

# #         print("bbox, confidence", bbox, confidence)
# #         if bbox is None:
# #             return jsonify({
# #                 "status": False,
# #                 "message": "No fish detected.",
# #                 "body": None
# #             }), 200
# #         return jsonify({
# #             "status": True,
# #             "message": "Top fish detected.",
# #             "body": {
# #                 "bbox": {
# #                     "x_center": bbox[0],
# #                     "y_center": bbox[1],
# #                     "width": bbox[2],
# #                     "height": bbox[3]
# #                 },
# #                 "confidence": confidence
# #             }
# #         }), 200
    
# #     except Exception as e:
# #         logging.exception("Unexpected server error in /predict_bbox endpoint.", e)
# #         return jsonify({
# #             "status": False,
# #             "message": f"Unexpected server error: {str(e)}",
# #             "body": None
# #         }), 500

# # @fish_bp.route("/species_detection", methods=["POST"])
# # def species_detection():
# #     try:
# #         start_time = time.time()
# #         if 'image' not in request.files:
# #             logging.warning("Request does not contain 'image' key.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "No image file provided.",
# #                 "body": None
# #             }), 400
        
# #         image_file = request.files['image']
# #         if image_file.filename == '':
# #             logging.warning("Image filename is empty.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "No selected file.",
# #                 "body": None
# #             }), 400

# #         image_bytes = image_file.read()  # Read only once
# #         if len(image_bytes) == 0:
# #             logging.warning("Uploaded file is empty.")
# #             return jsonify({
# #                 "status": False,
# #                 "message": "Uploaded file is empty.",
# #                 "body": None
# #             }), 400
        
# #         logging.info("Image file received successfully.")

        
# #         np_img = np.frombuffer(image_bytes, np.uint8)
# #         img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
# #         if img is None or not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
# #             logging.error("Uploaded file is not a valid 3-channel image.")
# #             raise ValueError("Uploaded file is not a valid 3-channel color image.")

        
# #         # Detect fish
# #         try:
# #             _, specie_name, box = detect_fish_species(img.copy())
# #             logging.info(f"Fish detected. Specie: {specie_name}")
# #         except ValueError as e:
# #             logging.warning(str(e))
# #             return jsonify({
# #                 "status": False,
# #                 "message": str(e),
# #                 "body": None
# #             }), 200

# #         print("Fish detected: ", time.time() - start_time)


# #         # Save annotated image to static folder
# #         try:
# #             filename = f"{uuid.uuid4().hex}.jpg"
# #             save_path = os.path.join(current_app.static_folder, 'saved', filename)
# #             os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
# #             cv2.imwrite(save_path, img)
# #             logging.info(f"Annotated image saved at {save_path}")

# #         except Exception as err:
# #             logging.exception("Failed to save annotated image.", str(err))
# #             return jsonify({
# #                 "status": False,
# #                 "message": f"Image save failed: {str(err)}",
# #                 "body": None
# #             }), 500

# #         image_url = f"{request.host_url}static/saved/{filename}"
# #         logging.info(f"Image URL: {image_url}")
# #         logging.info(f"Request completed in {time.time() - start_time:.2f}s")
# #         print(f"Request completed in {time.time() - start_time:.2f}s")
# #         server_time = time.time() - start_time
# #         return jsonify({
# #             "status": True,
# #             "message": "Fish detected successfully.",
# #             "body": {
# #                 "specie_name": specie_name,
# #                 "image": image_url
# #             }
# #         }), 200

# #     except Exception as e:
# #         logging.exception("Unexpected server error in /species_detection endpoint.", e)
# #         return jsonify({
# #             "status": False,
# #             "message": f"Unexpected server error: {str(e)}",
# #             "body": None
# #         }), 500
     



# # -------------------------------
# # Fish Detect API Endpoint
# # -------------------------------

# # # LOCAL_MODELS_DIR = r"D:\\Rahul Puri Data\\Projects\\Project GFL\\app_backend\\models_cache"
# # # OUTPUT_DIR = r"D:\\Rahul Puri Data\\Projects\\Project GFL\\app_backend\\Predictions"
# # os.makedirs(LOCAL_PREDICTION_MODELS_DIR, exist_ok=True)
# # os.makedirs(PREDICTED_OUTPUT_DIR, exist_ok=True)


# # _loaded_models = None
# # # LOCAL_PREDICTION_MODELS_DIR = Path("D:\\Rahul Puri Data\\Projects\\Project GFL\\app_backend\\modules\\fish_training\\models")

# # def get_loaded_models(limit=5):
# #     global _loaded_models
# #     if _loaded_models is None:
# #         _loaded_models = {}

# #         # ✅ Ensure local dir exists
# #         LOCAL_PREDICTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# #         # 1️⃣ Load any models already available locally
# #         local_models = list(LOCAL_PREDICTION_MODELS_DIR.glob("*.pt"))
# #         for model_path in local_models[:limit]:
# #             try:
# #                 _loaded_models[model_path.name] = YOLO(str(model_path))
# #                 print(f"✅ Loaded local model: {model_path.name}")
# #             except Exception as e:
# #                 print(f"❌ Failed to load local model {model_path.name}: {e}")

# #         # 2️⃣ If still below limit → fetch from Azure
# #         if len(_loaded_models) < limit:
# #             try:
# #                 blob_service = BlobServiceClient(
# #                     account_url="https://gflstorageaccount.blob.core.windows.net",
# #                     credential=AZURE_CONFIG["sas_token"]
# #                 )
# #                 container_client = blob_service.get_container_client("dotnetbackend-container")

# #                 count = len(_loaded_models)
# #                 for blob in container_client.list_blobs(name_starts_with="models/"):
# #                     if blob.name.endswith(".pt"):
# #                         local_path = LOCAL_PREDICTION_MODELS_DIR / os.path.basename(blob.name)
# #                         if not local_path.exists():
# #                             with open(local_path, "wb") as f:
# #                                 f.write(container_client.download_blob(blob.name).readall())
# #                             print(f"⬇️ Downloaded model from Azure: {blob.name}")

# #                         if local_path.name not in _loaded_models:
# #                             try:
# #                                 _loaded_models[local_path.name] = YOLO(str(local_path))
# #                                 print(f"✅ Loaded model: {blob.name}")
# #                                 count += 1
# #                             except Exception as e:
# #                                 print(f"❌ Failed to load {blob.name}: {e}")

# #                         if count >= limit:
# #                             break
# #             except Exception as ex:
# #                 print(f"⚠️ Error loading models from Azure: {ex}")

# #     return _loaded_models


# # @fish_bp.route("/detect_fish", methods=["POST"])
# # def detect_fish_multimodel():
# #     try:
# #         if "image" not in request.files:
# #             return jsonify({"status": False, "message": "No image uploaded"}), 400

# #         image_file = request.files["image"]
# #         image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

# #         # ✅ lazy-load models (first call only)
# #         models = get_loaded_models(limit=5)  # adjust limit as you need

# #         results_summary = []
# #         best_detection = {"class": "Not detected", "confidence": 0.0, "model": None, "output_image": None}

# #         for model_name, model in models.items():
# #             # results = model(image, conf=0.10)  # try low conf first
# #             results = model.predict(image, imgsz=640, conf=0.1)
# #             names = model.names
# #             detected_classes = []

# #             for box in results[0].boxes:
# #                 cls_id = int(box.cls[0].item())
# #                 conf = float(box.conf[0].item())
# #                 detected_classes.append({
# #                     "class": names[cls_id],
# #                     "confidence": round(conf, 3)
# #                 })

# #             # Filename based on classes
# #             unique_names = {d["class"] for d in detected_classes}
# #             cls_name = "_".join(unique_names) if unique_names else "Unknown"
# #             cls_name = "".join(c for c in cls_name if c.isalnum() or c in (" ", "_")).rstrip()
# #             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #             output_path = os.path.join(PREDICTED_OUTPUT_DIR, f"{model_name}_{cls_name}_{timestamp}.jpg")

# #             annotated_img = results[0].plot()
# #             cv2.imwrite(output_path, annotated_img)

# #             # Save summary for debug/logging
# #             results_summary.append({
# #                 "model": model_name,
# #                 "detections": detected_classes if detected_classes else [{"class": "Unknown", "confidence": 0.0}],
# #                 "output_image": output_path
# #             })

# #             print(f"Model: {model_name}, Model_class: {names}")

# #             # ✅ Track best detection
# #             for det in detected_classes:
# #                 if det["confidence"] > best_detection["confidence"]:
# #                     best_detection = {
# #                         "class": det["class"],
# #                         "confidence": det["confidence"],
# #                         "model": model_name,
# #                         "output_image": output_path
# #                     }

# #         # ✅ If no detection across all models
# #         if best_detection["confidence"] == 0.0:
# #             best_detection = {"class": "Not detected", "confidence": 0.0, "model": None, "output_image": None}

# #         return jsonify({
# #             "status": True,
# #             "message": "Species Detection Successfully",
# #             "best_result": best_detection,
# #             "all_results": results_summary  # keep full history for debugging
# #         })

# #     except Exception as e:
# #         return jsonify({
# #             "status": False,
# #             "message": f"Error during detection: {str(e)}"
# #         }), 500




# # -------------------------
# # Routes
# # -------------------------

# import re
# # ---- Helper function to sanitize folder names for Windows ----
# def sanitize_filename(name: str) -> str:
#     # Replace invalid Windows filename characters with underscore
#     return re.sub(r'[\\/*?:"<>|]', "_", name)


# # ---- Species ----
# @fish_bp.route("/species", methods=["GET", "POST"])
# def species():
#     if request.method == "POST":
#         data = request.json
#         species_name = data.get("species_name")
#         if not species_name:
#             return jsonify({"error": "species_name is required"}), 400

#         try:
#             conn = sqlite3.connect(DB_PATH_DATA_COLLECTION)
#             cursor = conn.cursor()
#             cursor.execute(
#                 "INSERT INTO species (species_name) VALUES (?)", (species_name,)
#             )
#             conn.commit()
#             species_id = cursor.lastrowid
#             conn.close()
#             return jsonify({"message": f"Species '{species_name}' added", "id": species_id}), 201
#         except sqlite3.IntegrityError:
#             return jsonify({"error": f"Species '{species_name}' already exists"}), 400

#     else:  # GET method
#         conn = sqlite3.connect(DB_PATH_DATA_COLLECTION)
#         cursor = conn.cursor()
#         cursor.execute("SELECT id, species_name FROM species")
#         rows = cursor.fetchall()
#         conn.close()
#         return jsonify([{"id": r[0], "species_name": r[1]} for r in rows])


# # @fish_bp.route("/images", methods=["GET", "POST"])
# # def images():
# #     if request.method == "POST":
# #         try:
# #             # ----- POST logic (upload) -----
# #             if request.is_json:
# #                 payload = request.get_json()
# #                 species_id = payload.get("species_id")
# #                 handheld = bool(payload.get("handheld", False))
# #                 images_data = payload.get("images", [])
# #             else:
# #                 species_id = request.form.get("species_id")
# #                 handheld = request.form.get("handheld", "false").lower() in ["true", "1", "yes"]
# #                 images_data = request.files.getlist("images")

# #             if not species_id:
# #                 return jsonify({"error": "species_id is required"}), 400
# #             species_id = int(species_id)

# #             if not images_data or len(images_data) == 0:
# #                 return jsonify({"error": "At least one image is required"}), 400

# #             # Fetch species name
# #             conn = sqlite3.connect(DB_PATH_DATA_COLLECTION)
# #             cursor = conn.cursor()
# #             cursor.execute("SELECT species_name FROM species WHERE id = ?", (species_id,))
# #             row = cursor.fetchone()
# #             conn.close()
# #             if not row:
# #                 return jsonify({"error": f"Species ID '{species_id}' not found"}), 400
# #             species_name = row[0]

# #             images = []
# #             for idx, img in enumerate(images_data, start=0):
# #                 if request.is_json:
# #                     # base64 → bytes
# #                     image_bytes = base64.b64decode(img.get("data"))
# #                     image_url = upload_file_to_blob(species_name, handheld, BytesIO(image_bytes))
# #                 else:
# #                     image_url = upload_file_to_blob(species_name, handheld, img)

# #                 # Insert into DB
# #                 conn = sqlite3.connect(DB_PATH_DATA_COLLECTION)
# #                 cursor = conn.cursor()
# #                 cursor.execute(
# #                     "INSERT INTO images (image_path, handheld, species_id) VALUES (?, ?, ?)",
# #                     (image_url, handheld, species_id)
# #                 )
# #                 conn.commit()
# #                 conn.close()

# #                 images.append({
# #                     "imageId": idx,
# #                     "imagePath": image_url
# #                 })

# #             response = {
# #                 "speciesId": species_id,
# #                 "isHeld": handheld,
# #                 "images": images
# #             }
# #             return jsonify(response), 201

# #         except Exception as e:
# #             return jsonify({"error": str(e)}), 500

# #     else:
# #         # ----- GET logic -----
# #         try:
# #             species_id = request.args.get("species_id")
# #             if not species_id:
# #                 return jsonify({"error": "species_id query param is required"}), 400
# #             species_id = int(species_id)

# #             conn = sqlite3.connect(DB_PATH_DATA_COLLECTION)
# #             cursor = conn.cursor()
# #             cursor.execute("""
# #                 SELECT id, image_path, handheld
# #                 FROM images
# #                 WHERE species_id = ?
# #             """, (species_id,))
# #             rows = cursor.fetchall()
# #             conn.close()

# #             images = []
# #             for idx, row in enumerate(rows, start=0):
# #                 images.append({
# #                     "imageId": idx,
# #                     "imagePath": row[1]
# #                 })

# #             # If multiple handheld values exist, pick the first (just for response)
# #             isHeld = bool(rows[0][2]) if rows else False

# #             response = {
# #                 "speciesId": species_id,
# #                 "isHeld": isHeld,
# #                 "images": images
# #             }
# #             return jsonify(response), 200

# #         except Exception as e:
# #             return jsonify({"error": str(e)}), 500

 

# training_lock = threading.Lock()
# TRAINING_STATUS = {"state": "idle", "message": "No training running."}


# # @fish_bp.route("/run_pipeline", methods=["POST"])
# # def run_pipeline():
# #     """Run the full training pipeline via Azure ML."""
# #     global TRAINING_STATUS

# #     data = request.get_json(silent=True) or {}
# #     species_name = data.get("speciesName")
# #     project_id = data.get("projectId")

# #     # --- Basic validation ---
# #     if not species_name:
# #         return jsonify({"status": "error", "message": "speciesName is required"}), 400
# #     if not project_id:
# #         return jsonify({"status": "error", "message": "projectId is required"}), 400

# #     if TRAINING_STATUS.get("state") == "running":
# #         return jsonify({
# #             "status": "busy",
# #             "message": "Training is already running. Please wait for it to complete."
# #         }), 429

# #     # # --- Background worker for non-blocking response ---
# #     # def background_job(species_name: str, project_id: int):
# #     #     global TRAINING_STATUS
# #     #     if not training_lock.acquire(blocking=False):
# #     #         print("⚠️ Another training job is already running — skipping new one.")
# #     #         return

# #     #     try:
# #     #         TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}
# #     #         print(f"🚀 Submitting Azure ML training for project {project_id}, species: {species_name}")

# #     #         # Authenticate to Azure ML
# #     #         # # Use your Azure CLI credentials
# #     #         # credential = AzureCliCredential()

# #     #         credential = DefaultAzureCredential()
# #     #         ml_client = MLClient(
# #     #             credential=credential,
# #     #             subscription_id=AZURE_SUBSCRIPTION_ID,
# #     #             resource_group_name=AZURE_RESOURCE_GROUP,
# #     #             workspace_name=AZURE_ML_WORKSPACE,
# #     #         )

# #     #         compute_name = "GflTrainCompute"

# #     #         # --- Ensure compute exists (create if missing) ---
# #     #         # ✅ Just check compute exists, do NOT create
# #     #         try:
# #     #             ml_client.compute.get(compute_name)
# #     #             print(f"✅ Compute '{compute_name}' is available.")
# #     #         except Exception as e:
# #     #             print(f"❌ Compute target '{compute_name}' not found: {e}")
# #     #             TRAINING_STATUS = {
# #     #                 "state": "failed",
# #     #                 "message": f"Compute target '{compute_name}' not found in Azure ML workspace."
# #     #             }
# #     #             training_lock.release()
# #     #             return


# #     #         # --- Entry script path in your Azure ML blob store ---
# #     #         entry_code_path = (
# #     #             "https://gflstorageblob.blob.core.windows.net/"
# #     #             "azureml-blobstore-bbd3a4dc-14ed-4039-80c7-7d0d2df1bd36/"
# #     #             "UI/JobSubmission/10-23-2025_102845_UTC/"
# #     #         )

# #     #         # --- Define and submit the job ---
# #     #         job = command(
# #     #             code=entry_code_path,  # this is your remote folder in Azure ML datastore
# #     #             command=(
# #     #                 f"pip install -r requirements.txt && "
# #     #                 f"python train_pipeline_entry.py --species '{species_name}' --project_id {project_id}"
# #     #             ),
# #     #             environment=(
# #     #                 "azureml://locations/eastus2/workspaces/"
# #     #                 "bbd3a4dc-14ed-4039-80c7-7d0d2df1bd36/"
# #     #                 "environments/gflTrainEnv/versions/5"
# #     #             ),
# #     #             compute=compute_name,
# #     #             experiment_name="GFLTrainingModelV1",
# #     #             display_name=f"fish_train_{species_name}_{project_id}",
# #     #             description="YOLO fish species training job"
# #     #         )

# #     #         submitted_job = ml_client.jobs.create_or_update(job)

# #     #         TRAINING_STATUS = {
# #     #             "state": "submitted",
# #     #             "message": f"✅ Azure ML job submitted: {submitted_job.name}",
# #     #             "job_name": submitted_job.name
# #     #         }
# #     #         print(f"✅ Submitted Azure ML job: {submitted_job.name}")

# #     #     except Exception as e:
# #     #         TRAINING_STATUS = {
# #     #             "state": "failed",
# #     #             "message": f"❌ Azure ML job submission failed: {str(e)}"
# #     #         }
# #     #         print(f"❌ Azure ML submission failed: {e}")

# #     #     finally:
# #     #         training_lock.release()

# #     def background_job(species_name: str, project_id: int):
# #         global TRAINING_STATUS
# #         if not training_lock.acquire(blocking=False):
# #             print("⚠️ Another training job is already running — skipping new one.")
# #             return

# #         try:
# #             TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}

# #             credential = DefaultAzureCredential()
# #             ml_client = MLClient(
# #                 credential=credential,
# #                 subscription_id=AZURE_SUBSCRIPTION_ID,
# #                 resource_group_name=AZURE_RESOURCE_GROUP,
# #                 workspace_name=AZURE_ML_WORKSPACE,
# #             )

# #             compute_name = "GflTrainCompute"
# #             ml_client.compute.get(compute_name)
# #             print(f"✅ Compute '{compute_name}' is available.")

# #             # ✅ Correct datastore path
# #             entry_code_path = Input(
# #                 type="uri_folder",
# #                 path=(
# #                     "azureml://subscriptions/417ecdbc-9b50-421a-82c2-c45f29686df1/"
# #                     "resourcegroups/GFL-App-Rg/workspaces/GFL-MLAI/"
# #                     "datastores/workspaceblobstore/paths/UI/JobSubmission/10-23-2025_102845_UTC/"
# #                 ),
# #             )

# #             job = command(
# #                 code=entry_code_path,
# #                 command=(
# #                     "pip install -r requirements.txt && "
# #                     f"python train_pipeline_entry.py "
# #                     f"--species '{species_name}' --project_id {project_id}"
# #                 ),
# #                 environment=(
# #                     "azureml://locations/eastus2/workspaces/"
# #                     "bbd3a4dc-14ed-4039-80c7-7d0d2df1bd36/"
# #                     "environments/gflTrainEnv/versions/5"
# #                 ),
# #                 compute=compute_name,
# #                 experiment_name="GFLTrainingModelV1",
# #                 display_name=f"fish_train_{species_name}_{project_id}",
# #                 description="YOLO fish species training job",
# #             )

# #             submitted_job = ml_client.jobs.create_or_update(job)
# #             TRAINING_STATUS = {
# #                 "state": "submitted",
# #                 "message": f"✅ Azure ML job submitted: {submitted_job.name}",
# #                 "job_name": submitted_job.name,
# #             }
# #             print(f"✅ Submitted Azure ML job: {submitted_job.name}")

# #         except Exception as e:
# #             TRAINING_STATUS = {"state": "failed", "message": str(e)}
# #             print(f"❌ Azure ML submission failed: {e}")

# #         finally:
# #             training_lock.release()


# #     # --- Run background thread ---
# #     thread = threading.Thread(target=background_job, args=(species_name, project_id), daemon=True)
# #     thread.start()

# #     # Immediate HTTP response
# #     return jsonify({
# #         "status": "started",
# #         "message": f"Training job submitted for project {project_id}, species {species_name}.",
# #         "check_status_at": "/pipeline_status"
# #     }), 202




# @fish_bp.route("/run_pipeline", methods=["POST"])
# def run_pipeline():
#     """Run the full training pipeline via Azure ML."""
#     global TRAINING_STATUS

#     data = request.get_json(silent=True) or {}
#     species_name = data.get("speciesName")
#     project_id = data.get("projectId")

#     # --- Basic validation ---
#     if not species_name:
#         return jsonify({"status": "error", "message": "speciesName is required"}), 400
#     if not project_id:
#         return jsonify({"status": "error", "message": "projectId is required"}), 400

#     if TRAINING_STATUS.get("state") == "running":
#         return jsonify({
#             "status": "busy",
#             "message": "Training is already running. Please wait for it to complete."
#         }), 429

#     # --- Background worker for non-blocking response ---
#     def background_job(species_name: str, project_id: int):
#         global TRAINING_STATUS
#         lock_acquired = training_lock.acquire(blocking=False)

#         if not lock_acquired:
#             print("⚠️ Another training job is already running — skipping new one.")
#             return

#         try:
#             TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}
#             print(f"🚀 Submitting Azure ML training for project {project_id}, species: {species_name}")

#             # --- Authenticate to Azure ML ---
#             credential = DefaultAzureCredential()
#             ml_client = MLClient(
#                 credential=credential,
#                 subscription_id=AZURE_SUBSCRIPTION_ID,
#                 resource_group_name=AZURE_RESOURCE_GROUP,
#                 workspace_name=AZURE_ML_WORKSPACE,
#             )

#             # compute_name = "GflTrainCompute"
#             compute_name = "gflTrainComputeV1"
            
#             # --- Ensure compute exists and is running ---
#             try:
#                 compute = ml_client.compute.get(compute_name)
#                 state = compute.properties.get("provisioning_state", None)
#                 print(f"ℹ️ Compute '{compute_name}' current state: {state}")
#                 ml_client.compute.begin_start(compute_name).wait(120)
#                 print(f"✅ Compute '{compute.provisioning_state}' is starting...")

#                 # if compute.provisioning_state == 'Stopped':
#                 #     print(f"⚠️ Compute '{compute_name}' is stopped. Starting it now...")
#                 #     ml_client.compute.begin_start(compute_name)
#                 #     print(f"✅ Compute '{compute_name}' is starting...")
                
#                 if state in ["Stopped", "Deallocated", "Unknown"]:
#                     print(f"⚠️ Compute '{compute_name}' is not running. Starting now...")
#                     poller = ml_client.compute.begin_start(compute_name)
#                     poller.wait()  # Wait for start to complete
#                     print(f"✅ Compute '{compute_name}' has started successfully.")
#                 else:
#                     print(f"✅ Compute '{compute_name}' is already running.")


#             except Exception as e:
#                 print(f"❌ Error fetching compute target '{compute_name}': {e}")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"Compute target '{compute_name}' not found or failed to start in Azure ML workspace."
#                 }
#                 training_lock.release()
#                 return

#             # --- Access datastore dynamically ---
#             try:
#                 print("🔍 Retrieving workspace datastore...")
#                 # Connect using AzureML v1 to get blob path
#                 from azureml.core import Workspace as WSv1
#                 ws_v1 = WSv1.get(
#                     name=AZURE_ML_WORKSPACE,
#                     subscription_id=AZURE_SUBSCRIPTION_ID,
#                     resource_group=AZURE_RESOURCE_GROUP
#                 )
#                 datastore = Datastore.get(ws_v1, "workspaceblobstore")
#                 print(f"✅ Datastore retrieved: {datastore.name}")

#                 # Define source directory path inside blob
#                 source_path = "UI/JobSubmission/10-23-2025_102845_UTC"
#                 datastore_uri = f"azureml://datastores/{datastore.name}/paths/{source_path}/"
#                 print(f"📁 Using datastore path: {datastore_uri}")

#             except Exception as ds_error:
#                 print(f"❌ Datastore access failed: {ds_error}")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"Datastore access failed: {ds_error}"
#                 }
#                 training_lock.release()
#                 return

#             # --- Define and submit the job ---
#             job = command(
#                 code=datastore_uri,  # ✅ now uses datastore URI dynamically
#                 command=(
#                     f"pip install -r requirements.txt && "
#                     f"python train_pipeline_entry.py --species '{species_name}' --project_id {project_id}"
#                 ),
#                 environment=(
#                     "azureml://locations/eastus2/workspaces/"
#                     "bbd3a4dc-14ed-4039-80c7-7d0d2df1bd36/"
#                     "environments/gflTrainEnv/versions/5"
#                 ),
#                 compute=compute_name,
#                 experiment_name="GFLTrainingModelV1",
#                 display_name=f"fish_train_{species_name}_{project_id}",
#                 description="YOLO fish species training job"
#             )

#             submitted_job = ml_client.jobs.create_or_update(job)

#             TRAINING_STATUS = {
#                 "state": "submitted",
#                 "message": f"✅ Azure ML job submitted: {submitted_job.name}",
#                 "job_name": submitted_job.name
#             }
#             print(f"✅ Submitted Azure ML job: {submitted_job.name}")

#         except Exception as e:
#             TRAINING_STATUS = {
#                 "state": "failed",
#                 "message": f"❌ Azure ML job submission failed: {str(e)}"
#             }
#             print(f"❌ Azure ML submission failed: {e}")

#         # finally:
#         #     if lock_acquired:
#         #         training_lock.release()
                
#         finally:
#             # Always release safely
#             if training_lock.locked():
#                 training_lock.release()


#     # --- Run background thread ---
#     thread = threading.Thread(target=background_job, args=(species_name, project_id), daemon=True)
#     thread.start()

#     # Immediate HTTP response
#     return jsonify({
#         "status": "started",
#         "message": f"Training job submitted for project {project_id}, species {species_name}.",
#         "check_status_at": "/pipeline_status"
#     }), 202




# # @fish_bp.route("/run_pipeline", methods=["POST"])
# # def run_pipeline():
# #     """Run the full training pipeline via Azure ML (v1 SDK)."""
# #     global TRAINING_STATUS

# #     data = request.get_json(silent=True) or {}
# #     species_name = data.get("speciesName")
# #     project_id = data.get("projectId")

# #     # --- Validation ---
# #     if not species_name:
# #         return jsonify({"status": "error", "message": "speciesName is required"}), 400
# #     if not project_id:
# #         return jsonify({"status": "error", "message": "projectId is required"}), 400

# #     if TRAINING_STATUS.get("state") == "running":
# #         return jsonify({
# #             "status": "busy",
# #             "message": "Training is already running. Please wait for it to complete."
# #         }), 429

# #     # def background_job(species_name: str, project_id: int):
# #     #     global TRAINING_STATUS
# #     #     if not training_lock.acquire(blocking=False):
# #     #         logging.info("⚠️ Another training job is already running — skipping new one.")
# #     #         return

# #     #     try:
# #     #         TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}
# #     #         logging.info(f"🚀 Submitting Azure ML training for project {project_id}, species: {species_name}")

# #     #         # --- Connect to workspace ---
# #     #         ws = Workspace(
# #     #             subscription_id=AZURE_SUBSCRIPTION_ID,
# #     #             resource_group=AZURE_RESOURCE_GROUP,
# #     #             workspace_name=AZURE_ML_WORKSPACE,
# #     #         )
# #     #         logging.info("✅ Connected to Azure ML Workspace.")

# #     #         # --- Get compute target ---
# #     #         compute_target = ComputeTarget(workspace=ws, name="GflTrainCompute")
# #     #         logging.info("✅ Compute target located.")
# #     #         # --- Get environment ---
# #     #         env = ws.environments["gflTrainEnv"]
# #     #         logging.info("✅ Loaded environment 'gflTrainEnv'.")

# #     #         # # --- Set up datastore path as source directory ---
# #     #         # # This assumes files exist at this path in your workspaceblobstore
# #     #         # datastore = Datastore.get(ws, "workspaceblobstore")
# #     #         # source_directory = "UI/JobSubmission/10-23-2025_102845_UTC"
# #     #         # logging.info(datastore.endpoint)
# #     #         # # --- Define ScriptRunConfig ---
# #     #         # command = [
# #     #         #     "python", "train_pipeline_entry.py",
# #     #         #     "--species", species_name,
# #     #         #     "--project_id", str(project_id)
# #     #         # ]

# #     #         # config = ScriptRunConfig(
# #     #         #     source_directory=source_directory,
# #     #         #     command=command,
# #     #         #     compute_target=compute_target,
# #     #         #     environment=env,
# #     #         # )

# #     #         # # --- Create experiment and submit job ---
# #     #         # exp = Experiment(ws, "GFLTrainingModelV2")
# #     #         # run = exp.submit(config)
# #     #         # print(f"✅ Submitted run. View here: {run.get_portal_url()}")

# #     #         # TRAINING_STATUS = {
# #     #         #     "state": "submitted",
# #     #         #     "message": f"Azure ML job submitted successfully.",
# #     #         #     "run_id": run.id,
# #     #         #     "portal_url": run.get_portal_url()
# #     #         # }


# #     #         try:
# #     #             # --- Set up datastore path as source directory ---
# #     #             logging.info("🔍 Fetching workspace datastore...")
# #     #             datastore = Datastore.get(ws, "workspaceblobstore")
# #     #             logging.info(f"✅ Datastore retrieved: {datastore.name}")

# #     #             source_directory = "UI/JobSubmission/10-23-2025_102845_UTC"
# #     #             logging.info(f"📁 Source directory set to: {source_directory}")
# #     #             logging.info(f"🧭 Datastore endpoint: {datastore.endpoint}")

# #     #             try:
# #     #                 # --- Define ScriptRunConfig ---
# #     #                 command = [
# #     #                     "python", "train_pipeline_entry.py",
# #     #                     "--species", species_name,
# #     #                     "--project_id", str(project_id)
# #     #                 ]

# #     #                 logging.info("⚙️ Creating ScriptRunConfig...")
# #     #                 config = ScriptRunConfig(
# #     #                     source_directory=source_directory,
# #     #                     command=command,
# #     #                     compute_target=compute_target,
# #     #                     environment=env,
# #     #                 )
# #     #                 logging.info("✅ ScriptRunConfig created successfully.")

# #     #                 try:
# #     #                     # --- Create experiment and submit job ---
# #     #                     exp = Experiment(ws, "GFLTrainingModelV2")
# #     #                     logging.info("🚀 Submitting Azure ML job...")
# #     #                     run = exp.submit(config)
# #     #                     logging.info(f"✅ Submitted run successfully. View at: {run.get_portal_url()}")

# #     #                     TRAINING_STATUS = {
# #     #                         "state": "submitted",
# #     #                         "message": "Azure ML job submitted successfully.",
# #     #                         "run_id": run.id,
# #     #                         "portal_url": run.get_portal_url()
# #     #                     }

# #     #                 except Exception as submit_error:
# #     #                     logging.error(f"❌ Failed to submit Azure ML experiment: {submit_error}")
# #     #                     TRAINING_STATUS = {
# #     #                         "state": "failed",
# #     #                         "message": f"Azure ML experiment submission failed: {submit_error}"
# #     #                     }

# #     #             except Exception as config_error:
# #     #                 logging.error(f"❌ Failed to create ScriptRunConfig: {config_error}")
# #     #                 TRAINING_STATUS = {
# #     #                     "state": "failed",
# #     #                     "message": f"ScriptRunConfig creation failed: {config_error}"
# #     #                 }

# #     #         except Exception as datastore_error:
# #     #             logging.error(f"❌ Failed to access datastore: {datastore_error}")
# #     #             TRAINING_STATUS = {
# #     #                 "state": "failed",
# #     #                 "message": f"Datastore access failed: {datastore_error}"
# #     #             }



# #     #     except Exception as e:
# #     #         print(f"❌ Azure ML job submission failed: {e}")
# #     #         TRAINING_STATUS = {"state": "failed", "message": str(e)}

# #     #     finally:
# #     #         training_lock.release()

# #     def background_job(species_name: str, project_id: int):
# #         """Background Azure ML job submission using v1 SDK with Service Principal authentication."""
# #         global TRAINING_STATUS

# #         if not training_lock.acquire(blocking=False):
# #             logging.info("⚠️ Another training job is already running — skipping new one.")
# #             return

# #         try:
# #             TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}
# #             logging.info(f"🚀 Starting Azure ML training for project {project_id}, species: {species_name}")

# #             # --- Authenticate using Service Principal ---
# #             try:
# #                 sp_auth = ServicePrincipalAuthentication(
# #                     tenant_id=os.environ["AZURE_TENANT_ID"],
# #                     service_principal_id=os.environ["AZURE_CLIENT_ID"],
# #                     service_principal_password=os.environ["AZURE_CLIENT_SECRET"]
# #                 )
# #                 logging.info("✅ Authenticated successfully with Service Principal.")
# #             except KeyError as e:
# #                 missing_var = str(e)
# #                 logging.error(f"❌ Missing environment variable: {missing_var}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Missing environment variable: {missing_var}"
# #                 }
# #                 return

# #             # --- Connect to Azure ML Workspace ---
# #             try:
# #                 ws = Workspace(
# #                     subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
# #                     resource_group=os.environ["AZURE_RESOURCE_GROUP"],
# #                     workspace_name=os.environ["AZURE_ML_WORKSPACE"],
# #                     auth=sp_auth
# #                 )
# #                 logging.info(f"✅ Connected to Azure ML Workspace: {ws.name}")
# #             except Exception as ws_error:
# #                 logging.error(f"❌ Failed to connect to Azure ML Workspace: {ws_error}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Workspace connection failed: {ws_error}"
# #                 }
# #                 return

# #             # --- Get compute target ---
# #             try:
# #                 compute_target = ComputeTarget(workspace=ws, name="GflTrainCompute")
# #                 logging.info("✅ Compute target 'GflTrainCompute' found.")
# #             except Exception as compute_error:
# #                 logging.error(f"❌ Failed to locate compute target: {compute_error}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Compute target not found: {compute_error}"
# #                 }
# #                 return

# #             # --- Get environment ---
# #             try:
# #                 env = ws.environments["gflTrainEnv"]
# #                 logging.info("✅ Environment 'gflTrainEnv' loaded successfully.")
# #             except KeyError:
# #                 logging.error("❌ Environment 'gflTrainEnv' not found in workspace.")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": "Environment 'gflTrainEnv' not found in workspace."
# #                 }
# #                 return

# #             # --- Access datastore ---
# #             try:
# #                 logging.info("🔍 Retrieving workspace datastore...")
# #                 datastore = Datastore.get(ws, "workspaceblobstore")
# #                 logging.info(f"✅ Datastore retrieved: {datastore.name}")
# #             except Exception as ds_error:
# #                 logging.error(f"❌ Failed to access datastore: {ds_error}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Datastore access failed: {ds_error}"
# #                 }
# #                 return

# #             # --- Define job source and command ---
# #             source_directory = "UI/JobSubmission/10-23-2025_102845_UTC"
# #             logging.info(f"📁 Source directory: {source_directory}")

# #             command = [
# #                 "python", "train_pipeline_entry.py",
# #                 "--species", species_name,
# #                 "--project_id", str(project_id)
# #             ]
# #             logging.info(f"⚙️ Training command: {command}")

# #             # --- Create ScriptRunConfig ---
# #             try:
# #                 config = ScriptRunConfig(
# #                     source_directory=source_directory,
# #                     command=command,
# #                     compute_target=compute_target,
# #                     environment=env,
# #                 )
# #                 logging.info("✅ ScriptRunConfig created successfully.")
# #             except Exception as config_error:
# #                 logging.error(f"❌ Failed to create ScriptRunConfig: {config_error}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"ScriptRunConfig creation failed: {config_error}"
# #                 }
# #                 return

# #             # --- Submit experiment ---
# #             try:
# #                 exp = Experiment(ws, "GFLTrainingModelV2")
# #                 logging.info("🚀 Submitting job to Azure ML experiment...")
# #                 run = exp.submit(config)
# #                 run_url = run.get_portal_url()
# #                 logging.info(f"✅ Job submitted successfully. Portal: {run_url}")

# #                 TRAINING_STATUS = {
# #                     "state": "submitted",
# #                     "message": "Azure ML job submitted successfully.",
# #                     "run_id": run.id,
# #                     "portal_url": run_url
# #                 }

# #             except Exception as submit_error:
# #                 logging.error(f"❌ Experiment submission failed: {submit_error}")
# #                 TRAINING_STATUS = {
# #                     "state": "failed",
# #                     "message": f"Experiment submission failed: {submit_error}"
# #                 }

# #         except Exception as e:
# #             logging.exception("❌ Unhandled error during Azure ML job submission.")
# #             TRAINING_STATUS = {"state": "failed", "message": str(e)}

# #         finally:
# #             training_lock.release()

# #     # --- Launch background job ---
# #     thread = threading.Thread(target=background_job, args=(species_name, project_id), daemon=True)
# #     thread.start()

# #     return jsonify({
# #         "status": "started",
# #         "message": f"Training job submitted for project {project_id}, species {species_name}.",
# #         "check_status_at": "/pipeline_status"
# #     }), 202




# # # Global training status
# # TRAINING_STATUS = {
# #     "state": "idle",
# #     "message": "No training has been run yet."
# # }

# # # Simple lock to prevent concurrent runs
# # training_lock = threading.Lock()

# # @fish_bp.route("/run_pipeline", methods=["POST"])
# # def run_pipeline():
# #     """Run the full training pipeline in the background or via Azure ML."""
# #     global TRAINING_STATUS

# #     data = request.get_json(silent=True) or {}
# #     species_name = data.get("speciesName")
# #     project_id = data.get("projectId")

# #     if not species_name:
# #         return jsonify({"status": "error", "message": "speciesName is required"}), 400
# #     if not project_id:
# #         return jsonify({"status": "error", "message": "projectId is required"}), 400

# #     if TRAINING_STATUS["state"] == "running":
# #         return jsonify({
# #             "status": "busy",
# #             "message": "Training is already running. Please wait for it to complete before starting another."
# #         }), 429

# #     def background_job(species_name: str, project_id: int):
# #         global TRAINING_STATUS
# #         if not training_lock.acquire(blocking=False):
# #             print("⚠️ Another training job is already running — skipping new one.")
# #             return

# #         try:
# #             TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}
# #             print(f"🚀 Submitting Azure ML training for project {project_id}, species: {species_name}")

# #             credential = DefaultAzureCredential()
# #             ml_client = MLClient(
# #                 credential=credential,
# #                 subscription_id=AZURE_SUBSCRIPTION_ID,
# #                 resource_group_name=AZURE_RESOURCE_GROUP,
# #                 workspace_name=AZURE_ML_WORKSPACE,
                
# #             )


# #             entry_script_url = (
# #                 "https://gflstorageblob.blob.core.windows.net/"
# #                 "azureml-blobstore-bbd3a4dc-14ed-4039-80c7-7d0d2df1bd36/"
# #                 "UI/JobSubmission/10-23-2025_102845_UTC/"
# #             )

# #             ml_client.compute.begin_start("GflTrainCompute").wait(200)
# #             comp_start = ml_client.compute.begin_start("GflTrainCompute").done()


# #             job = ml_client.jobs.create_or_update(
# #                 command(
# #                     code=entry_script_url,
# #                     experiment_name="GFLTrainingModelV1",
# #                     command=f"python train_pipeline_entry.py --species '{species_name}' --project_id {project_id} && pip install -r requirements.txt",
# #                     environment="azureml://locations/eastus2/workspaces/bbd3a4dc-14ed-4039-80c7-7d0d2df1bd36/environments/gflTrainEnv/versions/5",
# #                     compute="GflTrainCompute",
# #                     display_name=f"fish_train_{species_name}_{project_id}",
# #                     description="YOLO fish species training job"
# #                 )
# #             )

# #             TRAINING_STATUS = {
# #                 "state": "submitted",
# #                 "message": f"✅ Azure ML job submitted: {job.name}"
# #             }
# #             print(f"✅ Submitted Azure ML job: {job.name}")

# #         except Exception as e:
# #             TRAINING_STATUS = {
# #                 "state": "failed",
# #                 "message": f"❌ Azure ML job submission failed: {str(e)}"
# #             }
# #             print(f"❌ Azure ML submission failed: {e}")

# #         finally:
# #             training_lock.release()

# #     thread = threading.Thread(target=background_job, args=(species_name, project_id), daemon=True)
# #     thread.start()

# #     return jsonify({
# #         "status": "started",
# #         "message": f"Training job submitted for project {project_id}, species {species_name}.",
# #         "check_status_at": "/pipeline_status"
# #     }), 202


# # @fish_bp.route("/run_pipeline", methods=["POST"])
# # def run_pipeline():
# #     """Run the full training pipeline in the background."""
# #     global TRAINING_STATUS

# #     data = request.get_json(silent=True) or {}
# #     species_name = data.get("speciesName")
# #     project_id = data.get("projectId")

# #     if not species_name:
# #         return jsonify({"status": "error", "message": "speciesName is required"}), 400
# #     if not project_id:
# #         return jsonify({"status": "error", "message": "projectId is required"}), 400

# #     # 🧠 Prevent starting if training is already running
# #     if TRAINING_STATUS["state"] == "running":
# #         return jsonify({
# #             "status": "busy",
# #             "message": "Training is already running. Please wait for it to complete before starting another."
# #         }), 429

# #     def background_job(species_name: str, project_id: int):
# #         global TRAINING_STATUS
# #         if not training_lock.acquire(blocking=False):
# #             print("⚠️ Another training job is already running — skipping new one.")
# #             return

# #         try:
# #             TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}
# #             print(f"🚀 Background training started for project {project_id}, species: {species_name}")

# #             # Make PROJECT_ID available to the training code
# #             os.environ["PROJECT_ID"] = str(project_id)

# #             # Run your training sequence
# #             asyncio.run(build_augmented_dataset(species_name=species_name))
# #             task1_run()
# #             task2_run()
# #             generate_filtered_group_dataset()
# #             train_all_groups()

# #             TRAINING_STATUS = {
# #                 "state": "completed",
# #                 "message": f"✅ Training completed successfully for project {project_id}, species {species_name}."
# #             }
# #             print(f"✅ Training completed for project {project_id}, species {species_name}")

# #         except Exception as e:
# #             TRAINING_STATUS = {
# #                 "state": "failed",
# #                 "message": f"❌ Training pipeline failed: {str(e)}"
# #             }
# #             print(f"❌ Training failed: {e}")

# #         finally:
# #             training_lock.release()

# #     # Start the background thread
# #     thread = threading.Thread(target=background_job, args=(species_name, project_id), daemon=True)
# #     thread.start()

# #     return jsonify({
# #         "status": "started",
# #         "message": f"Training pipeline started for project {project_id}, species {species_name}. "
# #                    f"Use /pipeline_status to check progress."
# #     }), 202


# @fish_bp.route("/pipeline_status", methods=["GET"])
# def pipeline_status():
#     """Check current training pipeline status."""
#     return jsonify(TRAINING_STATUS), 200

