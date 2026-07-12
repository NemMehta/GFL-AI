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
# from PIL import Image
import logging
from werkzeug.exceptions import RequestEntityTooLarge
from app_backend.config import IMG_SIZE, DB_FOLDER, DATASET_ROOT, BASE_PUBLIC_URL, DB_FOLDER_Unknown_Fish, LOCAL_PREDICTION_MODELS_DIR, PREDICTED_OUTPUT_DIR
from app_backend.services.uniqueness import compare_two_images
from app_backend.utils.helpers import ensure_db_folder
from app_backend.services.fish_sqlite import save_image_file, BASE_DIR 
from app_backend.database.db import DB_PATH_DATA_COLLECTION, insert_unknown_fish, init_db_for_data_collection
import sqlite3
from pathlib import Path
import json
import base64
from io import BytesIO
import requests
import asyncio
import threading
from flask import Flask, request, jsonify
# from ultralytics import YOLO
import os
import requests
# from PIL import Image
import io
from datetime import datetime
import logging
logger = logging.getLogger(__name__)
from azure.storage.blob import ContentSettings
from PIL import Image


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
from azure.storage.blob import ContentSettings, ContainerClient

import json
from flask import request

from pathlib import Path
import os
# from azure.storage.blob import BlobServiceClient
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


# from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobClient, BlobServiceClient
import uuid
import zipfile
# -------------------- Azure Config --------------------
# -------------------- Azure Config --------------------
AZURE_CONFIG = {
# 	# App maisters
#     "ConnectionString": "https://gflstorageaccount.blob.core.windows.net/dotnetbackend-container",
#     "Container": "dotnetbackend-container",
#     # "SasUrl": "https://gflstorageaccount.blob.core.windows.net/dotnetbackend-container?<sas-token>",
#     "SasToken": "sp=rcwd&st=2025-09-08T09:08:53Z&se=2026-09-08T17:23:53Z&spr=https&sv=2024-11-04&sr=c&sig=%2FPkcD%2FANv4T6EtPEAjrNw43HQ9Q6nTLM8%2BCkhxq6kw8%3D"

# client
    "ConnectionString": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container",
    "Container": "dotnetbackend-container",
    # "SasUrl": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container?sp=r&st=2025-10-14T12:00:00Z&se=2026-10-14T20:15:00Z&spr=https&sv=2024-11-04&sr=c&sig=6haTCzklALCl4RSQtVPC5f2F9RZvx6CVA6KtMFWkmUE%3D%22,
    "SasToken": "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2026-01-31T16:03:23Z&st=2025-12-03T07:48:23Z&spr=https&sig=XeBVuCefwPKVMc2loUsAvi8bXU1klfEi89jmT1fZBjI%3D"
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



# ================================
#  MODEL STORAGE & AZURE CONFIG
# ================================


# =========================================================
#   CONFIGURATION
# =========================================================


# =========================================================
#  MODEL STORAGE & SYNC LOGIC
# =========================================================

LOCAL_MODEL_DIR = "/home/site/wwwroot/app_backend/model_cache"
PREDICTION_DIR = "/home/site/wwwroot/app_backend/predictions"

BLOB_MODEL_FOLDER = "models/"
AZURE_CONTAINER_URL = f"{AZURE_CONFIG['ConnectionString']}?{AZURE_CONFIG['SasToken']}"

_cached_model = None
_cached_model_path = None


def get_latest_blob_model():
    container = ContainerClient.from_container_url(AZURE_CONTAINER_URL)
    models = [
        b for b in container.list_blobs(name_starts_with=BLOB_MODEL_FOLDER)
        if b.name.endswith(".pt")
    ]
    if not models:
        return None, None

    models.sort(key=lambda x: x.last_modified, reverse=True)
    return models[0].name, models[0].last_modified


def download_latest_model_from_blob():
    blob_name, _ = get_latest_blob_model()
    if not blob_name:
        raise Exception("❌ No model found in Blob Storage")

    print(f"⬇ Downloading model: {blob_name}")
    container = ContainerClient.from_container_url(AZURE_CONTAINER_URL)
    blob_client = container.get_blob_client(blob_name)

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    local_path = Path(LOCAL_MODEL_DIR) / blob_name.split("/")[-1]

    with open(local_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    print(f"✔ Model downloaded → {local_path}")
    return str(local_path)


def get_latest_local_model():
    files = sorted(Path(LOCAL_MODEL_DIR).glob("*.pt"), reverse=True)
    return str(files[0]) if files else None


def sync_model():
    local_model = get_latest_local_model()
    blob_model, _ = get_latest_blob_model()

    if not local_model:
        print("⚠ No local model → downloading...")
        return download_latest_model_from_blob()

    container = ContainerClient.from_container_url(AZURE_CONTAINER_URL)
    blob_props = container.get_blob_client(blob_model).get_blob_properties()

    # --- FIX: Handle timezone mismatch ---
    from datetime import timezone

    blob_time = blob_props.last_modified  # tz-aware
    local_timestamp = os.path.getmtime(local_model)
    local_time = datetime.fromtimestamp(local_timestamp, tz=timezone.utc)

    if blob_time > local_time:
        print("🔄 Blob has newer model → downloading update...")
        return download_latest_model_from_blob()

    print("✔ Local model is latest")
    return local_model

def load_model_once():
    """
    Ensures model is synced, then loads it once per worker.
    """
    global _cached_model, _cached_model_path

    model_path = sync_model()

    if _cached_model is None or _cached_model_path != model_path:
        print("🔥 Loading YOLO:", model_path)
        _cached_model = YOLO(model_path)
        _cached_model_path = model_path

    return _cached_model


@fish_bp.route("/detect_fish", methods=["POST"])
def detect_fish_single_model():
    try:
        if "image" not in request.files:
            return jsonify({"status": False, "message": "No image uploaded"}), 400

        # Model sync happens automatically inside load_model_once()
        model = load_model_once()

        # Read image
        image_file = request.files["image"]
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

        # Predict
        results = model.predict(image, imgsz=640, conf=0.25)
        names = model.names

        detections = []
        best = {"class": "Not detected", "confidence": 0.0}

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = names[cls_id]

            detections.append({
                "class": cls_name,
                "confidence": round(conf, 3)
            })

            if conf > best["confidence"]:
                best = {"class": cls_name, "confidence": round(conf, 3)}

        os.makedirs(PREDICTION_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{PREDICTION_DIR}/prediction_{timestamp}.jpg"

        annotated = results[0].plot()
        cv2.imwrite(output_path, annotated)

        return jsonify({
            "status": True,
            "message": "Detection complete",
            "best_result": {
                "class": best["class"],
                "confidence": best["confidence"],
                "model": Path(_cached_model_path).name,
                "output_image": output_path
            },
            "detections": detections
        })

    except Exception as e:
        logger.exception("❌ Detect Fish Error")
        return jsonify({"status": False, "message": str(e)}), 500





# get species names from yaml file



def download_dataset_yaml() -> Path:
    """
    Downloads augmentation/training_data/dataset.yaml from Azure Blob
    and returns local file path.
    """
    blob_url = (
        "https://gflstorageblob.blob.core.windows.net/"
        "dotnetbackend-container/augmentation/training_data/dataset.yaml"
    )

    sas = AZURE_CONFIG["SasToken"]
    blob_client = BlobClient.from_blob_url(f"{blob_url}?{sas}")

    tmp_dir = Path(tempfile.mkdtemp())
    yaml_path = tmp_dir / "dataset.yaml"

    with open(yaml_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    return yaml_path


@fish_bp.route("/dataset/species_yaml", methods=["GET"])
def get_dataset_species_yaml():
    """
    Reads dataset.yaml from Azure Blob and returns species list.
    """
    try:
        yaml_path = download_dataset_yaml()

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        names = data.get("names", [])
        if not names:
            return jsonify({
                "count": 0,
                "species": [],
                "message": "No species found in dataset.yaml"
            }), 200

        species_list = [
            {"id": idx, "name": name}
            for idx, name in enumerate(names)
        ]

        return jsonify({
            "count": len(species_list),
            "species": species_list
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Failed to load dataset.yaml",
            "details": str(e)
        }), 500


training_lock = threading.Lock()
TRAINING_STATUS = {"state": "idle", "message": "No training running."}



# @fish_bp.route("/run_pipeline", methods=["POST"])
# def run_pipeline():
#     """Run the full training pipeline via Azure ML (v1 SDK)."""
#     global TRAINING_STATUS

#     data = request.get_json(silent=True) or {}
#     species_name = data.get("speciesName")
#     project_id = data.get("projectId")

#     # --- Validation ---
#     if not species_name:
#         return jsonify({"status": "error", "message": "speciesName is required"}), 400
#     if not project_id:
#         return jsonify({"status": "error", "message": "projectId is required"}), 400

#     if TRAINING_STATUS.get("state") == "running":
#         return jsonify({
#             "status": "busy",
#             "message": "Training is already running. Please wait for it to complete."
#         }), 429
        



#     def background_job(species_name: str, project_id: int):
#         """Background Azure ML job submission using v1 SDK with Service Principal authentication."""
#         global TRAINING_STATUS

#         if not training_lock.acquire(blocking=False):
#             logging.info("⚠️ Another training job is already running — skipping new one.")
#             return

#         try:
#             TRAINING_STATUS = {"state": "running", "message": "Training in progress..."}
#             logging.info(f"🚀 Starting Azure ML training for project {project_id}, species: {species_name}")

#             # --- Authenticate using Service Principal ---
#             try:
#                 sp_auth = ServicePrincipalAuthentication(
#                     tenant_id=os.environ["AZURE_TENANT_ID"],
#                     service_principal_id=os.environ["AZURE_CLIENT_ID"],
#                     service_principal_password=os.environ["AZURE_CLIENT_SECRET"]
#                 )
#                 logging.info("✅ Authenticated successfully with Service Principal.")
#             except KeyError as e:
#                 missing_var = str(e)
#                 logging.error(f"❌ Missing environment variable: {missing_var}")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"Missing environment variable: {missing_var}"
#                 }
#                 return

#             # --- Connect to Azure ML Workspace ---
#             try:
#                 ws = Workspace(
#                     subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
#                     resource_group=os.environ["AZURE_RESOURCE_GROUP"],
#                     workspace_name=os.environ["AZURE_ML_WORKSPACE"],
#                     auth=sp_auth
#                 )
#                 logging.info(f"✅ Connected to Azure ML Workspace: {ws.name}")
#             except Exception as ws_error:
#                 logging.error(f"❌ Failed to connect to Azure ML Workspace: {ws_error}")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"Workspace connection failed: {ws_error}"
#                 }
#                 return

#             # --- Get compute target ---
#             try:
#                 compute_target = ComputeTarget(workspace=ws, name="GflTrainCompute")
#                 logging.info("✅ Compute target 'GflTrainCompute' found.")
#             except Exception as compute_error:
#                 logging.error(f"❌ Failed to locate compute target: {compute_error}")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"Compute target not found: {compute_error}"
#                 }
#                 return

#             # --- Get environment ---
#             try:
#                 env = ws.environments["gflTrainEnv"]
#                 logging.info("✅ Environment 'gflTrainEnv' loaded successfully.")
#             except KeyError:
#                 logging.error("❌ Environment 'gflTrainEnv' not found in workspace.")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": "Environment 'gflTrainEnv' not found in workspace."
#                 }
#                 return

#             # --- Access datastore ---
#             try:
#                 logging.info("🔍 Retrieving workspace datastore...")
#                 datastore = Datastore.get(ws, "workspaceblobstore")
#                 logging.info(f"✅ Datastore retrieved: {datastore.name}")
#             except Exception as ds_error:
#                 logging.error(f"❌ Failed to access datastore: {ds_error}")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"Datastore access failed: {ds_error}"
#                 }
#                 return

#             # --- Define job source and command ---
#             source_directory = "UI/JobSubmission/10-23-2025_102845_UTC"
#             logging.info(f"📁 Source directory: {source_directory}")

#             command = [
#                 "python", "train_pipeline_entry.py",
#                 "--species", species_name,
#                 "--project_id", str(project_id)
#             ]
#             logging.info(f"⚙️ Training command: {command}")

#             # --- Create ScriptRunConfig ---
#             try:
#                 config = ScriptRunConfig(
#                     source_directory=source_directory,
#                     command=command,
#                     compute_target=compute_target,
#                     environment=env,
#                 )
#                 logging.info("✅ ScriptRunConfig created successfully.")
#             except Exception as config_error:
#                 logging.error(f"❌ Failed to create ScriptRunConfig: {config_error}")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"ScriptRunConfig creation failed: {config_error}"
#                 }
#                 return

#             # --- Submit experiment ---
#             try:
#                 exp = Experiment(ws, "GFLTrainingModelV2")
#                 logging.info("🚀 Submitting job to Azure ML experiment...")
#                 run = exp.submit(config)
#                 run_url = run.get_portal_url()
#                 logging.info(f"✅ Job submitted successfully. Portal: {run_url}")

#                 TRAINING_STATUS = {
#                     "state": "submitted",
#                     "message": "Azure ML job submitted successfully.",
#                     "run_id": run.id,
#                     "portal_url": run_url
#                 }

#             except Exception as submit_error:
#                 logging.error(f"❌ Experiment submission failed: {submit_error}")
#                 TRAINING_STATUS = {
#                     "state": "failed",
#                     "message": f"Experiment submission failed: {submit_error}"
#                 }

#         except Exception as e:
#             logging.exception("❌ Unhandled error during Azure ML job submission.")
#             TRAINING_STATUS = {"state": "failed", "message": str(e)}

#         finally:
#             training_lock.release()


#     # --- Launch background job ---
#     thread = threading.Thread(target=background_job, args=(species_name, project_id), daemon=True)
#     thread.start()

#     return jsonify({
#         "status": "started",
#         "message": f"Training job submitted for project {project_id}, species {species_name}.",
#         "check_status_at": "/pipeline_status"
#     }), 202




# # working on single model
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

#                 compute_name = "GFL-GPU"
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
#             # data_asset = ml_client.data.get("training_pipeline", version="2")
#             code_input = Input(
#                 type="uri_folder",
#                 # path=data_asset.path
#                 path="azureml://subscriptions/417ecdbc-9b50-421a-82c2-c45f29686df1/"
#                      "resourcegroups/GFL-App-Rg/workspaces/GFL-MLAI/"
#                      "datastores/workspaceblobstore/paths/UI/JobSubmission/11-25-2025_120606_UTC/"
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
#                     "BLOB_SAS_TOKEN": "sp=rcwd&st=2025-09-08T09:08:53Z&se=2026-09-08T17:23:53Z&spr=https&sv=2024-11-04&sr=c&sig=%2FPkcD%2FANv4T6EtPEAjrNw43HQ9Q6nTLM8%2BCkhxq6kw8%3D"
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





# ---------------------- GLOBAL TRAINING STATE ----------------------
TRAINING_STATUS = {
    "state": "idle",
    "message": "No active training job"
}

training_lock = threading.Lock()

logger = logging.getLogger(__name__)





from pathlib import Path
import tempfile

TEMP_DATA_DIR = Path(tempfile.gettempdir()) / "gfl_training"


import re


# -----------------------------------------------------
# Species name normalization
# -----------------------------------------------------
def normalize_species_to_name_format(species: str) -> str:

    species = species.lower()
    species = species.replace("(", " ").replace(")", " ")
    species = re.sub(r"[^a-z0-9\s]", "", species)
    species = re.sub(r"\s+", "_", species).strip("_")
    return species


# -----------------------------------------------------
# Download YAML from Blob
# -----------------------------------------------------
def download_species_yaml() -> Path:
    blob_url = (
        "https://gflstorageblob.blob.core.windows.net/"
        "dotnetbackend-container/augmentation/training_data/data.yaml"
    )

    TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_yaml_path = TEMP_DATA_DIR / "species_data.yaml"

    logger.info(f"⬇ Downloading species YAML from: {blob_url}")
    response = requests.get(blob_url, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to download YAML: HTTP {response.status_code}")

    with open(local_yaml_path, "wb") as f:
        f.write(response.content)

    logger.info(f"📁 Saved species YAML to: {local_yaml_path}")
    return local_yaml_path




import yaml

# -----------------------------------------------------
# Load YOLO species names from YAML
# -----------------------------------------------------
def load_species_from_class_map() -> dict[str, int]:
    """
    Returns:
      {
        "Snook (Centropomus undecimalis)": 10,
        "Blacktip Kingfish (Blacktip Trevally) (Caranx heberi)": 69,
        ...
      }
    """
    yaml_path = download_species_yaml()

    with open(yaml_path, "r") as f:
        yaml_content = yaml.safe_load(f)

    class_map = yaml_content.get("class_map")
    if not class_map or not isinstance(class_map, dict):
        raise ValueError("class_map not found in YAML")

    # Normalize keys to strings, IDs to ints
    species_map = {
        str(species_name): int(class_id)
        for class_id, species_name in class_map.items()
    }

    logger.info(f"✅ Loaded {len(species_map)} species from YAML (class_map)")
    return species_map

# -----------------------------------------------------
# Validate & normalize species from API input
# -----------------------------------------------------
def validate_and_normalize_species(entries: list[dict]) -> list[str]:
    """
    Validates API species names against YAML `names`
    and returns a list of normalized YOLO species names.
    """
    yaml_name_set = load_species_names_from_yaml()

    normalized_species = []

    for e in entries:
        raw_name = e["speciesName"]
        normalized_name = normalize_species_to_name_format(raw_name)

        if normalized_name not in yaml_name_set:
            raise ValueError(
                f"Invalid species '{raw_name}' "
                f"(normalized='{normalized_name}') — not found in YAML names"
            )

        if normalized_name not in normalized_species:
            normalized_species.append(normalized_name)

    logger.info(f"🐟 Validated species: {normalized_species}")
    return normalized_species


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

    for entry in data:
        if not entry.get("speciesName") or not entry.get("projectId"):
            return jsonify({
                "status": "error",
                "message": "Each entry must contain speciesName and projectId"
            }), 400

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
            return

        try:
            TRAINING_STATUS = {
                "state": "running",
                "message": "Training started..."
            }

            # -----------------------------------------
            # STEP A — Load species from class_map
            # -----------------------------------------
            species_map = load_species_from_class_map()
            valid_species_names = set(species_map.keys())

            species_list = []
            species_to_projects = {}

            for e in entries:
                sp = e["speciesName"]
                pid = int(e["projectId"])

                if sp not in valid_species_names:
                    raise ValueError(
                        f"Invalid species '{sp}' — not found in species YAML class_map"
                    )

                if sp not in species_list:
                    species_list.append(sp)

                species_to_projects.setdefault(sp, [])
                if pid not in species_to_projects[sp]:
                    species_to_projects[sp].append(pid)

            logger.info(f"🐟 Species selected: {species_list}")
            logger.info(f"📦 Species → Projects: {species_to_projects}")

            train_species_json = json.dumps(species_list)
            species_project_map_json = json.dumps(species_to_projects)

            # -----------------------------------------
            # SUBMIT AZURE ML JOB
            # -----------------------------------------
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=AZURE_SUBSCRIPTION_ID,
                resource_group_name=AZURE_RESOURCE_GROUP,
                workspace_name=AZURE_ML_WORKSPACE,
            )

            compute_name = "GFL-GPU-Max"
            compute = ml_client.compute.get(compute_name)

            code_input = Input(
                type="uri_folder",
                path="azureml://subscriptions/417ecdbc-9b50-421a-82c2-c45f29686df1/"
                     "resourcegroups/GFL-App-Rg/workspaces/GFL-MLAI/"
                     "datastores/workspaceblobstore/paths/UI/JobSubmission/11-25-2025_120606_UTC/"
            )

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
                environment_variables={
                    "TRAIN_SPECIES": train_species_json,
                    "SPECIES_PROJECT_MAP": species_project_map_json,
                }
            )

            submitted_job = ml_client.jobs.create_or_update(job)

            TRAINING_STATUS = {
                "state": "submitted",
                "message": f"Azure ML job submitted: {submitted_job.name}",
                "job_name": submitted_job.name,
                "species": species_list
            }

        except Exception as e:
            logger.exception("❌ Training pipeline failed")
            TRAINING_STATUS = {
                "state": "failed",
                "message": str(e)
            }

        finally:
            training_lock.release()

    threading.Thread(target=background_job, args=(data,), daemon=True).start()

    return jsonify({
        "status": "started",
        "message": f"Training pipeline started for {len(data)} species."
    }), 202



# ================================================================
# Status API
# ================================================================
@fish_bp.route("/pipeline_status", methods=["GET"])
def pipeline_status():
    return jsonify(TRAINING_STATUS), 200



