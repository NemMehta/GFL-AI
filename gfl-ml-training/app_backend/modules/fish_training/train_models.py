









import shutil
import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
import logging
import requests

from ultralytics import YOLO
from azure.storage.blob import ContainerClient

from .config import (
    TEMP_DATA_DIR,
    MODELS_DIR,
    BASE_MODEL,
    EPOCHS,
    IMGSZ,
    BATCH,
    GROUP_HASHES_CURR_PATH,
    GROUP_HASHES_PREV_PATH,
)
from app_backend.config import AZURE_CONFIG
from .db_utils import insert_model_record

logger = logging.getLogger(__name__)


# ----------------------- DEVICE SELECTOR -----------------------
try:
    import torch
except ImportError:
    torch = None


def get_training_device():
    env_device = os.getenv("TRAIN_DEVICE")
    if env_device:
        return env_device

    if torch is None:
        return "cpu"

    if torch.cuda.is_available():
        return "0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


# -------------------------- HELPERS -----------------------------
def load_json(path: Path):
    if path.exists():
        try:
            return json.load(open(path, "r"))
        except Exception:
            logger.exception(f"Failed loading JSON {path}")
    return {}



def archive_old_models_in_blob(new_model_name: str):
    """
    Moves all models except the newest one from:
       /models/  →  /models_previous/
    """

    container = ContainerClient.from_container_url(
        f"{AZURE_CONFIG['ConnectionString']}?{AZURE_CONFIG['SasToken']}"
    )

    models_prefix = "models/"
    archive_prefix = "models_previous/"

    # Ensure archive folder exists (Blob creates automatically on upload)
    print("📦 Checking models directory for old models...")

    blobs = [
        blob for blob in container.list_blobs(name_starts_with=models_prefix)
        if blob.name.endswith(".pt")
    ]

    if not blobs:
        print("⚠ No models found to archive.")
        return

    # Step 1: Upload a dummy file to create the models_previous folder (if not already created)
    dummy_file_path = "dummy_file.txt"  # Create a dummy file
    dummy_blob_name = f"{archive_prefix}/.dummyfile"  # Upload to the models_previous directory
    try:
        # Upload dummy file to trigger folder creation
        container.upload_blob(dummy_blob_name, "Dummy content", overwrite=True)
        print(f"📁 Dummy file uploaded to {dummy_blob_name} to create the folder.")
    except Exception as e:
        print(f"⚠ Failed to upload dummy file for folder creation: {e}")

    for blob in blobs:
        if blob.name == new_model_name:
            print(f"📦 Checking model {new_model_name}")
            continue  # Skip newest model

        old_blob_path = blob.name
        archive_blob_path = old_blob_path.replace(models_prefix, archive_prefix)

        print(f"📁 Archiving old model → {archive_blob_path}")

        # Copy old model to archive
        # source_blob_url = f"{container.url}/{old_blob_path}"
        source_blob_url = f"{container.url}/{old_blob_path}?{AZURE_CONFIG['SasToken']}"

        archive_blob_client = container.get_blob_client(archive_blob_path)

        try:
            archive_blob_client.start_copy_from_url(source_blob_url)
            print(f"✔ Copied old model to archive: {archive_blob_path}")

            # Delete original model
            container.get_blob_client(old_blob_path).delete_blob()
            print(f"✔ Archived & removed old model: {old_blob_path}")

        except Exception as e:
            logger.exception(f"Failed to archive model {old_blob_path}: {e}")


async def upload_file_async(file_path: Path, blob_prefix: str, config: dict) -> str:
    try:
        container_client = ContainerClient.from_container_url(
            f"{config['ConnectionString']}?{config['SasToken']}"
        )
        blob_name = f"{blob_prefix}/{file_path.name}"
        blob_client = container_client.get_blob_client(blob_name)

        with open(file_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)

        url = f"{config['ConnectionString']}/{blob_name}"
        logger.info(f"Uploaded {blob_prefix}: {url}")
        return url
    except Exception:
        logger.exception(f"Failed to upload file {file_path}")
        return ""


# ============================================================
# Save trained model to App Service (local) & delete old ones
# ============================================================

def store_model_in_app_service(source_model_path: Path):
    """
    Saves the newly trained model into Azure App Service local storage,
    deletes old models, and keeps only a single global_model.pt file.
    """
    try:
        APP_SERVICE_MODEL_DIR = Path("/home/site/wwwroot/app_backend/model_cache")
        APP_SERVICE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Delete old models
        for file in APP_SERVICE_MODEL_DIR.glob("*.pt"):
            try:
                file.unlink()
                logger.info(f"🧹 Deleted old local model: {file}")
            except Exception as e:
                logger.warning(f"⚠ Failed to delete old model {file}: {e}")

        # Copy new model
        local_model_path = APP_SERVICE_MODEL_DIR / "Model.pt"
        shutil.copy(source_model_path, local_model_path)

        logger.info(f"📌 NEW model stored in App Service: {local_model_path}")
        return str(local_model_path)

    except Exception as e:
        logger.exception(f"❌ Failed storing model in App Service: {e}")
        return None


from azure.storage.blob import BlobClient
import yaml as pyyaml

# def download_species_yaml():
#     """
#     Downloads data.yaml from Azure Blob Storage into TEMP_DATA_DIR/species_data.yaml
#     """
#     logger.info("📥 Downloading species_data.yaml from Azure Blob...")

#     blob_url = (
#         "https://gflstorageblob.blob.core.windows.net/"
#         "dotnetbackend-container/augmentation/training_data/data.yaml"
#     )

#     sas = AZURE_CONFIG["SasToken"]

#     # Create blob client using full SAS URL
#     blob_client = BlobClient.from_blob_url(f"{blob_url}?{sas}")

#     # 🔥 STORE LOCALLY AS species_data.yaml
#     local_yaml_path = TEMP_DATA_DIR / "species_data.yaml"
#     local_yaml_path.parent.mkdir(parents=True, exist_ok=True)

#     with open(local_yaml_path, "wb") as file:
#         file.write(blob_client.download_blob().readall())

#     logger.info(f"📌 species_data.yaml saved to: {local_yaml_path}")

#     return local_yaml_path


def download_species_yaml() -> Path:
    """
    Downloads augmentation/training_data/data.yaml
    (GLOBAL SPECIES SCHEMA) into TEMP_DATA_DIR/species_data.yaml
    """
    logger.info("📥 Downloading species schema (data.yaml) from Azure...")

    blob_url = (
        "https://gflstorageblob.blob.core.windows.net/"
        "dotnetbackend-container/augmentation/training_data/data.yaml"
    )

    sas = AZURE_CONFIG["SasToken"]
    blob_client = BlobClient.from_blob_url(f"{blob_url}?{sas}")

    local_yaml_path = TEMP_DATA_DIR / "species_data.yaml"
    local_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(local_yaml_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    logger.info(f"📌 Species schema saved → {local_yaml_path}")
    return local_yaml_path




def notify_training_result(results: list):
    """
    Send an array of training results to the API.
    """
    # api_url = "https://gfladmin-stage.azurewebsites.net/api/TrainResult/Train-Result"
    # api_url = "https://gfl-admin.appmaister.com/api/TrainResult/Train-Result"
    api_url = "https://gfladmin-stage.azurewebsites.net/api/TrainResult/Train-Result"

    try:
        resp = requests.post(api_url, json=results, timeout=30)
        resp.raise_for_status()
        logger.info(f"Training results sent successfully: {resp.text}")
    except Exception as e:
        logger.exception(f"Notify failed: {e}")




# ---------------------------------------------------------------
# MAIN TRAINING
# ---------------------------------------------------------------
# def train_single_model():
#     logger.info("🚀 Starting SINGLE MODEL training...")

#     # -------------------------------------------------------------------
#     # FIXED: use correct dataset.yaml created by augmentation pipeline
#     # -------------------------------------------------------------------
#     data_yaml = TEMP_DATA_DIR / "dataset.yaml"

#     if not data_yaml.exists():
#         logger.error(f"❌ dataset.yaml NOT FOUND at: {data_yaml}")
#         raise FileNotFoundError("dataset.yaml missing — augmentation step failed")

#     logger.info(f"📌 Using dataset.yaml for training: {data_yaml}")

#     # HASH check (optional but preserved)
#     prev_hash = load_json(GROUP_HASHES_PREV_PATH)
#     curr_hash = load_json(GROUP_HASHES_CURR_PATH)

#     if prev_hash == curr_hash and list(MODELS_DIR.glob("Model_*.pt")):
#         logger.info("✅ Dataset unchanged — skipping retraining.")
#         return

#     MODELS_DIR.mkdir(parents=True, exist_ok=True)
#     device = get_training_device()

#     existing_models = sorted(MODELS_DIR.glob("Model_*.pt"), reverse=True)
#     start_from = str(existing_models[0]) if existing_models else BASE_MODEL

#     logger.info(f"Training starting model: {start_from}")
#     logger.info(f"YOLO Data YAML: {data_yaml}")
#     logger.info(f"Device: {device}")

#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     final_model_path = MODELS_DIR / f"Model_{timestamp}.pt"

#     tmp_dir = MODELS_DIR / "tmp_global"
#     tmp_dir.mkdir(exist_ok=True)

#     try:
#         model = YOLO(start_from)

#         model.train(
#             data=str(data_yaml),
#             epochs=EPOCHS,
#             imgsz=IMGSZ,
#             batch=BATCH,
#             project=str(MODELS_DIR),
#             name="tmp_global",
#             exist_ok=True,
#             device=device,
#             workers=0,
#             save=True,
#         )

#         # ---------------------------------------------------------------
#         # MOVE BEST MODEL OUT
#         # ---------------------------------------------------------------
#         best_pt = tmp_dir / "weights" / "best.pt"
#         if not best_pt.exists():
#             logger.error("❌ No best.pt found — training failed.")
#             return

#         shutil.move(str(best_pt), str(final_model_path))
#         logger.info(f"✔ Saved final model: {final_model_path}")

#         # LOCAL COPY
#         local_cache_path = store_model_in_app_service(final_model_path)

#         # UPLOAD MODEL
#         new_model_url = asyncio.run(upload_file_async(final_model_path, "models", AZURE_CONFIG))
#         new_blob_name = f"models/{final_model_path.name}"

#         archive_old_models_in_blob(new_blob_name)
#         model_url = new_model_url

#         # UPLOAD METADATA
#         json_url = asyncio.run(upload_file_async(GROUP_HASHES_CURR_PATH, "json", AZURE_CONFIG))
#         yaml_url = asyncio.run(upload_file_async(data_yaml, "augmentation/training_data", AZURE_CONFIG))

#         # SPECIES → PROJECT MAPPING
#         species_to_projects = json.loads(os.getenv("SPECIES_PROJECT_MAP", "{}"))
#         if not species_to_projects:
#             raise ValueError("SPECIES_PROJECT_MAP missing")

#         response_array = []
#         for index, (species_name, project_ids) in enumerate(species_to_projects.items()):
#             for pid in project_ids:
#                 response_array.append({
#                     "index": index,
#                     "projectId": pid,
#                     "modelPath": model_url,
#                     "jsonPath": json_url,
#                     "status": "Success"
#                 })

#         shutil.copy(GROUP_HASHES_CURR_PATH, GROUP_HASHES_PREV_PATH)

#         # Extract class_map from yaml
#         with open(data_yaml, "r") as f:
#             yaml_content = pyyaml.safe_load(f)

#         class_map = yaml_content.get("class_map", {})
#         speciesIndexes = [
#             {"speciesName": class_map[key], "index": int(key)}
#             for key in sorted(class_map.keys(), key=int)
#         ]

#         # FINAL API PAYLOAD
#         api_payload = {
#             "results": response_array,
#             "speciesIndexes": speciesIndexes,
#             "modelName": final_model_path.name,
#             "message": "Model trained and uploaded successfully"
#         }

#         notify_training_result(api_payload)

#     except Exception as e:
#         logger.exception(f"❌ Training failed: {e}")

#     logger.info("🏁 Training complete.")


# def train_single_model():
#     logger.info("🚀 Starting SINGLE MODEL (ALL SPECIES) training...")

#     # -----------------------------------------------------------
#     # YOLO TRAINING CONFIG (LOCAL ONLY)
#     # -----------------------------------------------------------
#     dataset_yaml = TEMP_DATA_DIR / "data.yaml"

#     if not dataset_yaml.exists():
#         raise FileNotFoundError(f"❌ dataset.yaml missing at {dataset_yaml}")

#     logger.info(f"📌 Using YOLO dataset config: {dataset_yaml}")

#     # -----------------------------------------------------------
#     # HASH CHECK (optional skip)
#     # -----------------------------------------------------------
#     prev_hash = _load_json(GROUP_HASHES_PREV_PATH)
#     curr_hash = _load_json(GROUP_HASHES_CURR_PATH)

#     if prev_hash == curr_hash and list(MODELS_DIR.glob("Model_*.pt")):
#         logger.info("✅ Dataset unchanged — skipping retraining.")
#         return

#     MODELS_DIR.mkdir(parents=True, exist_ok=True)

#     device = _get_training_device()

#     existing_models = sorted(MODELS_DIR.glob("Model_*.pt"), reverse=True)
#     start_from = str(existing_models[0]) if existing_models else BASE_MODEL

#     logger.info(f"🔁 Starting from model: {start_from}")
#     logger.info(f"🖥 Device: {device}")

#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     final_model_path = MODELS_DIR / f"Model_{timestamp}.pt"

#     tmp_dir = MODELS_DIR / "tmp_global"
#     tmp_dir.mkdir(exist_ok=True)

#     try:
#         # -------------------------------------------------------
#         # TRAIN
#         # -------------------------------------------------------
#         num_workers = max(2, min(8, (os.cpu_count() or 4) - 1))

#         model = YOLO(start_from)

#         model.train(
#             data=str(dataset_yaml),
#             epochs=EPOCHS,
#             imgsz=IMGSZ,
#             batch=BATCH,
#             project=str(MODELS_DIR),
#             name="tmp_global",
#             exist_ok=True,
#             device=device,
#             workers=num_workers,
#             save=True,
#         )

#         best_pt = tmp_dir / "weights" / "best.pt"
#         if not best_pt.exists():
#             raise RuntimeError("❌ best.pt not found — training failed")

#         shutil.move(str(best_pt), str(final_model_path))
#         logger.info(f"✔ Final model saved → {final_model_path}")

#         # -------------------------------------------------------
#         # STORE LOCALLY + UPLOAD MODEL
#         # -------------------------------------------------------
#         _store_model_in_app_service(final_model_path)

#         model_url = asyncio.run(
#             _upload_file_async(final_model_path, "models", AZURE_CONFIG)
#         )

#         # -------------------------------------------------------
#         # LOAD SPECIES SCHEMA (CRITICAL FIX)
#         # -------------------------------------------------------
#         species_yaml_path = download_species_yaml()

#         with open(species_yaml_path, "r", encoding="utf-8") as f:
#             species_yaml = pyyaml.safe_load(f) or {}

#         class_map = species_yaml.get("class_map", {})
#         if not class_map:
#             raise RuntimeError("❌ class_map missing in data.yaml")

#         speciesIndexes = [
#             {"index": int(i), "speciesName": class_map[str(i)]}
#             for i in sorted(map(int, class_map.keys()))
#         ]

#         if not speciesIndexes:
#             raise RuntimeError("❌ speciesIndexes EMPTY — aborting publish")

#         # -------------------------------------------------------
#         # BUILD API PAYLOAD
#         # -------------------------------------------------------
#         species_to_projects = json.loads(os.getenv("SPECIES_PROJECT_MAP", "{}"))
#         if not species_to_projects:
#             raise RuntimeError("❌ SPECIES_PROJECT_MAP missing")

#         response_array = []
#         for idx, (_, project_ids) in enumerate(species_to_projects.items()):
#             for pid in project_ids:
#                 response_array.append({
#                     "index": idx,
#                     "projectId": pid,
#                     "modelPath": model_url,
#                     "jsonPath": "",
#                     "status": "Success"
#                 })

#         shutil.copy(GROUP_HASHES_CURR_PATH, GROUP_HASHES_PREV_PATH)

#         payload = {
#             "results": response_array,
#             "speciesIndexes": speciesIndexes,
#             "modelName": final_model_path.name,
#             "message": "Model trained and uploaded successfully"
#         }

#         notify_training_result(payload)

#     except Exception as e:
#         logger.exception(f"❌ Training failed: {e}")
#         raise

#     logger.info("🏁 Training complete.")



def train_single_model():
    logger.info("🚀 Starting SINGLE MODEL (ALL SPECIES) training...")

    # -----------------------------------------------------------
    # DATASET CONFIG (YOLO)
    # -----------------------------------------------------------
    dataset_yaml = TEMP_DATA_DIR / "data.yaml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"❌ dataset.yaml missing at {dataset_yaml}")

    logger.info(f"📌 Using YOLO dataset config: {dataset_yaml}")

    # -----------------------------------------------------------
    # HASH CHECK (OPTIONAL SKIP)
    # -----------------------------------------------------------
    prev_hash = _load_json(GROUP_HASHES_PREV_PATH)
    curr_hash = _load_json(GROUP_HASHES_CURR_PATH)

    if prev_hash == curr_hash and list(MODELS_DIR.glob("Model_*.pt")):
        logger.info("✅ Dataset unchanged — skipping retraining.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = _get_training_device()

    existing_models = sorted(MODELS_DIR.glob("Model_*.pt"), reverse=True)
    start_from = str(existing_models[0]) if existing_models else BASE_MODEL

    logger.info(f"🔁 Starting from model: {start_from}")
    logger.info(f"🖥 Training device: {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_model_path = MODELS_DIR / f"Model_{timestamp}.pt"

    tmp_dir = MODELS_DIR / "tmp_global"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # -------------------------------------------------------
        # TRAIN (AZURE-SAFE CONFIG)
        # -------------------------------------------------------
        model = YOLO(start_from)

        model.train(
            data=str(dataset_yaml),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            project=str(MODELS_DIR),
            name="tmp_global",
            exist_ok=True,
            device=device,
            workers=2,              # 🔒 low worker count
            cache=False,            # 🔒 prevent disk explosion
            amp=True,
            plots=False,
            save=True,
            verbose=False,
        )

        best_pt = tmp_dir / "weights" / "best.pt"
        if not best_pt.exists():
            raise RuntimeError("❌ best.pt not found — training failed")

        shutil.move(str(best_pt), str(final_model_path))
        logger.info(f"✔ Final model saved → {final_model_path}")

        # -------------------------------------------------------
        # CLEAN YOLO TEMP FILES (CRITICAL)
        # -------------------------------------------------------
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("🧹 Cleaned YOLO tmp_global directory")

        # -------------------------------------------------------
        # STORE LOCALLY + UPLOAD MODEL
        # -------------------------------------------------------
        _store_model_in_app_service(final_model_path)

        model_url = asyncio.run(
            _upload_file_async(final_model_path, "models", AZURE_CONFIG)
        )

        # -------------------------------------------------------
        # LOAD SPECIES SCHEMA (SOURCE OF TRUTH)
        # -------------------------------------------------------
        species_yaml_path = download_species_yaml()

        with open(species_yaml_path, "r", encoding="utf-8") as f:
            species_yaml = pyyaml.safe_load(f) or {}

        class_map = species_yaml.get("class_map", {})
        if not class_map:
            raise RuntimeError("❌ class_map missing in data.yaml")

        speciesIndexes = [
            {"index": int(i), "speciesName": class_map[str(i)]}
            for i in sorted(map(int, class_map.keys()))
        ]

        if not speciesIndexes:
            raise RuntimeError("❌ speciesIndexes EMPTY — aborting publish")

        logger.info(f"🐟 Species indexes: {speciesIndexes}")

        # -------------------------------------------------------
        # BUILD API PAYLOAD
        # -------------------------------------------------------
        species_to_projects = json.loads(os.getenv("SPECIES_PROJECT_MAP", "{}"))
        if not species_to_projects:
            raise RuntimeError("❌ SPECIES_PROJECT_MAP missing")

        response_array = []
        for idx, (_, project_ids) in enumerate(species_to_projects.items()):
            for pid in project_ids:
                response_array.append({
                    "index": idx,
                    "projectId": pid,
                    "modelPath": model_url,
                    "jsonPath": "",
                    "status": "Success"
                })

        shutil.copy(GROUP_HASHES_CURR_PATH, GROUP_HASHES_PREV_PATH)

        payload = {
            "results": response_array,
            "speciesIndexes": speciesIndexes,
            "modelName": final_model_path.name,
            "message": "Model trained and uploaded successfully"
        }

        notify_training_result(payload)

    except Exception as e:
        logger.exception(f"❌ Training failed: {e}")
        raise

    finally:
        # -------------------------------------------------------
        # FINAL SAFETY CLEANUP
        # -------------------------------------------------------
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("🏁 Training complete.")




# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------
def _load_json(path: Path):
    if path.exists():
        return json.load(open(path, "r"))
    return {}


def _get_training_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def get_training_device():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()

            # Use all GPUs if more than one is available
            if gpu_count > 1:
                return ",".join(str(i) for i in range(gpu_count))

            # Single GPU
            return "0"
    except Exception:
        pass

    return "cpu"



def _store_model_in_app_service(source_model_path: Path):
    target_dir = Path("/home/site/wwwroot/app_backend/model_cache")
    target_dir.mkdir(parents=True, exist_ok=True)

    for f in target_dir.glob("*.pt"):
        f.unlink(missing_ok=True)

    shutil.copy(source_model_path, target_dir / "Model.pt")
    logger.info("📌 Model stored in App Service")


async def _upload_file_async(file_path: Path, blob_prefix: str, config: dict) -> str:
    container = ContainerClient.from_container_url(
        f"{config['ConnectionString']}?{config['SasToken']}"
    )
    blob_name = f"{blob_prefix}/{file_path.name}"
    blob_client = container.get_blob_client(blob_name)

    with open(file_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)

    return f"{config['ConnectionString']}/{blob_name}"



