

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import json
import logging

from .config import (
    INPUT_DIR, TEMP_DATA_DIR, MODELS_DIR,
    MODEL_SPECIES_MAP_PATH, GROUP_SPECIES_DATA_COUNT_PATH,
    GROUP_HASHES_PREV_PATH, GROUP_HASHES_CURR_PATH,
    GROUP_SPECIES_DATA_COUNT_PREV_PATH,
    CONF_THRESHOLD, MAX_THREADS, BASE_MODEL, EPOCHS, IMGSZ, BATCH,
    MAX_SPECIES_PER_GROUP, IMAGE_EXTENSIONS, TRAIN_DIR_NAME, VALID_DIR_NAME,
    NORMALIZE_PATTERN
)

logger = logging.getLogger(__name__)

# Cache for preloaded models
_loaded_models = {}

# -------------------------
# Utility functions
# -------------------------
def load_model_species_map():
    if MODEL_SPECIES_MAP_PATH.exists():
        try:
            with open(MODEL_SPECIES_MAP_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.exception(f"❌ Failed to load species map: {e}")
            return {}
    else:
        logger.warning(f"⚠️ No species map found at {MODEL_SPECIES_MAP_PATH}")
        return {}

def preload_models():
    """Load all YOLO models and their species mapping into memory once."""
    species_map = load_model_species_map()
    for uuid, group_data in species_map.items():
        matching_files = list(MODELS_DIR.glob(f"*{uuid}*.pt*"))
        if matching_files:
            model_path = matching_files[0]
            try:
                logger.info(f"🔹 Preloading model for group {uuid}: {model_path.name}")
                _loaded_models[uuid] = {
                    "model": YOLO(model_path),
                    "species": group_data["species"]
                }
            except Exception as e:
                logger.exception(f"❌ Failed to preload model {model_path}: {e}")
        else:
            logger.warning(f"⚠️ No model file found for UUID {uuid}")

def predict_with_loaded_model(uuid, image_path):
    """Run prediction using a preloaded YOLO model."""
    if uuid not in _loaded_models:
        logger.warning(f"⚠️ Model for {uuid} not preloaded.")
        return []
    try:
        model_info = _loaded_models[uuid]
        results = model_info["model"].predict(image_path, conf=CONF_THRESHOLD, verbose=False)
        preds = []
        for r in results:
            for b in r.boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                species = model_info["species"][cls_id] if cls_id < len(model_info["species"]) else "Unknown"
                preds.append({
                    "class_id": cls_id,
                    "species": species,
                    "confidence": round(conf, 3),
                    "model_group": uuid
                })
        return preds
    except Exception as e:
        logger.exception(f"❌ Error during prediction with model {uuid}: {e}")
        return []

# -------------------------
# Main Prediction Function
# -------------------------
def predict_best_multithread(image_path):
    """Run prediction on all models in parallel, return best detection."""
    if not _loaded_models:
        logger.warning("⚠️ No models preloaded, loading now...")
        preload_models()

    best_prediction = None

    def worker(uuid):
        return predict_with_loaded_model(uuid, image_path)

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(worker, uuid): uuid for uuid in _loaded_models}
        for future in futures:
            uuid = futures[future]
            try:
                preds = future.result()
                for pred in preds:
                    if (best_prediction is None) or (pred["confidence"] > best_prediction["confidence"]):
                        best_prediction = pred
            except Exception as e:
                logger.exception(f"❌ Error predicting with model {uuid}: {e}")

    if best_prediction:
        logger.info(f"✅ Best prediction: {best_prediction}")
    else:
        logger.warning("⚠️ No valid predictions made by any model.")

    return best_prediction or {"species": None, "confidence": 0.0, "model_group": None}


















# import os
# from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor
# from ultralytics import YOLO
# from .config import (
#     INPUT_DIR, TEMP_DATA_DIR, MODELS_DIR,
#     MODEL_SPECIES_MAP_PATH, GROUP_SPECIES_DATA_COUNT_PATH,
#     GROUP_HASHES_PREV_PATH, GROUP_HASHES_CURR_PATH,
#     GROUP_SPECIES_DATA_COUNT_PREV_PATH,
#     CONF_THRESHOLD, MAX_THREADS, BASE_MODEL, EPOCHS, IMGSZ, BATCH,
#     MAX_SPECIES_PER_GROUP, IMAGE_EXTENSIONS, TRAIN_DIR_NAME, VALID_DIR_NAME,
#     NORMALIZE_PATTERN
# )
# import json

# # Cache for preloaded models
# _loaded_models = {}

# # -------------------------
# # Utility functions
# # -------------------------
# def load_model_species_map():
#     if MODEL_SPECIES_MAP_PATH.exists():
#         with open(MODEL_SPECIES_MAP_PATH, "r", encoding="utf-8") as f:
#             return json.load(f)
#     else:
#         print(f"⚠️ No species map found at {MODEL_SPECIES_MAP_PATH}")
#         return {}

# def preload_models():
#     """Load all YOLO models and their species mapping into memory once."""
#     species_map = load_model_species_map()
#     for uuid, group_data in species_map.items():
#         matching_files = list(MODELS_DIR.glob(f"*{uuid}*.pt*"))
#         if matching_files:
#             model_path = matching_files[0]
#             print(f"🔹 Preloading model for group {uuid}: {model_path.name}")
#             _loaded_models[uuid] = {
#                 "model": YOLO(model_path),
#                 "species": group_data["species"]
#             }
#         else:
#             print(f"⚠️ No model file found for UUID {uuid}")

# def predict_with_loaded_model(uuid, image_path):
#     """Run prediction using a preloaded YOLO model."""
#     if uuid not in _loaded_models:
#         print(f"⚠️ Model for {uuid} not preloaded.")
#         return []
#     model_info = _loaded_models[uuid]
#     results = model_info["model"].predict(image_path, conf=CONF_THRESHOLD, verbose=False)
#     preds = []
#     for r in results:
#         for b in r.boxes:
#             cls_id = int(b.cls[0].item())
#             conf = float(b.conf[0].item())
#             species = model_info["species"][cls_id] if cls_id < len(model_info["species"]) else "Unknown"
#             preds.append({
#                 "class_id": cls_id,
#                 "species": species,
#                 "confidence": round(conf, 3),
#                 "model_group": uuid
#             })
#     return preds

# # -------------------------
# # Main Prediction Function
# # -------------------------
# def predict_best_multithread(image_path):
#     """Run prediction on all models in parallel, return best detection."""
#     if not _loaded_models:
#         print("⚠️ No models preloaded, loading now...")
#         preload_models()

#     best_prediction = None

#     def worker(uuid):
#         return predict_with_loaded_model(uuid, image_path)

#     with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
#         futures = {executor.submit(worker, uuid): uuid for uuid in _loaded_models}
#         for future in futures:
#             try:
#                 preds = future.result()
#                 for pred in preds:
#                     if (best_prediction is None) or (pred["confidence"] > best_prediction["confidence"]):
#                         best_prediction = pred
#             except Exception as e:
#                 print(f"❌ Error predicting with model {futures[future]}: {e}")

#     return best_prediction or {"species": None, "confidence": 0.0, "model_group": None}

# -------------------------
# For Flask Integration
# -------------------------
# if __name__ == "__main__":
#     # Preload models at script start
#     preload_models()
    # test_img = "path/to/test.jpg"
    # best = predict_best_multithread(test_img)
    # print("Best prediction:", best)













# # scripts/predict_best_model.py

# import os
# from pathlib import Path
# from ultralytics import YOLO
# import json
# from pprint import pprint
# import sys
# from concurrent.futures import ThreadPoolExecutor, as_completed

# sys.path.append(str(Path(__file__).resolve().parent.parent))
# from utils.config import MODELS_DIR, MODEL_SPECIES_MAP_PATH, CONF_THRESHOLD, MAX_THREADS

# # CONF_THRESHOLD = 0.7
# # MAX_THREADS = 4  # Adjust based on your CPU/GPU capacity

# # -------------------- Load model-species mapping --------------------
# def load_model_species_map(json_path=MODEL_SPECIES_MAP_PATH):
#     with open(json_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# # -------------------- Predict using a single model --------------------
# def predict_with_model(model_path, image_path, class_names):
#     model_path = Path(model_path)
#     if not model_path.exists():
#         print(f"❌ Model not found: {model_path}")
#         return []

#     model = YOLO(model_path)
#     results = model.predict(image_path, conf=CONF_THRESHOLD)
#     preds = []

#     for r in results:
#         boxes = r.boxes
#         for b in boxes:
#             cls_id = int(b.cls[0].item())
#             conf = float(b.conf[0].item())
#             species = class_names[cls_id] if cls_id < len(class_names) else "Unknown"
#             preds.append({
#                 "class_id": cls_id,
#                 "species": species,
#                 "confidence": round(conf, 3),
#                 "model_group": model_path.name
#             })
#     return preds

# # -------------------- Worker for multithreading --------------------
# def worker(uuid, species_list, image_path):
#     matching_files = list(MODELS_DIR.glob(f"*{uuid}*.pt*"))
#     if not matching_files:
#         print(f"❌ No model found for UUID {uuid}")
#         return []
#     model_path = matching_files[0]
#     return predict_with_model(model_path, image_path, species_list)

# # -------------------- Multi-model prediction with multithreading --------------------
# def predict_best_multithread(image_path):
#     species_map = load_model_species_map()
#     best_prediction = None

#     futures = []
#     with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
#         for uuid, group_data in species_map.items():
#             species_list = group_data["species"]
#             futures.append(executor.submit(worker, uuid, species_list, image_path))

#         for future in as_completed(futures):
#             preds = future.result()
#             for pred in preds:
#                 if (best_prediction is None) or (pred["confidence"] > best_prediction["confidence"]):
#                     best_prediction = pred

#     if best_prediction is None:
#         return {"species": None, "confidence": 0.0, "model_group": None}
#     return best_prediction

# # -------------------- CLI Testing --------------------
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Predict fish species using all trained YOLO models.")
#     parser.add_argument("--image", required=True, help="Path to input image")
#     args = parser.parse_args()

#     image_path = Path(args.image)
#     if not image_path.exists():
#         print(f"❌ Image not found: {image_path}")
#         exit(1)

#     result = predict_best_multithread(image_path)
#     print("\n🎯 Best Prediction:")
#     pprint(result)


