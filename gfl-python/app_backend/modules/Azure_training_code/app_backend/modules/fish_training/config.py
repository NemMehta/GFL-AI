

# compatibility shim — re-export training settings for existing imports
from app_backend.config import TRAIN

# Optional: re-export common names if your scripts import symbols directly
DATASET_ROOT = TRAIN.DATASET_ROOT
INPUT_DIR    = TRAIN.INPUT_DIR
TEMP_DATA_DIR = TRAIN.TEMP_DATA_DIR
MODELS_DIR    = TRAIN.MODELS_DIR
METADATA_DIR  = TRAIN.METADATA_DIR

MODEL_SPECIES_MAP_PATH             = TRAIN.MODEL_SPECIES_MAP_PATH
GROUP_HASHES_PREV_PATH             = TRAIN.GROUP_HASHES_PREV_PATH
GROUP_HASHES_CURR_PATH             = TRAIN.GROUP_HASHES_CURR_PATH
GROUP_SPECIES_DATA_COUNT_PATH      = TRAIN.GROUP_SPECIES_DATA_COUNT_PATH
GROUP_SPECIES_DATA_COUNT_PREV_PATH = TRAIN.GROUP_SPECIES_DATA_COUNT_PREV_PATH

CONF_THRESHOLD = TRAIN.CONF_THRESHOLD
MAX_THREADS    = TRAIN.MAX_THREADS

BASE_MODEL = TRAIN.BASE_MODEL
EPOCHS     = TRAIN.EPOCHS
IMGSZ      = TRAIN.IMGSZ
BATCH      = TRAIN.BATCH

MAX_SPECIES_PER_GROUP = TRAIN.MAX_SPECIES_PER_GROUP
IMAGE_EXTENSIONS      = TRAIN.IMAGE_EXTENSIONS
TRAIN_DIR_NAME        = TRAIN.TRAIN_DIR_NAME
VALID_DIR_NAME        = TRAIN.VALID_DIR_NAME
NORMALIZE_PATTERN     = TRAIN.NORMALIZE_PATTERN

CLEAN_MODELS_DIR_BEFORE_TRAIN = TRAIN.CLEAN_MODELS_DIR_BEFORE_TRAIN
















# # training/config.py
# from __future__ import annotations
# from pathlib import Path
# import os

# def _env_path(name: str, default: Path, root: Path | None = None) -> Path:
#     """
#     Resolve a path from ENV if present, else use default.
#     - Expands ~
#     - If env value is relative and root is provided, resolve relative to root.
#     - Always returns absolute Path.
#     """
#     v = os.getenv(name)
#     if not v:
#         return default.expanduser().resolve()
#     p = Path(v).expanduser()
#     if not p.is_absolute() and root is not None:
#         p = (root / p)
#     return p.resolve()

# # ---- Anchors that survive moves ----
# ANCHOR       = Path(__file__).resolve()
# MODULE_ROOT  = ANCHOR.parent                               # .../training
# APP_BACKEND  = MODULE_ROOT.parent.parent                   # .../app_backend (expected layout)

# # ---- Where training reads the dataset built by annotator ----
# # DATASET_ROOT = _env_path("DATASET_ROOT", APP_BACKEND / "dataset" / APP_BACKEND)
# DATASET_ROOT = _env_path(
#     "DATASET_ROOT",
#     APP_BACKEND / "modules" / "fish_annotation" / "dataset",
#     APP_BACKEND
# )

# INPUT_DIR    = DATASET_ROOT
# print("Input dir for training:", INPUT_DIR)
# print("Dataset root:", DATASET_ROOT)
# # ---- Writable artifacts for training ----
# TEMP_DATA_DIR = _env_path("TEMP_DATA_DIR", MODULE_ROOT / "temp_data", MODULE_ROOT)
# MODELS_DIR    = _env_path("MODELS_DIR",    MODULE_ROOT / "models",    MODULE_ROOT)
# METADATA_DIR  = _env_path("METADATA_DIR",  MODULE_ROOT / "metadata",  MODULE_ROOT)

# # Ensure dirs exist
# for p in [INPUT_DIR, TEMP_DATA_DIR, MODELS_DIR, METADATA_DIR]:
#     Path(p).mkdir(parents=True, exist_ok=True)

# # ---- Metadata files ----
# MODEL_SPECIES_MAP_PATH            = METADATA_DIR / "model_species_map.json"
# GROUP_HASHES_PREV_PATH            = METADATA_DIR / "group_hashes.json"
# GROUP_HASHES_CURR_PATH            = METADATA_DIR / "group_hashes_current.json"
# GROUP_SPECIES_DATA_COUNT_PATH     = METADATA_DIR / "group_species_data_count.json"
# GROUP_SPECIES_DATA_COUNT_PREV_PATH= METADATA_DIR / "group_species_data_count_prev.json"

# # ---- Knobs (env-overridable) ----
# CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.7"))
# MAX_THREADS    = int(os.getenv("MAX_THREADS", "4"))

# BASE_MODEL = str(_env_path("BASE_MODEL", MODULE_ROOT / "yolov8n.pt", MODULE_ROOT))
# EPOCHS     = int(os.getenv("EPOCHS", "5"))
# IMGSZ      = int(os.getenv("IMGSZ", "640"))
# BATCH      = int(os.getenv("BATCH", "16"))

# MAX_SPECIES_PER_GROUP = int(os.getenv("MAX_SPECIES_PER_GROUP", "3"))
# IMAGE_EXTENSIONS      = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
# TRAIN_DIR_NAME        = "train"
# VALID_DIR_NAME        = "valid"
# NORMALIZE_PATTERN     = r"[^a-z0-9]"



# # -------------------- Cleanup behavior --------------------
# # If True, delete all existing model_group_*.pt files before training starts.
# CLEAN_MODELS_DIR_BEFORE_TRAIN = True


