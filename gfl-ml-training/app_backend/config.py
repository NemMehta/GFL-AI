




# app_backend/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import os
import numpy as np

# ---------------------------
# Helpers
# ---------------------------
def _bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:  return True
    if s in {"0", "false", "no", "n", "off"}: return False
    return default

def _env_path(name: str, default: Path, root: Path | None = None) -> Path:
    """
    Resolve a path from ENV if present, else use default.
    - Expands ~
    - If env value is relative and root is provided, resolve relative to root.
    - Always returns absolute Path.
    """
    v = os.getenv(name)
    if not v:
        return default.expanduser().resolve()
    p = Path(v).expanduser()
    if not p.is_absolute() and root is not None:
        p = (root / p)
    return p.resolve()

def _norm_url(url: str) -> str:
    return url.rstrip("/")

# ---------------------------
# App anchors / core paths
# ---------------------------
APP_BACKEND  = Path(__file__).resolve().parent            # .../app_backend
PROJECT_ROOT = APP_BACKEND.parent                         # .../Project GFL


# Public URL root (used to build shareable links)
# BASE_PUBLIC_URL = _norm_url(os.environ.get("BASE_PUBLIC_URL", "http://127.0.0.1:5000"))
# BASE_PUBLIC_URL = _norm_url(os.environ.get("BASE_PUBLIC_URL", "http://192.168.168.137:5000"))
BASE_PUBLIC_URL = _norm_url(os.environ.get("BASE_PUBLIC_URL", "https://gfl-api.appmaister.com"))

# Detect Azure ML run environment
AZURE_ML_RUN_DIR = Path(os.getenv("AZUREML_CR_WORKING_DIR", "/mnt/azureml/temp")).resolve()
WRITABLE_DIR = AZURE_ML_RUN_DIR / "outputs"
WRITABLE_DIR.mkdir(parents=True, exist_ok=True)

# Instead of writing inside app_backend (read-only)
STATIC_DIR    = _env_path("STATIC_DIR", WRITABLE_DIR / "static", APP_BACKEND)
PREDICTED_DIR = _env_path("PREDICTED_DIR", WRITABLE_DIR / "predicted", APP_BACKEND)
SAVED_DIR     = _env_path("SAVED_DIR", WRITABLE_DIR / "saved", APP_BACKEND)


# print("App Backend root: ", APP_BACKEND)


# # Static served by Flask
# STATIC_DIR    = _env_path("STATIC_DIR", APP_BACKEND / "static", APP_BACKEND)
# PREDICTED_DIR = _env_path("PREDICTED_DIR", STATIC_DIR / "predicted", APP_BACKEND)
# SAVED_DIR     = _env_path("SAVED_DIR",     STATIC_DIR / "saved",     APP_BACKEND)


# print("App Backend root: ", APP_BACKEND)

# Models Paths for Predictions
# # Using your existing _env_path helper
# LOCAL_PREDICTION_MODELS_DIR = str(_env_path("LOCAL_MODELS_DIR", APP_BACKEND.parent / "models_cache", APP_BACKEND.parent))

LOCAL_PREDICTION_MODELS_DIR = Path(
    _env_path("LOCAL_MODELS_DIR", APP_BACKEND / "modules" / "fish_training" / "models", APP_BACKEND.parent)
)




# Azure ML writable path (safe for outputs)
AZURE_WRITABLE_DIR = Path(os.getenv("AZUREML_CR_WORKING_DIR", "/mnt/azureml/outputs")).resolve()
AZURE_WRITABLE_DIR.mkdir(parents=True, exist_ok=True)

DB_FOLDER = str(_env_path("DB_FOLDER", AZURE_WRITABLE_DIR / "DB_images", APP_BACKEND.parent))
DB_FOLDER_Unknown_Fish = os.path.join(DB_FOLDER, "Unknown_fishes")

os.makedirs(DB_FOLDER_Unknown_Fish, exist_ok=True)


PREDICTED_OUTPUT_DIR = str(_env_path("OUTPUT_DIR", AZURE_WRITABLE_DIR / "predictions", APP_BACKEND.parent))




# PREDICTED_OUTPUT_DIR = str(_env_path("OUTPUT_DIR", APP_BACKEND.parent / "Predictions", APP_BACKEND.parent))




# Ensure on import
for p in [STATIC_DIR, PREDICTED_DIR, SAVED_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Databases (main + data-collection)
# ---------------------------
# DB_PATH                 = _env_path("DB_PATH",                 APP_BACKEND.parent / "fish_records.db", APP_BACKEND.parent)
# DB_PATH_DATA_COLLECTION = _env_path("DB_PATH_DATA_COLLECTION", APP_BACKEND.parent / "fish_data.db",    APP_BACKEND.parent)

DB_PATH = _env_path("DB_PATH", AZURE_WRITABLE_DIR / "fish_records.db", APP_BACKEND.parent)
DB_PATH_DATA_COLLECTION = _env_path("DB_PATH_DATA_COLLECTION", AZURE_WRITABLE_DIR / "fish_data.db", APP_BACKEND.parent)




# ---------------------------
# Main pipeline settings (kept as top-level names for existing imports)
# ---------------------------
# YOLO model for detect.py (single-model flow)
# MODEL_PATH      = str(_env_path("MODEL_PATH", APP_BACKEND.parent / "Models" / "fish" / "model_group_09e7f81f-2e0b-45be-8f05-b08d6690f1ab_2025-10-21_22-56-52.pt", APP_BACKEND.parent))
MODEL_PATH      = str(_env_path("MODEL_PATH", APP_BACKEND / "modules" / "fish_annotation" / "models" / "fish.pt", APP_BACKEND))

# MODEL_PATH      = str(_env_path("MODEL_PATH", APP_BACKEND.parent / "Models" / "fish" / "fishbest_v3.pt", APP_BACKEND.parent))
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.88"))

# Output directory for annotated predictions
# OUTPUT_DIR = str(_env_path("OUTPUT_DIR", APP_BACKEND.parent / "Outputs" / "Fishes v3" / "Prediction", APP_BACKEND.parent))
# REFERENCE_OBJECT_WIDTH_INCH = float(os.getenv("REFERENCE_OBJECT_WIDTH_INCH", "1.0"))

OUTPUT_DIR = str(_env_path("OUTPUT_DIR", AZURE_WRITABLE_DIR / "outputs" / "fish_predictions", APP_BACKEND.parent))
REFERENCE_OBJECT_WIDTH_INCH = float(os.getenv("REFERENCE_OBJECT_WIDTH_INCH", "1.0"))


# HSV for neon marker
LOWER_COLOR = np.array([int(x) for x in os.getenv("LOWER_COLOR", "50,200,200").split(",")], dtype=np.uint8)
UPPER_COLOR = np.array([int(x) for x in os.getenv("UPPER_COLOR", "70,255,255").split(",")], dtype=np.uint8)

# Uniqueness model (keras)
# Uniqueness_MODEL_PATH = str(_env_path("UNIQUENESS_MODEL_PATH", APP_BACKEND.parent / "Models" / "Uniqueness" / "fish_embedding_model.keras", APP_BACKEND.parent))
Uniqueness_MODEL_PATH = str(_env_path("UNIQUENESS_MODEL_PATH", APP_BACKEND.parent / "Models" / "Uniqueness" / "fish_embedding_model.keras", APP_BACKEND.parent))

# DB_FOLDER             = str(_env_path("DB_FOLDER",             APP_BACKEND.parent / "DB images", APP_BACKEND.parent))
# DB_FOLDER_Unknown_Fish= str(_env_path("DB_FOLDER_Unknown_Fish",             APP_BACKEND.parent / "DB images" / "Unknown_fishes", APP_BACKEND.parent))
# DB_FOLDER_Unknown_Fish = os.path.join(DB_FOLDER, "Unknown_fishes")
# os.makedirs(DB_FOLDER_Unknown_Fish, exist_ok=True)

IMG_SIZE              = (int(os.getenv("IMG_W", "105")), int(os.getenv("IMG_H", "105")))
THRESHOLD             = float(os.getenv("UNIQUENESS_THRESHOLD", "0.88"))


# -------------------- Azure Config --------------------
AZURE_CONFIG = {
	# # App maisters
    # "ConnectionString": "https://gflstorageaccount.blob.core.windows.net/dotnetbackend-container",
    # "Container": "dotnetbackend-container",
    # # "SasUrl": "https://gflstorageaccount.blob.core.windows.net/dotnetbackend-container?<sas-token>",
    # "SasToken": "sp=rcwd&st=2025-09-08T09:08:53Z&se=2026-09-08T17:23:53Z&spr=https&sv=2024-11-04&sr=c&sig=%2FPkcD%2FANv4T6EtPEAjrNw43HQ9Q6nTLM8%2BCkhxq6kw8%3D"

# client
    "ConnectionString": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container",
    "Container": "dotnetbackend-container",
    # "SasUrl": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container?sp=r&st=2025-10-14T12:00:00Z&se=2026-10-14T20:15:00Z&spr=https&sv=2024-11-04&sr=c&sig=6haTCzklALCl4RSQtVPC5f2F9RZvx6CVA6KtMFWkmUE%3D%22,
    # "SasToken": "sp=rcwl&st=2025-10-15T15:17:31Z&se=2026-10-15T23:32:31Z&spr=https&sv=2024-11-04&sr=c&sig=Cu7aKTRa3s7Thflh3pr1mGm%2BG8TflhSrdYHD%2FXgxJ80%3D"
    "SasToken": "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2026-01-31T16:03:23Z&st=2025-12-03T07:48:23Z&spr=https&sig=XeBVuCefwPKVMc2loUsAvi8bXU1klfEi89jmT1fZBjI%3D"

}
 

AZURE_SUBSCRIPTION_ID = "417ecdbc-9b50-421a-82c2-c45f29686df1"
AZURE_RESOURCE_GROUP = "GFL-App-Rg"
AZURE_ML_WORKSPACE = "GFL-MLAI"



# ---------------------------
# Annotator (merged from modules/fish_annotation/src/config.py)
# ---------------------------
@dataclass(frozen=True)
class AnnotatorSettings:
    # Paths
    PROJECT_ROOT: Path
    MODELS_DIR: Path
    YOLO_MODEL_PATH: Path
    DB_PATH: Path

    DATASET_ROOT: Path               # internal dataset root (builder writes here)
    STATIC_DIR: Path
    STATIC_YOLO_DATASET: Path        # served dataset (Flask static)
    STATIC_FISH_DATASET: Path
    ANNOTATED_ROOT: Path
    # BY_SPECIES_ROOT: Path

    # DB tables / columns
    TABLE: str
    IMAGE_COL: str
    SPECIES_COL: str
    HANDHELD_COL: str
    NAME_ID_COL: str | None

    SPECIES_MAP_TABLE: str
    SPECIES_MAP_ID: str
    SPECIES_MAP_NAME: str

    # Pipeline knobs
    VAL_FRACTION: float
    AUGMENTATIONS: int
    CLEAN: bool
    DOWNLOAD_TIMEOUT: int

    # App / server (for URL building only)
    APP_HOST: str
    APP_PORT: int
    DEBUG: bool

    # Public bases
    BASE_URL: str                # root
    STATIC_YOLO_BASE_URL: str    # <BASE_PUBLIC_URL>/static/yolo_dataset

    # Logging
    LOG_LEVEL: str

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

def _load_annotator_cfg() -> AnnotatorSettings:
    # Root of fish_annotation module
    # anno_root = _env_path("FISH_ANNOTATOR_ROOT", APP_BACKEND / "modules" / "fish_annotation", APP_BACKEND)
    anno_root = _env_path("FISH_ANNOTATOR_ROOT", AZURE_WRITABLE_DIR / "fish_annotation", AZURE_WRITABLE_DIR)

    models_dir      = _env_path("ANNOTATOR_MODELS_DIR",      anno_root / "models", anno_root)
    yolo_model_path = _env_path("YOLO_MODEL_PATH",           models_dir / "fish.pt", anno_root)
    db_path         = _env_path("ANNOTATOR_DB_PATH",         AZURE_WRITABLE_DIR / "fish_data.db", AZURE_WRITABLE_DIR)


    # models_dir      = _env_path("ANNOTATOR_MODELS_DIR",      anno_root / "models", anno_root)
    # yolo_model_path = _env_path("YOLO_MODEL_PATH",           models_dir / "fish.pt", anno_root)
    # Prefer centralized data-collection DB unless overridden
    # db_path         = _env_path("ANNOTATOR_DB_PATH",         DB_PATH_DATA_COLLECTION, APP_BACKEND)


    # You requested this default explicitly:
    # dataset_root        = _env_path("DATASET_ROOT",        anno_root / "dataset",        anno_root)
    # static_dir          = STATIC_DIR  # share app static
    # static_yolo_dataset = _env_path("STATIC_YOLO_DATASET", static_dir / "yolo_dataset",  APP_BACKEND)
    # static_fish_dataset = _env_path("STATIC_FISH_DATASET", static_dir / "fish_dataset",  APP_BACKEND)
    # annotated_root      = _env_path("ANNOTATED_ROOT",      anno_root / "annotated_data", anno_root)
    # Dataset and outputs redirected to writable storage
    dataset_root        = _env_path("DATASET_ROOT",        AZURE_WRITABLE_DIR / "dataset",        anno_root)
    static_dir          = STATIC_DIR  # shared static
    static_yolo_dataset = _env_path("STATIC_YOLO_DATASET", AZURE_WRITABLE_DIR / "yolo_dataset",  anno_root)
    static_fish_dataset = _env_path("STATIC_FISH_DATASET", AZURE_WRITABLE_DIR / "fish_dataset",  anno_root)
    annotated_root      = _env_path("ANNOTATED_ROOT",      AZURE_WRITABLE_DIR / "annotated_data", anno_root)


	# by_species_root     = _env_path("BY_SPECIES_ROOT",     anno_root / "by_species",     anno_root)

    # print("Using annotator dataset root: ", dataset_root)
    # print("Base public URL: ", BASE_PUBLIC_URL)
    # print("Using static dir: ", static_dir)
    # print("Using static yolo dataset: ", static_yolo_dataset)
    # print("Using static fish dataset: ", static_fish_dataset)

    # Tables/cols
    table        = os.getenv("IMAGES_TABLE", "images")
    image_col    = os.getenv("IMAGE_COL", "image_path")
    species_col  = os.getenv("SPECIES_COL", "species_id")
    handheld_col = os.getenv("HANDHELD_COL", "handheld")
    name_id_col  = os.getenv("NAME_ID_COL") or None

    species_map_table = os.getenv("SPECIES_MAP_TABLE", "species")
    species_map_id    = os.getenv("SPECIES_MAP_ID", "id")
    species_map_name  = os.getenv("SPECIES_MAP_NAME", "species_name")

    # Knobs
    val_fraction     = float(os.getenv("VAL_FRACTION", "0.2"))
    augmentations    = int(os.getenv("AUGMENTATIONS", "2"))
    clean            = _bool(os.getenv("CLEAN"), False)
    download_timeout = int(os.getenv("DOWNLOAD_TIMEOUT", "10"))

    # App host/port control just for building URLs; BASE_PUBLIC_URL is the source of truth
    app_host = os.getenv("APP_HOST", "192.168.168.137")
    app_port = int(os.getenv("APP_PORT", "5000"))
    debug    = _bool(os.getenv("DEBUG"), True)

    base_url_root        = _norm_url(os.getenv("BASE_URL", BASE_PUBLIC_URL))
    static_yolo_base_url = f"{base_url_root}/static/yolo_dataset"
    # print("Using static yolo base URL: ", static_yolo_base_url)

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    cfg = AnnotatorSettings(
        PROJECT_ROOT=anno_root,
        MODELS_DIR=models_dir,
        YOLO_MODEL_PATH=yolo_model_path,
        DB_PATH=db_path,

        DATASET_ROOT=dataset_root,
        STATIC_DIR=static_dir,
        STATIC_YOLO_DATASET=static_yolo_dataset,
        STATIC_FISH_DATASET=static_fish_dataset,
        ANNOTATED_ROOT=annotated_root,
        # BY_SPECIES_ROOT=by_species_root,

        TABLE=table,
        IMAGE_COL=image_col,
        SPECIES_COL=species_col,
        HANDHELD_COL=handheld_col,
        NAME_ID_COL=name_id_col,

        SPECIES_MAP_TABLE=species_map_table,
        SPECIES_MAP_ID=species_map_id,
        SPECIES_MAP_NAME=species_map_name,

        VAL_FRACTION=val_fraction,
        AUGMENTATIONS=augmentations,
        CLEAN=clean,
        DOWNLOAD_TIMEOUT=download_timeout,

        APP_HOST=app_host,
        APP_PORT=app_port,
        DEBUG=debug,

        BASE_URL=base_url_root,
        STATIC_YOLO_BASE_URL=static_yolo_base_url,

        LOG_LEVEL=log_level,
    )

    # # Ensure required dirs exist
    # for p in [
    #     cfg.MODELS_DIR, cfg.DATASET_ROOT, cfg.STATIC_DIR, cfg.STATIC_YOLO_DATASET,
    #     cfg.STATIC_FISH_DATASET, cfg.ANNOTATED_ROOT, cfg.BY_SPECIES_ROOT
    # ]:

    # Ensure required dirs exist
    for p in [
        cfg.MODELS_DIR, cfg.DATASET_ROOT, cfg.STATIC_DIR, cfg.STATIC_YOLO_DATASET,
        cfg.STATIC_FISH_DATASET, cfg.ANNOTATED_ROOT
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)

    return cfg

ANNO = _load_annotator_cfg()

# --- Compatibility aliases for legacy imports ---
DATASET_ROOT          = ANNO.DATASET_ROOT
STATIC_YOLO_DATASET   = ANNO.STATIC_YOLO_DATASET
STATIC_FISH_DATASET   = ANNO.STATIC_FISH_DATASET
STATIC_YOLO_BASE_URL  = ANNO.STATIC_YOLO_BASE_URL



# Database path (centralized here instead of hardcoding)
DB_PATH = _env_path(
    "DB_PATH",
    PROJECT_ROOT / "fish_records.db",
    PROJECT_ROOT
)



# ---------------------------
# Training (merged from modules/fish_training/config.py)
# ---------------------------
@dataclass(frozen=True)
class TrainingSettings:
    MODULE_ROOT: Path
    APP_BACKEND: Path

    DATASET_ROOT: Path   # where training reads from (defaults to ANNO.DATASET_ROOT)
    INPUT_DIR: Path

    MODEL_DB_PATH = DB_PATH

    TEMP_DATA_DIR: Path
    MODELS_DIR: Path
    METADATA_DIR: Path

    MODEL_SPECIES_MAP_PATH: Path
    GROUP_HASHES_PREV_PATH: Path
    GROUP_HASHES_CURR_PATH: Path
    GROUP_SPECIES_DATA_COUNT_PATH: Path
    GROUP_SPECIES_DATA_COUNT_PREV_PATH: Path

    CONF_THRESHOLD: float
    MAX_THREADS: int

    BASE_MODEL: str
    EPOCHS: int
    IMGSZ: int
    BATCH: int

    MAX_SPECIES_PER_GROUP: int
    IMAGE_EXTENSIONS: set[str]
    TRAIN_DIR_NAME: str
    VALID_DIR_NAME: str
    NORMALIZE_PATTERN: str

    CLEAN_MODELS_DIR_BEFORE_TRAIN: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            if isinstance(v, set):
                d[k] = sorted(v)
        return d

def _load_training_cfg() -> TrainingSettings:
    # module_root = _env_path("FISH_TRAINING_ROOT", APP_BACKEND / "modules" / "fish_training", APP_BACKEND)

    # # Default training dataset = annotator dataset
    # dataset_root = _env_path(
    #     "DATASET_ROOT",
    #     ANNO.DATASET_ROOT / "augmentation" / "training_data",   # ✅ new default
    #     APP_BACKEND
    # )
    

    # # dataset_root = _env_path("DATASET_ROOT", ANNO.DATASET_ROOT, APP_BACKEND)
    # input_dir    = dataset_root

    # temp_dir   = _env_path("TEMP_DATA_DIR", module_root / "temp_data", module_root)
    # models_dir = _env_path("MODELS_DIR",    module_root / "models",    module_root)
    # meta_dir   = _env_path("METADATA_DIR",  module_root / "metadata",  module_root)

    # Azure ML writable path
    AZURE_WRITABLE_DIR = Path(os.getenv("AZUREML_CR_WORKING_DIR", "/mnt/azureml/outputs")).resolve()
    AZURE_WRITABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Training module paths redirected to writable directory
    module_root = _env_path("FISH_TRAINING_ROOT", APP_BACKEND / "modules" / "fish_training", APP_BACKEND)

    # dataset_root can stay, or also be moved if you write to it
    dataset_root = _env_path("DATASET_ROOT", AZURE_WRITABLE_DIR / "dataset" / "training_data", AZURE_WRITABLE_DIR)
    input_dir    = dataset_root

    temp_dir   = _env_path("TEMP_DATA_DIR", AZURE_WRITABLE_DIR / "temp_data", AZURE_WRITABLE_DIR)
    models_dir = _env_path("MODELS_DIR",    AZURE_WRITABLE_DIR / "models",    AZURE_WRITABLE_DIR)
    meta_dir   = _env_path("METADATA_DIR",  AZURE_WRITABLE_DIR / "metadata",  AZURE_WRITABLE_DIR)



    # Ensure dirs
    for p in [input_dir, temp_dir, models_dir, meta_dir]:
        Path(p).mkdir(parents=True, exist_ok=True)

    model_species_map            = meta_dir / "model_species_map.json"
    group_hashes_prev            = meta_dir / "group_hashes.json"
    group_hashes_curr            = meta_dir / "group_hashes_current.json"
    species_counts_curr          = meta_dir / "group_species_data_count.json"
    species_counts_prev          = meta_dir / "group_species_data_count_prev.json"

    conf_thresh = float(os.getenv("CONF_THRESHOLD", "0.7"))
    max_threads = int(os.getenv("MAX_THREADS", "4"))

    base_model = str(_env_path("BASE_MODEL", module_root / "yolov8n.pt", module_root))
    # base_model = str(_env_path("BASE_MODEL", module_root / "yolov8n.yaml", module_root))
    epochs     = int(os.getenv("EPOCHS", "100"))
    imgsz      = int(os.getenv("IMGSZ", "640"))
    # batch      = int(os.getenv("BATCH", "16"))
    batch_env = os.getenv("BATCH", "0.85")

    try:
        batch = float(batch_env)
        if batch <= 0:
            raise ValueError("BATCH must be > 0")
        if batch > 1:
            if not batch.is_integer():
                raise ValueError("BATCH > 1 must be an integer")
            batch = int(batch)
    except Exception:
        raise ValueError(f"Invalid BATCH value: {batch_env}")

    max_species_per_group = int(os.getenv("MAX_SPECIES_PER_GROUP", "300"))

    # clean_models_before = _bool(os.getenv("CLEAN_MODELS_DIR_BEFORE_TRAIN"), True)
    clean_models_before = _bool(os.getenv("CLEAN_MODELS_DIR_BEFORE_TRAIN"), False)


    cfg = TrainingSettings(
        MODULE_ROOT=module_root,
        APP_BACKEND=APP_BACKEND,

        DATASET_ROOT=dataset_root,
        INPUT_DIR=input_dir,

        TEMP_DATA_DIR=temp_dir,
        MODELS_DIR=models_dir,
        METADATA_DIR=meta_dir,

        MODEL_SPECIES_MAP_PATH=model_species_map,
        GROUP_HASHES_PREV_PATH=group_hashes_prev,
        GROUP_HASHES_CURR_PATH=group_hashes_curr,
        GROUP_SPECIES_DATA_COUNT_PATH=species_counts_curr,
        GROUP_SPECIES_DATA_COUNT_PREV_PATH=species_counts_prev,

        CONF_THRESHOLD=conf_thresh,
        MAX_THREADS=max_threads,

        BASE_MODEL=base_model,
        EPOCHS=epochs,
        IMGSZ=imgsz,
        BATCH=batch,

        MAX_SPECIES_PER_GROUP=max_species_per_group,
        IMAGE_EXTENSIONS={".jpg", ".jpeg", ".png", ".bmp", ".gif"},
        TRAIN_DIR_NAME="train",
        VALID_DIR_NAME="valid",
        NORMALIZE_PATTERN=r"[^a-z0-9]",

        CLEAN_MODELS_DIR_BEFORE_TRAIN=clean_models_before,
    )

    return cfg

TRAIN = _load_training_cfg()

# TRAIN_DATASET_ROOT  = TRAIN.DATASET_ROOT
# TRAIN_INPUT_DIR     = TRAIN.INPUT_DIR
# TRAIN_TEMP_DATA_DIR = TRAIN.TEMP_DATA_DIR
# TRAIN_MODELS_DIR    = TRAIN.MODELS_DIR
# TRAIN_METADATA_DIR  = TRAIN.METADATA_DIR



# ---------------------------
# Logging helper
# ---------------------------
def configure_logging() -> None:
    import logging
    # Use annotator log level as the global default
    level = getattr(logging, ANNO.LOG_LEVEL, logging.INFO)
    logging.basicConfig(level=level)























# # azure code
# # app_backend/config.py
# from __future__ import annotations
# from dataclasses import dataclass, asdict
# from pathlib import Path
# import os
# import numpy as np

# # ---------------------------
# # Helpers
# # ---------------------------
# def _bool(val: str | None, default: bool = False) -> bool:
#     if val is None:
#         return default
#     s = str(val).strip().lower()
#     if s in {"1", "true", "yes", "y", "on"}:  return True
#     if s in {"0", "false", "no", "n", "off"}: return False
#     return default

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

# def _norm_url(url: str) -> str:
#     return url.rstrip("/")

# # ---------------------------
# # App anchors / core paths
# # ---------------------------
# APP_BACKEND  = Path(__file__).resolve().parent            # .../app_backend
# PROJECT_ROOT = APP_BACKEND.parent                         # .../Project GFL

# # Public URL root (used to build shareable links)
# # BASE_PUBLIC_URL = _norm_url(os.environ.get("BASE_PUBLIC_URL", "http://127.0.0.1:5000"))
# # BASE_PUBLIC_URL = _norm_url(os.environ.get("BASE_PUBLIC_URL", "http://192.168.168.137:5000"))
# # BASE_PUBLIC_URL = _norm_url(os.environ.get("BASE_PUBLIC_URL", "https://gfl-api.appmaister.com"))
# BASE_PUBLIC_URL = _norm_url(os.environ.get("BASE_PUBLIC_URL", "https://gfl-backendapi.azurewebsites.net/"))


# # Detect Azure ML run environment
# AZURE_ML_RUN_DIR = Path(os.getenv("AZUREML_CR_WORKING_DIR", "/mnt/azureml/temp")).resolve()
# WRITABLE_DIR = AZURE_ML_RUN_DIR / "outputs"
# WRITABLE_DIR.mkdir(parents=True, exist_ok=True)

# # Instead of writing inside app_backend (read-only)
# STATIC_DIR    = _env_path("STATIC_DIR", WRITABLE_DIR / "static", APP_BACKEND)
# PREDICTED_DIR = _env_path("PREDICTED_DIR", WRITABLE_DIR / "predicted", APP_BACKEND)
# SAVED_DIR     = _env_path("SAVED_DIR", WRITABLE_DIR / "saved", APP_BACKEND)


# # # Static served by Flask
# # STATIC_DIR    = _env_path("STATIC_DIR", APP_BACKEND / "static", APP_BACKEND)
# # PREDICTED_DIR = _env_path("PREDICTED_DIR", STATIC_DIR / "predicted", APP_BACKEND)
# # SAVED_DIR     = _env_path("SAVED_DIR",     STATIC_DIR / "saved",     APP_BACKEND)


# print("App Backend root: ", APP_BACKEND)

# # Models Paths for Predictions
# # # Using your existing _env_path helper
# # LOCAL_PREDICTION_MODELS_DIR = str(_env_path("LOCAL_MODELS_DIR", APP_BACKEND.parent / "models_cache", APP_BACKEND.parent))

# LOCAL_PREDICTION_MODELS_DIR = Path(
#     _env_path("LOCAL_MODELS_DIR", APP_BACKEND / "modules" / "fish_training" / "models", APP_BACKEND.parent)
# )



# # Azure ML writable path (safe for outputs)
# AZURE_WRITABLE_DIR = Path(os.getenv("AZUREML_CR_WORKING_DIR", "/mnt/azureml/outputs")).resolve()
# AZURE_WRITABLE_DIR.mkdir(parents=True, exist_ok=True)

# DB_FOLDER = str(_env_path("DB_FOLDER", AZURE_WRITABLE_DIR / "DB_images", APP_BACKEND.parent))
# DB_FOLDER_Unknown_Fish = os.path.join(DB_FOLDER, "Unknown_fishes")

# os.makedirs(DB_FOLDER_Unknown_Fish, exist_ok=True)


# PREDICTED_OUTPUT_DIR = str(_env_path("OUTPUT_DIR", AZURE_WRITABLE_DIR / "predictions", APP_BACKEND.parent))


# # PREDICTED_OUTPUT_DIR = str(_env_path("OUTPUT_DIR", APP_BACKEND.parent / "Predictions", APP_BACKEND.parent))




# # Ensure on import
# for p in [STATIC_DIR, PREDICTED_DIR, SAVED_DIR]:
#     p.mkdir(parents=True, exist_ok=True)

# # ---------------------------
# # Databases (main + data-collection)
# # ---------------------------
# # DB_PATH                 = _env_path("DB_PATH",                 APP_BACKEND.parent / "fish_records.db", APP_BACKEND.parent)
# # DB_PATH_DATA_COLLECTION = _env_path("DB_PATH_DATA_COLLECTION", APP_BACKEND.parent / "fish_data.db",    APP_BACKEND.parent)
# DB_PATH = _env_path("DB_PATH", AZURE_WRITABLE_DIR / "fish_records.db", APP_BACKEND.parent)
# DB_PATH_DATA_COLLECTION = _env_path("DB_PATH_DATA_COLLECTION", AZURE_WRITABLE_DIR / "fish_data.db", APP_BACKEND.parent)

# # ---------------------------
# # Main pipeline settings (kept as top-level names for existing imports)
# # ---------------------------
# # YOLO model for detect.py (single-model flow)
# # MODEL_PATH      = str(_env_path("MODEL_PATH", APP_BACKEND.parent / "Models" / "fish" / "fishbest_v3.pt", APP_BACKEND.parent))
# MODEL_PATH = str(_env_path("MODEL_PATH", AZURE_WRITABLE_DIR / "models" / "fishbest_v3.pt", APP_BACKEND.parent))

# YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.88"))

# # Output directory for annotated predictions
# # OUTPUT_DIR = str(_env_path("OUTPUT_DIR", APP_BACKEND.parent / "Outputs" / "Fishes v3" / "Prediction", APP_BACKEND.parent))
# OUTPUT_DIR = str(_env_path("OUTPUT_DIR", AZURE_WRITABLE_DIR / "outputs" / "fish_predictions", APP_BACKEND.parent))


# REFERENCE_OBJECT_WIDTH_INCH = float(os.getenv("REFERENCE_OBJECT_WIDTH_INCH", "1.0"))

# # HSV for neon marker
# LOWER_COLOR = np.array([int(x) for x in os.getenv("LOWER_COLOR", "50,200,200").split(",")], dtype=np.uint8)
# UPPER_COLOR = np.array([int(x) for x in os.getenv("UPPER_COLOR", "70,255,255").split(",")], dtype=np.uint8)

# # Uniqueness model (keras)
# Uniqueness_MODEL_PATH = str(_env_path("UNIQUENESS_MODEL_PATH", APP_BACKEND.parent / "Models" / "Uniqueness" / "fish_embedding_model.keras", APP_BACKEND.parent))



# # DB_FOLDER             = str(_env_path("DB_FOLDER",             APP_BACKEND.parent / "DB images", APP_BACKEND.parent))
# # # DB_FOLDER_Unknown_Fish= str(_env_path("DB_FOLDER_Unknown_Fish",             APP_BACKEND.parent / "DB images" / "Unknown_fishes", APP_BACKEND.parent))
# # DB_FOLDER_Unknown_Fish = os.path.join(DB_FOLDER, "Unknown_fishes")
# # os.makedirs(DB_FOLDER_Unknown_Fish, exist_ok=True)

# IMG_SIZE              = (int(os.getenv("IMG_W", "105")), int(os.getenv("IMG_H", "105")))
# THRESHOLD             = float(os.getenv("UNIQUENESS_THRESHOLD", "0.8"))


# # -------------------- Azure Config --------------------
# AZURE_CONFIG = {
#     "ConnectionString": "https://gflstorageaccount.blob.core.windows.net/dotnetbackend-container",
#     "Container": "dotnetbackend-container",
#     # "SasToken": "sp=rcwd&st=2025-09-08T09:08:53Z&se=2026-09-08T17:23:53Z&spr=https&sv=2024-11-04&sr=c&sig=%2FPkcD%2FANv4T6EtPEAjrNw43HQ9Q6nTLM8%2BCkhxq6kw8%3D"
#     "SasToken": "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-12-31T15:38:34Z&st=2025-09-17T07:23:34Z&spr=https&sig=IOG7m4n1ZfNb8SnCOBBURLvSPYKR0qiGjToBH8AmUzs%3D"


#     # "ConnectionString": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container",
#     # "Container": "dotnetbackend-container",
#     # # "SasUrl": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container?sp=r&st=2025-10-14T12:00:0…,
#     # # "SasToken": "sp=rw&st=2025-10-14T12:54:59Z&se=2025-10-14T21:09:59Z&spr=https&sv=2024-11-04&sr=c&sig=w5T3PVKngvvoGUWQq7keShQ90Hd5yXa8hoJK0KvwH1A%3D"
#     # "SasToken": "sp=rcwl&st=2025-10-15T15:17:31Z&se=2026-10-15T23:32:31Z&spr=https&sv=2024-11-04&sr=c&sig=Cu7aKTRa3s7Thflh3pr1mGm%2BG8TflhSrdYHD%2FXgxJ80%3D"

# }

# AZURE_SUBSCRIPTION_ID = "417ecdbc-9b50-421a-82c2-c45f29686df1"
# AZURE_RESOURCE_GROUP = "GFL-App-Rg"
# AZURE_ML_WORKSPACE = "GFL-MLAI"



# # ---------------------------
# # Annotator (merged from modules/fish_annotation/src/config.py)
# # ---------------------------
# @dataclass(frozen=True)
# class AnnotatorSettings:
#     # Paths
#     PROJECT_ROOT: Path
#     MODELS_DIR: Path
#     YOLO_MODEL_PATH: Path
#     DB_PATH: Path

#     DATASET_ROOT: Path               # internal dataset root (builder writes here)
#     STATIC_DIR: Path
#     STATIC_YOLO_DATASET: Path        # served dataset (Flask static)
#     STATIC_FISH_DATASET: Path
#     ANNOTATED_ROOT: Path
#     # BY_SPECIES_ROOT: Path

#     # DB tables / columns
#     TABLE: str
#     IMAGE_COL: str
#     SPECIES_COL: str
#     HANDHELD_COL: str
#     NAME_ID_COL: str | None

#     SPECIES_MAP_TABLE: str
#     SPECIES_MAP_ID: str
#     SPECIES_MAP_NAME: str

#     # Pipeline knobs
#     VAL_FRACTION: float
#     AUGMENTATIONS: int
#     CLEAN: bool
#     DOWNLOAD_TIMEOUT: int

#     # App / server (for URL building only)
#     APP_HOST: str
#     APP_PORT: int
#     DEBUG: bool

#     # Public bases
#     BASE_URL: str                # root
#     STATIC_YOLO_BASE_URL: str    # <BASE_PUBLIC_URL>/static/yolo_dataset

#     # Logging
#     LOG_LEVEL: str

#     def to_dict(self) -> dict:
#         d = asdict(self)
#         for k, v in d.items():
#             if isinstance(v, Path):
#                 d[k] = str(v)
#         return d

# def _load_annotator_cfg() -> AnnotatorSettings:
#     # Root of fish_annotation module
#     # anno_root = _env_path("FISH_ANNOTATOR_ROOT", APP_BACKEND / "modules" / "fish_annotation", APP_BACKEND)
#     # Redirect annotation root to writable dir
#     anno_root = _env_path("FISH_ANNOTATOR_ROOT", AZURE_WRITABLE_DIR / "fish_annotation", AZURE_WRITABLE_DIR)

#     models_dir      = _env_path("ANNOTATOR_MODELS_DIR",      anno_root / "models", anno_root)
#     yolo_model_path = _env_path("YOLO_MODEL_PATH",           models_dir / "fish.pt", anno_root)
#     db_path         = _env_path("ANNOTATOR_DB_PATH",         AZURE_WRITABLE_DIR / "fish_data.db", AZURE_WRITABLE_DIR)

#     # Dataset and outputs redirected to writable storage
#     dataset_root        = _env_path("DATASET_ROOT",        AZURE_WRITABLE_DIR / "dataset",        anno_root)
#     static_dir          = STATIC_DIR  # shared static
#     static_yolo_dataset = _env_path("STATIC_YOLO_DATASET", AZURE_WRITABLE_DIR / "yolo_dataset",  anno_root)
#     static_fish_dataset = _env_path("STATIC_FISH_DATASET", AZURE_WRITABLE_DIR / "fish_dataset",  anno_root)
#     annotated_root      = _env_path("ANNOTATED_ROOT",      AZURE_WRITABLE_DIR / "annotated_data", anno_root)



#     # models_dir      = _env_path("ANNOTATOR_MODELS_DIR",      anno_root / "models", anno_root)
#     # yolo_model_path = _env_path("YOLO_MODEL_PATH",           models_dir / "fish.pt", anno_root)
#     # # Prefer centralized data-collection DB unless overridden
#     # db_path         = _env_path("ANNOTATOR_DB_PATH",         DB_PATH_DATA_COLLECTION, APP_BACKEND)


#     # # You requested this default explicitly:
#     # # dataset_root        = _env_path("DATASET_ROOT",        anno_root / "dataset",        anno_root)
#     # dataset_root = _env_path("DATASET_ROOT", AZURE_WRITABLE_DIR / "dataset", APP_BACKEND)

#     # static_dir          = STATIC_DIR  # share app static
#     # static_yolo_dataset = _env_path("STATIC_YOLO_DATASET", static_dir / "yolo_dataset",  APP_BACKEND)
#     # static_fish_dataset = _env_path("STATIC_FISH_DATASET", static_dir / "fish_dataset",  APP_BACKEND)
#     # annotated_root      = _env_path("ANNOTATED_ROOT",      anno_root / "annotated_data", anno_root)
#     # by_species_root     = _env_path("BY_SPECIES_ROOT",     anno_root / "by_species",     anno_root)

#     # print("Using annotator dataset root: ", dataset_root)
#     # print("Base public URL: ", BASE_PUBLIC_URL)
#     # print("Using static dir: ", static_dir)
#     # print("Using static yolo dataset: ", static_yolo_dataset)
#     # print("Using static fish dataset: ", static_fish_dataset)

#     # Tables/cols
#     table        = os.getenv("IMAGES_TABLE", "images")
#     image_col    = os.getenv("IMAGE_COL", "image_path")
#     species_col  = os.getenv("SPECIES_COL", "species_id")
#     handheld_col = os.getenv("HANDHELD_COL", "handheld")
#     name_id_col  = os.getenv("NAME_ID_COL") or None

#     species_map_table = os.getenv("SPECIES_MAP_TABLE", "species")
#     species_map_id    = os.getenv("SPECIES_MAP_ID", "id")
#     species_map_name  = os.getenv("SPECIES_MAP_NAME", "species_name")

#     # Knobs
#     val_fraction     = float(os.getenv("VAL_FRACTION", "0.2"))
#     augmentations    = int(os.getenv("AUGMENTATIONS", "2"))
#     clean            = _bool(os.getenv("CLEAN"), False)
#     download_timeout = int(os.getenv("DOWNLOAD_TIMEOUT", "10"))

#     # App host/port control just for building URLs; BASE_PUBLIC_URL is the source of truth
#     app_host = os.getenv("APP_HOST", "192.168.168.137")
#     app_port = int(os.getenv("APP_PORT", "5000"))
#     debug    = _bool(os.getenv("DEBUG"), True)

#     base_url_root        = _norm_url(os.getenv("BASE_URL", BASE_PUBLIC_URL))
#     static_yolo_base_url = f"{base_url_root}/static/yolo_dataset"
#     print("Using static yolo base URL: ", static_yolo_base_url)

#     log_level = os.getenv("LOG_LEVEL", "INFO").upper()

#     cfg = AnnotatorSettings(
#         PROJECT_ROOT=anno_root,
#         MODELS_DIR=models_dir,
#         YOLO_MODEL_PATH=yolo_model_path,
#         DB_PATH=db_path,

#         DATASET_ROOT=dataset_root,
#         STATIC_DIR=static_dir,
#         STATIC_YOLO_DATASET=static_yolo_dataset,
#         STATIC_FISH_DATASET=static_fish_dataset,
#         ANNOTATED_ROOT=annotated_root,
#         # BY_SPECIES_ROOT=by_species_root,

#         TABLE=table,
#         IMAGE_COL=image_col,
#         SPECIES_COL=species_col,
#         HANDHELD_COL=handheld_col,
#         NAME_ID_COL=name_id_col,

#         SPECIES_MAP_TABLE=species_map_table,
#         SPECIES_MAP_ID=species_map_id,
#         SPECIES_MAP_NAME=species_map_name,

#         VAL_FRACTION=val_fraction,
#         AUGMENTATIONS=augmentations,
#         CLEAN=clean,
#         DOWNLOAD_TIMEOUT=download_timeout,

#         APP_HOST=app_host,
#         APP_PORT=app_port,
#         DEBUG=debug,

#         BASE_URL=base_url_root,
#         STATIC_YOLO_BASE_URL=static_yolo_base_url,

#         LOG_LEVEL=log_level,
#     )

#     # Ensure required dirs exist
#     for p in [
#         cfg.MODELS_DIR, cfg.DATASET_ROOT, cfg.STATIC_DIR, cfg.STATIC_YOLO_DATASET,
#         cfg.STATIC_FISH_DATASET, cfg.ANNOTATED_ROOT
#     ]:
#         Path(p).mkdir(parents=True, exist_ok=True)

#     return cfg

# ANNO = _load_annotator_cfg()

# # --- Compatibility aliases for legacy imports ---
# DATASET_ROOT          = ANNO.DATASET_ROOT
# STATIC_YOLO_DATASET   = ANNO.STATIC_YOLO_DATASET
# STATIC_FISH_DATASET   = ANNO.STATIC_FISH_DATASET
# STATIC_YOLO_BASE_URL  = ANNO.STATIC_YOLO_BASE_URL



# # Database path (centralized here instead of hardcoding)
# DB_PATH = _env_path(
#     "DB_PATH",
#     PROJECT_ROOT / "fish_records.db",
#     PROJECT_ROOT
# )



# # ---------------------------
# # Training (merged from modules/fish_training/config.py)
# # ---------------------------
# @dataclass(frozen=True)
# class TrainingSettings:
#     MODULE_ROOT: Path
#     APP_BACKEND: Path

#     DATASET_ROOT: Path   # where training reads from (defaults to ANNO.DATASET_ROOT)
#     INPUT_DIR: Path

#     MODEL_DB_PATH = DB_PATH

#     TEMP_DATA_DIR: Path
#     MODELS_DIR: Path
#     METADATA_DIR: Path

#     MODEL_SPECIES_MAP_PATH: Path
#     GROUP_HASHES_PREV_PATH: Path
#     GROUP_HASHES_CURR_PATH: Path
#     GROUP_SPECIES_DATA_COUNT_PATH: Path
#     GROUP_SPECIES_DATA_COUNT_PREV_PATH: Path

#     CONF_THRESHOLD: float
#     MAX_THREADS: int

#     BASE_MODEL: str
#     EPOCHS: int
#     IMGSZ: int
#     BATCH: int

#     MAX_SPECIES_PER_GROUP: int
#     IMAGE_EXTENSIONS: set[str]
#     TRAIN_DIR_NAME: str
#     VALID_DIR_NAME: str
#     NORMALIZE_PATTERN: str

#     CLEAN_MODELS_DIR_BEFORE_TRAIN: bool

#     def to_dict(self) -> dict:
#         d = asdict(self)
#         for k, v in d.items():
#             if isinstance(v, Path):
#                 d[k] = str(v)
#             if isinstance(v, set):
#                 d[k] = sorted(v)
#         return d

# def _load_training_cfg() -> TrainingSettings:
#     # module_root = _env_path("FISH_TRAINING_ROOT", APP_BACKEND / "modules" / "fish_training", APP_BACKEND)

#     # # Default training dataset = annotator dataset
#     # dataset_root = _env_path(
#     #     "DATASET_ROOT",
#     #     ANNO.DATASET_ROOT / "augmentation" / "training_data",   # ✅ new default
#     #     APP_BACKEND
#     # )
    

#     # # dataset_root = _env_path("DATASET_ROOT", ANNO.DATASET_ROOT, APP_BACKEND)
#     # input_dir    = dataset_root

#     # temp_dir   = _env_path("TEMP_DATA_DIR", module_root / "temp_data", module_root)
#     # models_dir = _env_path("MODELS_DIR",    module_root / "models",    module_root)
#     # meta_dir   = _env_path("METADATA_DIR",  module_root / "metadata",  module_root)

#     # Azure ML writable path
#     AZURE_WRITABLE_DIR = Path(os.getenv("AZUREML_CR_WORKING_DIR", "/mnt/azureml/outputs")).resolve()
#     AZURE_WRITABLE_DIR.mkdir(parents=True, exist_ok=True)

#     # Training module paths redirected to writable directory
#     module_root = _env_path("FISH_TRAINING_ROOT", APP_BACKEND / "modules" / "fish_training", APP_BACKEND)

#     # dataset_root can stay, or also be moved if you write to it
#     dataset_root = _env_path("DATASET_ROOT", AZURE_WRITABLE_DIR / "dataset" / "training_data", AZURE_WRITABLE_DIR)
#     input_dir    = dataset_root

#     temp_dir   = _env_path("TEMP_DATA_DIR", AZURE_WRITABLE_DIR / "temp_data", AZURE_WRITABLE_DIR)
#     models_dir = _env_path("MODELS_DIR",    AZURE_WRITABLE_DIR / "models",    AZURE_WRITABLE_DIR)
#     meta_dir   = _env_path("METADATA_DIR",  AZURE_WRITABLE_DIR / "metadata",  AZURE_WRITABLE_DIR)




#     # Ensure dirs
#     for p in [input_dir, temp_dir, models_dir, meta_dir]:
#         Path(p).mkdir(parents=True, exist_ok=True)

#     model_species_map            = meta_dir / "model_species_map.json"
#     group_hashes_prev            = meta_dir / "group_hashes.json"
#     group_hashes_curr            = meta_dir / "group_hashes_current.json"
#     species_counts_curr          = meta_dir / "group_species_data_count.json"
#     species_counts_prev          = meta_dir / "group_species_data_count_prev.json"

#     conf_thresh = float(os.getenv("CONF_THRESHOLD", "0.7"))
#     max_threads = int(os.getenv("MAX_THREADS", "4"))

#     base_model = str(_env_path("BASE_MODEL", module_root / "yolov8n.pt", module_root))
#     epochs     = int(os.getenv("EPOCHS", "100"))
#     imgsz      = int(os.getenv("IMGSZ", "640"))
#     batch      = int(os.getenv("BATCH", "16"))

#     max_species_per_group = int(os.getenv("MAX_SPECIES_PER_GROUP", "300"))

#     # clean_models_before = _bool(os.getenv("CLEAN_MODELS_DIR_BEFORE_TRAIN"), True)
#     clean_models_before = _bool(os.getenv("CLEAN_MODELS_DIR_BEFORE_TRAIN"), False)


#     cfg = TrainingSettings(
#         MODULE_ROOT=module_root,
#         APP_BACKEND=APP_BACKEND,

#         DATASET_ROOT=dataset_root,
#         INPUT_DIR=input_dir,

#         TEMP_DATA_DIR=temp_dir,
#         MODELS_DIR=models_dir,
#         METADATA_DIR=meta_dir,

#         MODEL_SPECIES_MAP_PATH=model_species_map,
#         GROUP_HASHES_PREV_PATH=group_hashes_prev,
#         GROUP_HASHES_CURR_PATH=group_hashes_curr,
#         GROUP_SPECIES_DATA_COUNT_PATH=species_counts_curr,
#         GROUP_SPECIES_DATA_COUNT_PREV_PATH=species_counts_prev,

#         CONF_THRESHOLD=conf_thresh,
#         MAX_THREADS=max_threads,

#         BASE_MODEL=base_model,
#         EPOCHS=epochs,
#         IMGSZ=imgsz,
#         BATCH=batch,

#         MAX_SPECIES_PER_GROUP=max_species_per_group,
#         IMAGE_EXTENSIONS={".jpg", ".jpeg", ".png", ".bmp", ".gif"},
#         TRAIN_DIR_NAME="train",
#         VALID_DIR_NAME="valid",
#         NORMALIZE_PATTERN=r"[^a-z0-9]",

#         CLEAN_MODELS_DIR_BEFORE_TRAIN=clean_models_before,
#     )

#     return cfg

# TRAIN = _load_training_cfg()


# # ---------------------------
# # Logging helper
# # ---------------------------
# def configure_logging() -> None:
#     import logging
#     # Use annotator log level as the global default
#     level = getattr(logging, ANNO.LOG_LEVEL, logging.INFO)
#     logging.basicConfig(level=level)






















# Old work on 21/11/2025
# # app_backend/config.py
# from __future__ import annotations
# from dataclasses import dataclass, asdict
# from pathlib import Path
# import os
# import numpy as np

# # ---------------------------
# # Helpers
# # ---------------------------
# def _bool(val: str | None, default: bool = False) -> bool:
#     if val is None:
#         return default
#     s = str(val).strip().lower()
#     if s in {"1", "true", "yes", "y", "on"}:  return True
#     if s in {"0", "false", "no", "n", "off"}: return False
#     return default

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

# def _norm_url(url: str) -> str:
#     return url.rstrip("/")

# # ---------------------------
# # App anchors / core paths
# # ---------------------------
# APP_BACKEND  = Path(__file__).resolve().parent            # .../app_backend
# PROJECT_ROOT = APP_BACKEND.parent                         # .../Project GFL

# # Public URL root (used to build shareable links)
# # BASE_PUBLIC_URL = _norm_url(os.environ.get("BASE_PUBLIC_URL", "http://127.0.0.1:5000"))
# # BASE_PUBLIC_URL = _norm_url(os.environ.get("BASE_PUBLIC_URL", "http://192.168.168.137:5000"))
# # BASE_PUBLIC_URL = _norm_url(os.environ.get("BASE_PUBLIC_URL", "https://gfl-api.appmaister.com"))
# BASE_PUBLIC_URL = _norm_url(os.environ.get("BASE_PUBLIC_URL", "https://gfl-backendapi.azurewebsites.net/"))



# # Static served by Flask
# STATIC_DIR    = _env_path("STATIC_DIR", APP_BACKEND / "static", APP_BACKEND)
# PREDICTED_DIR = _env_path("PREDICTED_DIR", STATIC_DIR / "predicted", APP_BACKEND)
# SAVED_DIR     = _env_path("SAVED_DIR",     STATIC_DIR / "saved",     APP_BACKEND)


# print("App Backend root: ", APP_BACKEND)

# # Models Paths for Predictions
# # # Using your existing _env_path helper
# # LOCAL_PREDICTION_MODELS_DIR = str(_env_path("LOCAL_MODELS_DIR", APP_BACKEND.parent / "models_cache", APP_BACKEND.parent))

# LOCAL_PREDICTION_MODELS_DIR = Path(
#     _env_path("LOCAL_MODELS_DIR", APP_BACKEND / "modules" / "fish_training" / "models", APP_BACKEND.parent)
# )


# PREDICTED_OUTPUT_DIR = str(_env_path("OUTPUT_DIR", APP_BACKEND.parent / "Predictions", APP_BACKEND.parent))




# # Ensure on import
# for p in [STATIC_DIR, PREDICTED_DIR, SAVED_DIR]:
#     p.mkdir(parents=True, exist_ok=True)

# # ---------------------------
# # Databases (main + data-collection)
# # ---------------------------
# DB_PATH                 = _env_path("DB_PATH",                 APP_BACKEND.parent / "fish_records.db", APP_BACKEND.parent)
# DB_PATH_DATA_COLLECTION = _env_path("DB_PATH_DATA_COLLECTION", APP_BACKEND.parent / "fish_data.db",    APP_BACKEND.parent)

# # ---------------------------
# # Main pipeline settings (kept as top-level names for existing imports)
# # ---------------------------
# # YOLO model for detect.py (single-model flow)
# # MODEL_PATH      = str(_env_path("MODEL_PATH", APP_BACKEND.parent / "Models" / "fish" / "fishbest_v3.pt", APP_BACKEND.parent))
# MODEL_PATH      = str(_env_path("MODEL_PATH", APP_BACKEND / "modules" / "fish_annotation" / "models" / "fish.pt", APP_BACKEND))

# YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.88"))

# # Output directory for annotated predictions
# OUTPUT_DIR = str(_env_path("OUTPUT_DIR", APP_BACKEND.parent / "Outputs" / "Fishes v3" / "Prediction", APP_BACKEND.parent))
# REFERENCE_OBJECT_WIDTH_INCH = float(os.getenv("REFERENCE_OBJECT_WIDTH_INCH", "1.0"))

# # HSV for neon marker
# LOWER_COLOR = np.array([int(x) for x in os.getenv("LOWER_COLOR", "50,200,200").split(",")], dtype=np.uint8)
# UPPER_COLOR = np.array([int(x) for x in os.getenv("UPPER_COLOR", "70,255,255").split(",")], dtype=np.uint8)

# # Uniqueness model (keras)
# Uniqueness_MODEL_PATH = str(_env_path("UNIQUENESS_MODEL_PATH", APP_BACKEND.parent / "Models" / "Uniqueness" / "fish_embedding_model.keras", APP_BACKEND.parent))
# DB_FOLDER             = str(_env_path("DB_FOLDER",             APP_BACKEND.parent / "DB images", APP_BACKEND.parent))
# # DB_FOLDER_Unknown_Fish= str(_env_path("DB_FOLDER_Unknown_Fish",             APP_BACKEND.parent / "DB images" / "Unknown_fishes", APP_BACKEND.parent))
# DB_FOLDER_Unknown_Fish = os.path.join(DB_FOLDER, "Unknown_fishes")
# os.makedirs(DB_FOLDER_Unknown_Fish, exist_ok=True)

# IMG_SIZE              = (int(os.getenv("IMG_W", "105")), int(os.getenv("IMG_H", "105")))
# THRESHOLD             = float(os.getenv("UNIQUENESS_THRESHOLD", "0.8"))


# # -------------------- Azure Config --------------------
# AZURE_CONFIG = {
#     # "ConnectionString": "https://gflstorageaccount.blob.core.windows.net/dotnetbackend-container",
#     # "Container": "dotnetbackend-container",
#     # # "SasToken": "sp=rcwd&st=2025-09-08T09:08:53Z&se=2026-09-08T17:23:53Z&spr=https&sv=2024-11-04&sr=c&sig=%2FPkcD%2FANv4T6EtPEAjrNw43HQ9Q6nTLM8%2BCkhxq6kw8%3D"
#     # "SasToken": "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-12-31T15:38:34Z&st=2025-09-17T07:23:34Z&spr=https&sig=IOG7m4n1ZfNb8SnCOBBURLvSPYKR0qiGjToBH8AmUzs%3D"


#     "ConnectionString": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container",
#     "Container": "dotnetbackend-container",
#     # "SasUrl": "https://gflstorageblob.blob.core.windows.net/dotnetbackend-container?sp=r&st=2025-10-14T12:00:0…,
#     # "SasToken": "sp=rw&st=2025-10-14T12:54:59Z&se=2025-10-14T21:09:59Z&spr=https&sv=2024-11-04&sr=c&sig=w5T3PVKngvvoGUWQq7keShQ90Hd5yXa8hoJK0KvwH1A%3D"
#     "SasToken": "sp=rcwl&st=2025-10-15T15:17:31Z&se=2026-10-15T23:32:31Z&spr=https&sv=2024-11-04&sr=c&sig=Cu7aKTRa3s7Thflh3pr1mGm%2BG8TflhSrdYHD%2FXgxJ80%3D"

# }

# AZURE_SUBSCRIPTION_ID = "417ecdbc-9b50-421a-82c2-c45f29686df1"
# AZURE_RESOURCE_GROUP = "GFL-App-Rg"
# AZURE_ML_WORKSPACE = "GFL-MLAI"



# # ---------------------------
# # Annotator (merged from modules/fish_annotation/src/config.py)
# # ---------------------------
# @dataclass(frozen=True)
# class AnnotatorSettings:
#     # Paths
#     PROJECT_ROOT: Path
#     MODELS_DIR: Path
#     YOLO_MODEL_PATH: Path
#     DB_PATH: Path

#     DATASET_ROOT: Path               # internal dataset root (builder writes here)
#     STATIC_DIR: Path
#     STATIC_YOLO_DATASET: Path        # served dataset (Flask static)
#     STATIC_FISH_DATASET: Path
#     ANNOTATED_ROOT: Path
#     # BY_SPECIES_ROOT: Path

#     # DB tables / columns
#     TABLE: str
#     IMAGE_COL: str
#     SPECIES_COL: str
#     HANDHELD_COL: str
#     NAME_ID_COL: str | None

#     SPECIES_MAP_TABLE: str
#     SPECIES_MAP_ID: str
#     SPECIES_MAP_NAME: str

#     # Pipeline knobs
#     VAL_FRACTION: float
#     AUGMENTATIONS: int
#     CLEAN: bool
#     DOWNLOAD_TIMEOUT: int

#     # App / server (for URL building only)
#     APP_HOST: str
#     APP_PORT: int
#     DEBUG: bool

#     # Public bases
#     BASE_URL: str                # root
#     STATIC_YOLO_BASE_URL: str    # <BASE_PUBLIC_URL>/static/yolo_dataset

#     # Logging
#     LOG_LEVEL: str

#     def to_dict(self) -> dict:
#         d = asdict(self)
#         for k, v in d.items():
#             if isinstance(v, Path):
#                 d[k] = str(v)
#         return d

# def _load_annotator_cfg() -> AnnotatorSettings:
#     # Root of fish_annotation module
#     anno_root = _env_path("FISH_ANNOTATOR_ROOT", APP_BACKEND / "modules" / "fish_annotation", APP_BACKEND)

#     models_dir      = _env_path("ANNOTATOR_MODELS_DIR",      anno_root / "models", anno_root)
#     yolo_model_path = _env_path("YOLO_MODEL_PATH",           models_dir / "fish.pt", anno_root)
#     # Prefer centralized data-collection DB unless overridden
#     db_path         = _env_path("ANNOTATOR_DB_PATH",         DB_PATH_DATA_COLLECTION, APP_BACKEND)


#     # You requested this default explicitly:
#     dataset_root        = _env_path("DATASET_ROOT",        anno_root / "dataset",        anno_root)
#     static_dir          = STATIC_DIR  # share app static
#     static_yolo_dataset = _env_path("STATIC_YOLO_DATASET", static_dir / "yolo_dataset",  APP_BACKEND)
#     static_fish_dataset = _env_path("STATIC_FISH_DATASET", static_dir / "fish_dataset",  APP_BACKEND)
#     annotated_root      = _env_path("ANNOTATED_ROOT",      anno_root / "annotated_data", anno_root)
#     # by_species_root     = _env_path("BY_SPECIES_ROOT",     anno_root / "by_species",     anno_root)

#     # print("Using annotator dataset root: ", dataset_root)
#     # print("Base public URL: ", BASE_PUBLIC_URL)
#     # print("Using static dir: ", static_dir)
#     # print("Using static yolo dataset: ", static_yolo_dataset)
#     # print("Using static fish dataset: ", static_fish_dataset)

#     # Tables/cols
#     table        = os.getenv("IMAGES_TABLE", "images")
#     image_col    = os.getenv("IMAGE_COL", "image_path")
#     species_col  = os.getenv("SPECIES_COL", "species_id")
#     handheld_col = os.getenv("HANDHELD_COL", "handheld")
#     name_id_col  = os.getenv("NAME_ID_COL") or None

#     species_map_table = os.getenv("SPECIES_MAP_TABLE", "species")
#     species_map_id    = os.getenv("SPECIES_MAP_ID", "id")
#     species_map_name  = os.getenv("SPECIES_MAP_NAME", "species_name")

#     # Knobs
#     val_fraction     = float(os.getenv("VAL_FRACTION", "0.2"))
#     augmentations    = int(os.getenv("AUGMENTATIONS", "2"))
#     clean            = _bool(os.getenv("CLEAN"), False)
#     download_timeout = int(os.getenv("DOWNLOAD_TIMEOUT", "10"))

#     # App host/port control just for building URLs; BASE_PUBLIC_URL is the source of truth
#     app_host = os.getenv("APP_HOST", "192.168.168.137")
#     app_port = int(os.getenv("APP_PORT", "5000"))
#     debug    = _bool(os.getenv("DEBUG"), True)

#     base_url_root        = _norm_url(os.getenv("BASE_URL", BASE_PUBLIC_URL))
#     static_yolo_base_url = f"{base_url_root}/static/yolo_dataset"
#     print("Using static yolo base URL: ", static_yolo_base_url)

#     log_level = os.getenv("LOG_LEVEL", "INFO").upper()

#     cfg = AnnotatorSettings(
#         PROJECT_ROOT=anno_root,
#         MODELS_DIR=models_dir,
#         YOLO_MODEL_PATH=yolo_model_path,
#         DB_PATH=db_path,

#         DATASET_ROOT=dataset_root,
#         STATIC_DIR=static_dir,
#         STATIC_YOLO_DATASET=static_yolo_dataset,
#         STATIC_FISH_DATASET=static_fish_dataset,
#         ANNOTATED_ROOT=annotated_root,
#         # BY_SPECIES_ROOT=by_species_root,

#         TABLE=table,
#         IMAGE_COL=image_col,
#         SPECIES_COL=species_col,
#         HANDHELD_COL=handheld_col,
#         NAME_ID_COL=name_id_col,

#         SPECIES_MAP_TABLE=species_map_table,
#         SPECIES_MAP_ID=species_map_id,
#         SPECIES_MAP_NAME=species_map_name,

#         VAL_FRACTION=val_fraction,
#         AUGMENTATIONS=augmentations,
#         CLEAN=clean,
#         DOWNLOAD_TIMEOUT=download_timeout,

#         APP_HOST=app_host,
#         APP_PORT=app_port,
#         DEBUG=debug,

#         BASE_URL=base_url_root,
#         STATIC_YOLO_BASE_URL=static_yolo_base_url,

#         LOG_LEVEL=log_level,
#     )

#     # Ensure required dirs exist
#     for p in [
#         cfg.MODELS_DIR, cfg.DATASET_ROOT, cfg.STATIC_DIR, cfg.STATIC_YOLO_DATASET,
#         cfg.STATIC_FISH_DATASET, cfg.ANNOTATED_ROOT
#     ]:
#         Path(p).mkdir(parents=True, exist_ok=True)

#     return cfg

# ANNO = _load_annotator_cfg()

# # --- Compatibility aliases for legacy imports ---
# DATASET_ROOT          = ANNO.DATASET_ROOT
# STATIC_YOLO_DATASET   = ANNO.STATIC_YOLO_DATASET
# STATIC_FISH_DATASET   = ANNO.STATIC_FISH_DATASET
# STATIC_YOLO_BASE_URL  = ANNO.STATIC_YOLO_BASE_URL



# # Database path (centralized here instead of hardcoding)
# DB_PATH = _env_path(
#     "DB_PATH",
#     PROJECT_ROOT / "fish_records.db",
#     PROJECT_ROOT
# )



# # ---------------------------
# # Training (merged from modules/fish_training/config.py)
# # ---------------------------
# @dataclass(frozen=True)
# class TrainingSettings:
#     MODULE_ROOT: Path
#     APP_BACKEND: Path

#     DATASET_ROOT: Path   # where training reads from (defaults to ANNO.DATASET_ROOT)
#     INPUT_DIR: Path

#     MODEL_DB_PATH = DB_PATH

#     TEMP_DATA_DIR: Path
#     MODELS_DIR: Path
#     METADATA_DIR: Path

#     MODEL_SPECIES_MAP_PATH: Path
#     GROUP_HASHES_PREV_PATH: Path
#     GROUP_HASHES_CURR_PATH: Path
#     GROUP_SPECIES_DATA_COUNT_PATH: Path
#     GROUP_SPECIES_DATA_COUNT_PREV_PATH: Path

#     CONF_THRESHOLD: float
#     MAX_THREADS: int

#     BASE_MODEL: str
#     EPOCHS: int
#     IMGSZ: int
#     BATCH: int

#     MAX_SPECIES_PER_GROUP: int
#     IMAGE_EXTENSIONS: set[str]
#     TRAIN_DIR_NAME: str
#     VALID_DIR_NAME: str
#     NORMALIZE_PATTERN: str

#     CLEAN_MODELS_DIR_BEFORE_TRAIN: bool

#     def to_dict(self) -> dict:
#         d = asdict(self)
#         for k, v in d.items():
#             if isinstance(v, Path):
#                 d[k] = str(v)
#             if isinstance(v, set):
#                 d[k] = sorted(v)
#         return d

# def _load_training_cfg() -> TrainingSettings:
#     module_root = _env_path("FISH_TRAINING_ROOT", APP_BACKEND / "modules" / "fish_training", APP_BACKEND)

#     # Default training dataset = annotator dataset
#     dataset_root = _env_path(
#         "DATASET_ROOT",
#         ANNO.DATASET_ROOT / "augmentation" / "training_data",   # ✅ new default
#         APP_BACKEND
#     )
    

#     # dataset_root = _env_path("DATASET_ROOT", ANNO.DATASET_ROOT, APP_BACKEND)
#     input_dir    = dataset_root

#     temp_dir   = _env_path("TEMP_DATA_DIR", module_root / "temp_data", module_root)
#     models_dir = _env_path("MODELS_DIR",    module_root / "models",    module_root)
#     meta_dir   = _env_path("METADATA_DIR",  module_root / "metadata",  module_root)

#     # Ensure dirs
#     for p in [input_dir, temp_dir, models_dir, meta_dir]:
#         Path(p).mkdir(parents=True, exist_ok=True)

#     model_species_map            = meta_dir / "model_species_map.json"
#     group_hashes_prev            = meta_dir / "group_hashes.json"
#     group_hashes_curr            = meta_dir / "group_hashes_current.json"
#     species_counts_curr          = meta_dir / "group_species_data_count.json"
#     species_counts_prev          = meta_dir / "group_species_data_count_prev.json"

#     conf_thresh = float(os.getenv("CONF_THRESHOLD", "0.7"))
#     max_threads = int(os.getenv("MAX_THREADS", "4"))

#     base_model = str(_env_path("BASE_MODEL", module_root / "yolov8n.pt", module_root))
#     epochs     = int(os.getenv("EPOCHS", "5"))
#     imgsz      = int(os.getenv("IMGSZ", "640"))
#     batch      = int(os.getenv("BATCH", "16"))

#     max_species_per_group = int(os.getenv("MAX_SPECIES_PER_GROUP", "3"))

#     # clean_models_before = _bool(os.getenv("CLEAN_MODELS_DIR_BEFORE_TRAIN"), True)
#     clean_models_before = _bool(os.getenv("CLEAN_MODELS_DIR_BEFORE_TRAIN"), False)


#     cfg = TrainingSettings(
#         MODULE_ROOT=module_root,
#         APP_BACKEND=APP_BACKEND,

#         DATASET_ROOT=dataset_root,
#         INPUT_DIR=input_dir,

#         TEMP_DATA_DIR=temp_dir,
#         MODELS_DIR=models_dir,
#         METADATA_DIR=meta_dir,

#         MODEL_SPECIES_MAP_PATH=model_species_map,
#         GROUP_HASHES_PREV_PATH=group_hashes_prev,
#         GROUP_HASHES_CURR_PATH=group_hashes_curr,
#         GROUP_SPECIES_DATA_COUNT_PATH=species_counts_curr,
#         GROUP_SPECIES_DATA_COUNT_PREV_PATH=species_counts_prev,

#         CONF_THRESHOLD=conf_thresh,
#         MAX_THREADS=max_threads,

#         BASE_MODEL=base_model,
#         EPOCHS=epochs,
#         IMGSZ=imgsz,
#         BATCH=batch,

#         MAX_SPECIES_PER_GROUP=max_species_per_group,
#         IMAGE_EXTENSIONS={".jpg", ".jpeg", ".png", ".bmp", ".gif"},
#         TRAIN_DIR_NAME="train",
#         VALID_DIR_NAME="valid",
#         NORMALIZE_PATTERN=r"[^a-z0-9]",

#         CLEAN_MODELS_DIR_BEFORE_TRAIN=clean_models_before,
#     )

#     return cfg

# TRAIN = _load_training_cfg()


# # ---------------------------
# # Logging helper
# # ---------------------------
# def configure_logging() -> None:
#     import logging
#     # Use annotator log level as the global default
#     level = getattr(logging, ANNO.LOG_LEVEL, logging.INFO)
#     logging.basicConfig(level=level)
















