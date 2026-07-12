


# compatibility shim — uses the new central config
from app_backend.config import ANNO as CFG, configure_logging










# # annotating/config.py
# """
# Centralized configuration for the Fish Annotator project.

# Usage:
#     from .config import CFG, configure_logging
#     configure_logging()

# Env overrides (examples):
#   FISH_ANNOTATOR_ROOT, DATASET_ROOT, STATIC_DIR, BASE_URL, APP_HOST, APP_PORT, ...
# """

# from __future__ import annotations
# from dataclasses import dataclass, asdict
# from pathlib import Path
# import os

# def _bool(val: str | None, default: bool = False) -> bool:
#     if val is None:
#         return default
#     s = str(val).strip().lower()
#     if s in {"1", "true", "yes", "y", "on"}:  return True
#     if s in {"0", "false", "no", "n", "off"}: return False
#     return default

# def _env_path(name: str, default: Path, root: Path | None = None) -> Path:
#     v = os.getenv(name)
#     if not v:
#         return default.expanduser().resolve()
#     p = Path(v).expanduser()
#     if not p.is_absolute() and root is not None:
#         p = root / p
#     return p.resolve()

# def _norm_base_url(url: str) -> str:
#     return url.rstrip("/")

# @dataclass(frozen=True)
# class Settings:
#     # Paths
#     PROJECT_ROOT: Path
#     MODELS_DIR: Path
#     YOLO_MODEL_PATH: Path
#     DB_PATH: Path

#     DATASET_ROOT: Path
#     STATIC_DIR: Path
#     STATIC_YOLO_DATASET: Path
#     STATIC_FISH_DATASET: Path
#     ANNOTATED_ROOT: Path
#     BY_SPECIES_ROOT: Path

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

#     # App / server
#     APP_HOST: str
#     APP_PORT: int
#     DEBUG: bool

#     # Public base URL (root) + derived dataset base URL
#     BASE_URL: str
#     STATIC_YOLO_BASE_URL: str

#     # Logging
#     LOG_LEVEL: str

#     # Helpful dump
#     def to_dict(self) -> dict:
#         d = asdict(self)
#         for k, v in d.items():
#             if isinstance(v, Path):
#                 d[k] = str(v)
#         return d

# def load() -> Settings:
#     here         = Path(__file__).resolve()
#     default_root = here.parent.parent
#     project_root = _env_path("FISH_ANNOTATOR_ROOT", default_root)

#     models_dir      = _env_path("MODELS_DIR",      project_root / "models",        project_root)
#     yolo_model_path = _env_path("YOLO_MODEL_PATH", models_dir / "fish.pt",         project_root)
#     db_path         = _env_path("DB_PATH",         project_root / "fish_data.db",  project_root)

#     dataset_root        = _env_path("DATASET_ROOT",        project_root / "dataset",        project_root)
#     static_dir          = _env_path("STATIC_DIR",          project_root / "static",         project_root)
#     static_yolo_dataset = _env_path("STATIC_YOLO_DATASET", static_dir / "yolo_dataset",     project_root)
#     static_fish_dataset = _env_path("STATIC_FISH_DATASET", static_dir / "fish_dataset",     project_root)
#     annotated_root      = _env_path("ANNOTATED_ROOT",      project_root / "annotated_data", project_root)
#     by_species_root     = _env_path("BY_SPECIES_ROOT",     project_root / "by_species",     project_root)

#     table        = os.getenv("IMAGES_TABLE", "images")
#     image_col    = os.getenv("IMAGE_COL", "image_path")
#     species_col  = os.getenv("SPECIES_COL", "species_id")
#     handheld_col = os.getenv("HANDHELD_COL", "handheld")
#     name_id_col  = os.getenv("NAME_ID_COL") or None

#     species_map_table = os.getenv("SPECIES_MAP_TABLE", "species")
#     species_map_id    = os.getenv("SPECIES_MAP_ID", "id")
#     species_map_name  = os.getenv("SPECIES_MAP_NAME", "species_name")

#     val_fraction     = float(os.getenv("VAL_FRACTION", "0.2"))
#     augmentations    = int(os.getenv("AUGMENTATIONS", "2"))
#     clean            = _bool(os.getenv("CLEAN"), False)
#     download_timeout = int(os.getenv("DOWNLOAD_TIMEOUT", "10"))

#     app_host = os.getenv("APP_HOST", "192.168.168.137")
#     app_port = int(os.getenv("APP_PORT", "5000"))
#     debug    = _bool(os.getenv("DEBUG"), True)

#     base_url_root        = _norm_base_url(os.getenv("BASE_URL", f"http://{app_host}:{app_port}"))
#     static_yolo_base_url = f"{base_url_root}/static/yolo_dataset"

#     log_level = os.getenv("LOG_LEVEL", "INFO").upper()

#     cfg = Settings(
#         PROJECT_ROOT=project_root,
#         MODELS_DIR=models_dir,
#         YOLO_MODEL_PATH=yolo_model_path,
#         DB_PATH=db_path,

#         DATASET_ROOT=dataset_root,
#         STATIC_DIR=static_dir,
#         STATIC_YOLO_DATASET=static_yolo_dataset,
#         STATIC_FISH_DATASET=static_fish_dataset,
#         ANNOTATED_ROOT=annotated_root,
#         BY_SPECIES_ROOT=by_species_root,

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
#         cfg.STATIC_FISH_DATASET, cfg.ANNOTATED_ROOT, cfg.BY_SPECIES_ROOT
#     ]:
#         Path(p).mkdir(parents=True, exist_ok=True)

#     return cfg

# CFG = load()

# def configure_logging() -> None:
#     import logging
#     level = getattr(logging, CFG.LOG_LEVEL, logging.INFO)
#     logging.basicConfig(level=level)
