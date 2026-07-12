import argparse
from pathlib import Path
from .config import CFG

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build/append YOLO dataset from SQLite; stratify AFTER augmentation; dynamic YAML; no test folder."
    )

    BASE_DIR = CFG.PROJECT_ROOT

    # DB + tables/cols
    p.add_argument("--db", default=CFG.DB_PATH, type=Path)
    p.add_argument("--table", default=CFG.TABLE)
    p.add_argument("--image-col", default=CFG.IMAGE_COL)
    p.add_argument("--species-col", default=CFG.SPECIES_COL)
    p.add_argument("--handheld-col", default=CFG.HANDHELD_COL)
    p.add_argument("--name-id-col", default=CFG.NAME_ID_COL)

    # Species map
    p.add_argument("--species-map-table", default=CFG.SPECIES_MAP_TABLE)
    p.add_argument("--species-map-id", default=CFG.SPECIES_MAP_ID)
    p.add_argument("--species-map-name", default=CFG.SPECIES_MAP_NAME)

    # YOLO / dataset
    p.add_argument("--yolo-model", default=CFG.YOLO_MODEL_PATH, type=Path)
    p.add_argument("--dataset-root", default=CFG.DATASET_ROOT, type=Path)

    # Base dir
    p.add_argument("--base-dir", default=BASE_DIR, type=Path)

    # Splits
    p.add_argument("--val-fraction", type=float, default=CFG.VAL_FRACTION)
    p.add_argument("--augmentations", type=int, default=CFG.AUGMENTATIONS, help="Ignored; all augs applied.")
    p.add_argument("--clean", action="store_true", help="Ignored", default=CFG.CLEAN)

    # Logging
    p.add_argument("--log-level", default=CFG.LOG_LEVEL, choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args([])  # so it works without CLI args





















# import argparse
# from pathlib import Path

# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(
#         description="Build/append YOLO dataset from SQLite; stratify AFTER augmentation; dynamic YAML; no test folder."
#     )

#     # ---- Base dir ----
#     BASE_DIR = Path(__file__).resolve().parent.parent   # project root (adjust if needed)
#     print(f"Base dir is: {BASE_DIR}")

#     # DB + tables/cols
#     p.add_argument("--db", default=BASE_DIR / "fish_data.db", type=Path)
#     p.add_argument("--table", default="images")
#     p.add_argument("--image-col", default="image_path")
#     p.add_argument("--species-col", default="species_id")
#     p.add_argument("--handheld-col", default="handheld")
#     p.add_argument("--name-id-col", default=None)

#     # Species map
#     p.add_argument("--species-map-table", default="species")
#     p.add_argument("--species-map-id", default="id")
#     p.add_argument("--species-map-name", default="species_name")

#     # YOLO / dataset
#     p.add_argument("--yolo-model", default=BASE_DIR / "models" / "fish.pt", type=Path)
#     p.add_argument("--dataset-root", default=BASE_DIR / "dataset", type=Path)

#     # Base dir
#     p.add_argument("--base-dir", default=BASE_DIR, type=Path)

#     # Splits
#     p.add_argument("--val-fraction", type=float, default=0.2)
#     p.add_argument("--augmentations", type=int, default=2, help="Ignored; all augs applied.")
#     p.add_argument("--clean", action="store_true", help="Ignored")

#     # Logging
#     p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

#     return p.parse_args([])  # so it works without CLI args













# import argparse
# from pathlib import Path

# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(
#         description="Build/append YOLO dataset from SQLite; stratify AFTER augmentation; dynamic YAML; no test folder."
#     )
#     # DB + tables/cols
#     p.add_argument("--db", default=Path(r"D:\\Rahul Puri Data\\Projects\\fish_annotator\\fish_data.db"), type=Path)
#     p.add_argument("--table", default="images")
#     p.add_argument("--image-col", default="image_path")
#     p.add_argument("--species-col", default="species_id")
#     p.add_argument("--handheld-col", default="handheld")
#     p.add_argument("--name-id-col", default=None)

#     # Species map
#     p.add_argument("--species-map-table", default="species")
#     p.add_argument("--species-map-id", default="id")
#     p.add_argument("--species-map-name", default="species_name")

#     # YOLO / dataset
#     p.add_argument("--yolo-model", default=Path(r"D:\\Rahul Puri Data\\Projects\\fish_annotator\\models\\fish.pt"), type=Path)
#     p.add_argument("--dataset-root", default=Path(r"D:\\Rahul Puri Data\\Projects\\fish_annotator\\dataset"), type=Path)

#     # Base dir
#     p.add_argument("--base-dir", default=Path(r"D:\\Rahul Puri Data\\Projects\\fish_annotator"), type=Path)

#     # Splits
#     p.add_argument("--val-fraction", type=float, default=0.2)
#     p.add_argument("--augmentations", type=int, default=2, help="Ignored; all augs applied.")
#     p.add_argument("--clean", action="store_true", help="Ignored")

#     # Logging
#     p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

#     return p.parse_args([])  # so it works without CLI args
