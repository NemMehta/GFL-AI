





# Old work 0n 27/11/2025

# filter_dataset.py  (SINGLE MODEL VERSION)

import shutil
import logging
from pathlib import Path
from collections import Counter
import yaml

from app_backend.modules.fish_training.parse_dataset import task1_run
from app_backend.config import TRAIN, ANNO
from app_backend.modules.fish_training.file_utils import read_label_file, write_label_file

logger = logging.getLogger(__name__)
MASTER_YAML = ANNO.DATASET_ROOT / "augmentation" / "training_data" / "data.yaml"


# -----------------------------------------------------------
# NEW: Load species list from the ONE TRUE YAML used by all
# -----------------------------------------------------------
def load_species_from_master_yaml():
    """
    Loads species list from the unified augmentation YAML:
    augmentation/augment_images/training_data/data.yaml
    """
    yaml_path = (
        ANNO.DATASET_ROOT
        / "training_data"
        / "data.yaml"
    )

    if not yaml_path.exists():
        raise FileNotFoundError(f"❌ Master YAML not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    species_list = data.get("names", [])

    logger.info(f"📄 Loaded species from master YAML: {species_list}")
    return species_list



# -----------------------------------------------------------
# CLASS DISTRIBUTION — now safe because species_list matches IDs
# -----------------------------------------------------------
def count_class_distribution(label_dir, species_list):
    counter = Counter()

    for lbl in label_dir.glob("*.txt"):
        for line in read_label_file(lbl):
            parts = line.split()
            if not parts:
                continue

            cls_id = int(parts[0])

            # Prevent crash if class ID outside range
            if cls_id >= len(species_list):
                logger.warning(f"⚠ Unexpected class_id {cls_id} in {lbl.name}")
                continue

            counter[cls_id] += 1

    logger.info("📊 Class distribution:")
    for cls, count in counter.items():
        logger.info(f"   {species_list[cls]} ({cls}): {count}")




# -----------------------------------------------------------
# MAIN COPY FUNCTION — unchanged
# -----------------------------------------------------------
def filter_and_copy_global(input_base, output_base, species_list):
    """
    Copies training images+labels using the unified dataset root.
    """

    for split in ["train", "valid"]:
        img_dir = input_base / split / "images"
        lbl_dir = input_base / split / "labels"

        out_img_dir = output_base / split / "images"
        out_lbl_dir = output_base / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        copied = 0

        for lbl_path in lbl_dir.glob("*.txt"):
            lines = read_label_file(lbl_path)
            if not lines:
                continue

            img_path = img_dir / f"{lbl_path.stem}.jpg"
            if not img_path.exists():
                img_path = img_dir / f"{lbl_path.stem}.png"

            if not img_path.exists():
                logger.warning(f"⚠️ Missing image for {lbl_path.name}")
                continue

            shutil.copy(img_path, out_img_dir / img_path.name)
            write_label_file(out_lbl_dir / lbl_path.name, lines, overwrite=False)
            copied += 1

        logger.info(f"✔ Copied {copied} pairs to {output_base / split}")

        count_class_distribution(output_base / split / "labels", species_list)





# -----------------------------------------------------------
# CREATE dataset.yaml — unchanged
# -----------------------------------------------------------
def create_dataset_yaml(output_path, species_list):
    data_yaml = {
        "path": str(output_path.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(species_list),
        "names": species_list,
    }

    with open(output_path / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    logger.info(f"📄 Created dataset.yaml → {output_path}")



# -----------------------------------------------------------
# FIXED MAIN DRIVER — loads species from MASTER YAML
# -----------------------------------------------------------
def generate_filtered_group_dataset():
    """
    Build a single unified dataset using locally stored augmented images.
    Loads species ONLY from unified data.yaml
    """

    try:
        temp_output = Path(TRAIN.TEMP_DATA_DIR)
        if temp_output.exists():
            shutil.rmtree(temp_output)
        temp_output.mkdir(parents=True)

        # 🟢 FIXED — load species list ONLY from master YAML
        species_list = load_species_from_master_yaml()

        logger.info(f"🐟 Building SINGLE MODEL dataset with species: {species_list}")

        # Unified augmented dataset root
        input_base = ANNO.DATASET_ROOT / "augmentation" / "training_data"

        filter_and_copy_global(input_base, temp_output, species_list)
        create_dataset_yaml(temp_output, species_list)

        logger.info("✅ Completed single-model dataset generation.")

    except Exception as e:
        logger.exception(f"❌ Failed to generate dataset: {e}")



