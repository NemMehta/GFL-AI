









# group_species.py

import json
import re
import logging
from pathlib import Path

from .config import (
    INPUT_DIR,
    MODEL_SPECIES_MAP_PATH,
    GROUP_SPECIES_DATA_COUNT_PATH,
    GROUP_HASHES_PREV_PATH,
    GROUP_HASHES_CURR_PATH,
    IMAGE_EXTENSIONS,
    TRAIN_DIR_NAME,
    VALID_DIR_NAME,
    NORMALIZE_PATTERN,
)

from app_backend.modules.fish_training.parse_dataset import task1_run

logger = logging.getLogger(__name__)


# ---------------------- JSON Helpers ----------------------
def load_json(path: Path):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.exception(f"❌ Failed to load JSON {path}: {e}")
        return {}


def save_json(path: Path, data):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"💾 Saved JSON → {path}")
    except Exception as e:
        logger.exception(f"❌ Failed to save JSON {path}: {e}")


def normalize_name(name):
    return re.sub(NORMALIZE_PATTERN, "", name.lower())


# ----------------------- Count Images -----------------------
def count_images_by_species(input_dir: Path, species_names):
    """Count train+valid images per species. Species prefix ALWAYS appears at start of filename."""
    species_norm = {normalize_name(s): s for s in species_names}
    species_counts = {s: 0 for s in species_names}

    train_images = input_dir / TRAIN_DIR_NAME / "images"
    valid_images = input_dir / VALID_DIR_NAME / "images"

    def match_species(filename):
        filename_norm = normalize_name(filename)
        # Filename ALWAYS begins with species slug
        for norm_name, original in species_norm.items():
            if filename_norm.startswith(norm_name):
                return original
        return None

    try:
        for folder in [train_images, valid_images]:
            if not folder.exists():
                logger.warning(f"⚠ Missing folder: {folder}")
                continue

            for img in folder.iterdir():
                if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                    sp = match_species(img.name)
                    if sp:
                        species_counts[sp] += 1

        logger.info(f"📊 Image counts → {species_counts}")
        return species_counts

    except Exception as e:
        logger.exception(f"❌ Failed counting images: {e}")
        return species_counts


def task2_run(input_path: Path = INPUT_DIR):
    """
    SINGLE-MODEL TRAINING TRACKER

    ✔ Load current species list
    ✔ Compare to previous species list (from GROUP_HASHES_PREV_PATH)
    ✔ Detect additions/removals
    ✔ Mark retrain_required = True if changed
    ✔ Save CURRENT state only
    ❗ DO NOT overwrite previous state here (this caused skip-retraining bug)
    """

    try:
        logger.info(f"🔍 Parsing dataset at: {input_path}")

        parsed = task1_run(input_path)
        species_list = parsed.get("names", [])

        if not species_list:
            logger.error("❌ No species found in dataset.")
            return

        species_list_sorted = sorted(species_list)

        # Load previous state (DO NOT OVERWRITE!)
        prev_map = load_json(MODEL_SPECIES_MAP_PATH)
        prev_species_list = prev_map.get("species", [])

        logger.info(f"📦 Previous species list: {prev_species_list}")
        logger.info(f"📦 Current species list:  {species_list_sorted}")

        # Detect changes
        species_added = [s for s in species_list_sorted if s not in prev_species_list]
        species_removed = [s for s in prev_species_list if s not in species_list_sorted]

        retrain_required = False

        if species_added:
            logger.info(f"➕ New species detected: {species_added}")
            retrain_required = True

        if species_removed:
            logger.warning(f"🗑 Species removed: {species_removed}")
            retrain_required = True

        if not retrain_required:
            logger.info("✅ Species list unchanged — no retrain needed.")
        else:
            logger.warning("⚡ Species list changed — retrain required.")

        # Count images
        species_counts = count_images_by_species(input_path, species_list_sorted)

        # Save species counts
        save_json(GROUP_SPECIES_DATA_COUNT_PATH, species_counts)

        # Build CURRENT state snapshot
        curr_state = {
            "species": species_list_sorted,
            "counts": species_counts
        }

        # ❗ Save CURRENT state only
        save_json(GROUP_HASHES_CURR_PATH, curr_state)

        # ❗ Update MODEL_SPECIES_MAP_PATH (safe)
        save_json(MODEL_SPECIES_MAP_PATH, {
            "species": species_list_sorted,
            "total_classes": len(species_list_sorted)
        })

        logger.info("✅ task2_run completed successfully.")

        # Return output to training logic
        return {
            "retrain_required": retrain_required,
            "species": species_list_sorted,
            "counts": species_counts,
        }

    except Exception as e:
        logger.exception(f"❌ task2_run failed: {e}")
        raise




