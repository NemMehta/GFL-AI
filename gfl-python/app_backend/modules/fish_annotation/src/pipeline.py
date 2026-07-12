



#Old work on 27/11/2025
# src/pipeline.py
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from shutil import copy2
import cv2
import requests
from urllib.parse import urlparse, urlunparse, quote
import os
import json
from tqdm import tqdm

from .file_utils import (
    ensure_dir, load_image, save_image, write_label_file,
)
from .yolo_utils import yolo_predict_boxes
from .config import CFG
from .azure_upload import upload_file_async, upload_augmented_file_async

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import albumentations as A
except ImportError:
    A = None

from app_backend.config import AZURE_CONFIG
import yaml
# -------------------- Logging --------------------
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, CFG.LOG_LEVEL, logging.INFO))

# -------------------- Utilities --------------------
def encode_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        path_parts = parsed.path.split('/')
        encoded_path = '/'.join(quote(p) for p in path_parts)
        return urlunparse(parsed._replace(path=encoded_path))
    except Exception as e:
        logger.exception(f"❌ Failed to encode URL {url}: {e}")
        return url

def _is_url(p: str) -> bool:
    try:
        return urlparse(str(p)).scheme in ("http", "https")
    except Exception as e:
        logger.warning(f"⚠️ Failed to check if path is URL: {p}, {e}")
        return False


def download_image(url: str, dst_path: Path) -> bool:
    try:
        original = url
        url = encode_url(url)
        logger.info(f"⬇️ Downloading: original={original} encoded={url}")
        resp = requests.get(url, stream=True, timeout=CFG.DOWNLOAD_TIMEOUT)
        logger.info(f"⬇️ Status for {url}: {resp.status_code}")
        resp.raise_for_status()
        ensure_dir(dst_path.parent)
        with open(dst_path, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        logger.info(f"⬇️ Downloaded {url} → {dst_path}")
        return True
    except Exception as e:
        logger.warning(f"❌ Failed to download {url} → {dst_path}: {e}")
        return False


def sanitize_species_name(species_name: str) -> str:
    """Normalize species name for filenames (safe alphanumeric with underscores)."""
    import re
    s = species_name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)   # replace non-alphanum with _
    return s.strip("_")

def organize_images_root(rows, project_root: Path):
    """
    Stage input images into Un-Annotated/{species_name}/{handheld}/
    and rename them as <species>_<uuid>.jpg
    """
    un_annotated_root = project_root / "Un-Annotated"
    ensure_dir(un_annotated_root)
    mapping: Dict[Path, Tuple[str, str, str, bool, int]] = {}

    for name_id, species_id, species_name, handheld, img_path, image_id in tqdm(
        rows, desc="Staging images into Un-Annotated"
    ):
        try:
            folder_name = species_name
            subfolder = "Hand-Held" if handheld else "Not-Hand-Held"
            dst_dir = un_annotated_root / folder_name / subfolder
            ensure_dir(dst_dir)

            # 🔑 Build new filename
            safe_species = sanitize_species_name(species_name)
            new_filename = f"{safe_species}_{name_id}{Path(img_path).suffix}"
            dst_path = dst_dir / new_filename

            if _is_url(str(img_path)):
                if download_image(str(img_path), dst_path):
                    mapping[dst_path.resolve()] = (name_id, species_id, species_name, handheld, image_id)
            else:
                p = Path(str(img_path))
                if p.exists():
                    copy2(p, dst_path)
                    mapping[dst_path.resolve()] = (name_id, species_id, species_name, handheld, image_id)
                else:
                    logger.warning(f"⚠️ File not found locally: {img_path}")
        except Exception as e:
            logger.exception(f"❌ Failed to stage image {img_path}: {e}")

    return mapping



def get_class_id(species_name: str, yaml_path: Path) -> int:
    """
    Returns the class ID for a given species name.
    If data.yaml doesn't exist, create it with this species as the first entry.
    If species not found in YAML, append it and return the new index.
    """
    if not yaml_path.exists():
        # Case: first species ever
        cfg = {"names": [species_name], "nc": 1}
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.safe_dump(cfg, f)
        return 0

    # Case: data.yaml exists → load it
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    names = cfg.get("names", [])
    if species_name in names:
        return names.index(species_name)

    # New species → append to YAML
    names.append(species_name)
    cfg["names"] = names
    cfg["nc"] = len(names)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f)

    return len(names) - 1


# -------------------- JSON Normalization --------------------
def _rows_from_api(payload: dict) -> list:
    try:
        sid = str(payload.get("species_id") or payload.get("speciesId"))
        sname = payload.get("species_name") or payload.get("speciesName") or f"class_{sid}"
        handheld = bool(payload.get("handheld", payload.get("isHeld", False)))
        imgs = payload.get("images") or []

        rows = []
        for img in imgs:
            try:
                image_id = int(img.get("imageId"))
                img_s = str(img.get("imagePath"))
                stem = Path(img_s).stem or os.path.basename(img_s).split("?")[0]
                name_id = stem if stem else f"img_{abs(hash(img_s))}"
                rows.append((name_id, sid, sname, handheld, img_s, image_id))
            except Exception as e:
                logger.warning(f"⚠️ Skipping invalid image record {img}: {e}")
        return rows
    except Exception as e:
        logger.exception(f"❌ Failed parsing API payload: {e}")
        return []

def group_records_for_json(export_records: list, payload: dict) -> tuple[dict, dict]:
    try:
        sid = int(payload.get("species_id") or payload.get("speciesId"))
        held = bool(payload.get("handheld", payload.get("isHeld", False)))
        sname = payload.get("species_name") or payload.get("speciesName")

        images_with_txt = []
        for rec in export_records:
            images_with_txt.append({
                "imageId": rec["image_id"],
                "imagePath": rec["image_url"],
                "txtPath": rec["text_url"]
            })

        external_payload = {
            "speciesId": sid,
            "speciesName": sname,
            "isHeld": held,
            "images": images_with_txt
        }

        user_response = {
            "speciesId": sid,
            "speciesName": sname,
            "isHeld": held,
            "images": images_with_txt
        }

        logger.info(f"📦 Prepared user + external payload for species {sname}")
        return user_response, external_payload
    except Exception as e:
        logger.exception(f"❌ Failed to build JSON response: {e}")
        return {}, {}



# -------------------------------------------------------------
# UNIFIED MASTER YAML HANDLING
# -------------------------------------------------------------

# MASTER_YAML_PATH = Path(CFG.DATASET_ROOT) / "augmentation" / "augment_images" / "training_data" / "data.yaml"

MASTER_YAML_PATH = Path(CFG.DATASET_ROOT) / "augmentation" / "training_data" / "data.yaml"


def ensure_species_in_yaml(species_name: str) -> int:
    """Append-only unified YAML using sanitized names + raw class_map."""

    safe = sanitize_species_name(species_name)

    MASTER_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MASTER_YAML_PATH.exists():
        with open(MASTER_YAML_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {"names": [], "nc": 0, "class_map": {}}

    names = data.get("names", [])
    class_map = data.get("class_map", {})

    # Already exists?
    for cid, raw in class_map.items():
        if raw == species_name:
            return int(cid)

    # Append new class
    cid = len(names)
    names.append(safe)
    class_map[cid] = species_name

    # write back
    data["names"] = names
    data["nc"] = len(names)
    data["class_map"] = class_map

    with open(MASTER_YAML_PATH, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    logger.info(f"🆕 Added species '{species_name}' → class {cid}")
    return cid


async def process_from_api(payload: dict,
                           yolo_model_path: Path = CFG.YOLO_MODEL_PATH,
                           dataset_root: Path = CFG.DATASET_ROOT,
                           base_dir: Path = CFG.PROJECT_ROOT):

    logger.info("🚀 Starting dataset processing from API JSON...")

    rows = _rows_from_api(payload)
    project_root = base_dir if base_dir else dataset_root.parent
    staged = organize_images_root(rows, project_root)

    if not staged:
        return {"status": "warning", "message": "No images processed."}

    # YOLO load
    model = YOLO(str(yolo_model_path))

    annotated_root = CFG.ANNOTATED_ROOT
    augment_root = Path(dataset_root) / "augment_images"
    aug_img_dir = augment_root / "images"
    aug_lbl_dir = augment_root / "labels"

    ensure_dir(annotated_root)
    ensure_dir(aug_img_dir)
    ensure_dir(aug_lbl_dir)

    export_records = []

    for src_path, (name_id, sid, sname, handheld, image_id) in tqdm(staged.items()):
        try:
            img = load_image(src_path)
            boxes = yolo_predict_boxes(model, img)
            if not boxes:
                logger.info(f"⚠️ No YOLO boxes for image {src_path} (species={sname}, id={sid})")
                continue
            
            logger.info(f"✅ YOLO boxes for {src_path}: {len(boxes)}")

            # --- CLASS ID (fixed)
            class_id = ensure_species_in_yaml(sname)

            # --- Annotated
            img_boxed = img.copy()
            h, w = img.shape[:2]
            for cx, cy, bw, bh in boxes:
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img_boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)

            out_img_annot = annotated_root / f"{src_path.stem}_annotated.jpg"
            out_lbl_annot = annotated_root / f"{src_path.stem}_annotated.txt"

            save_image(out_img_annot, img_boxed)
            write_label_file(out_lbl_annot,
                             [f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                              for (cx, cy, bw, bh) in boxes])

            # Upload annotated
            img_url = await upload_file_async(sname, handheld, out_img_annot, AZURE_CONFIG, "Annotated/images")
            txt_url = await upload_file_async(sname, handheld, out_lbl_annot, AZURE_CONFIG, "Annotated/labels")

            export_records.append({
                "species_id": sid,
                "handheld": handheld,
                "image_url": img_url,
                "text_url": txt_url,
                "image_id": image_id
            })

            # --- Clean version (augment_images)
            out_img_clean = aug_img_dir / f"{src_path.stem}.jpg"
            out_lbl_clean = aug_lbl_dir / f"{src_path.stem}.txt"

            save_image(out_img_clean, img)
            write_label_file(out_lbl_clean,
                             [f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                              for (cx, cy, bw, bh) in boxes])

            await upload_augmented_file_async(out_img_clean, AZURE_CONFIG, "images")
            await upload_augmented_file_async(out_lbl_clean, AZURE_CONFIG, "labels")

        except Exception as e:
            logger.exception(f"❌ YOLO failed: {e}")

    # Upload YAML once
    await upload_augmented_file_async(MASTER_YAML_PATH, AZURE_CONFIG, "training_data")

    user_response, _ = group_records_for_json(export_records, payload)
    return user_response




