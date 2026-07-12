





# ---------------------------------------------------------------
# augment_pipeline.py  (PART 1 / 3)
# ---------------------------------------------------------------

import os
import math
import logging
import tempfile
from pathlib import Path
import requests
import cv2
import yaml
from tqdm import tqdm
from azure.storage.blob import ContainerClient
from azure.storage.blob import ContainerClient, BlobServiceClient
from typing import List, Dict
from app_backend.modules.fish_training.file_utils import (
    ensure_dir,
    load_image,
    save_image,
    write_label_file
)
from app_backend.modules.fish_training.augmentations import make_aug_pipelines
from app_backend.config import ANNO, TRAIN as CFG
from app_backend.modules.fish_training.azure_upload import upload_training_file_async
from app_backend.config import AZURE_CONFIG, APP_BACKEND
from app_backend.modules.fish_training.augmentations import make_aug_pipelines
# from app_backend.modules.fish_training.augment_pipeline import download_yaml_from_blob

os.environ["YOLO_NO_CHECK"] = "1"   # Disable Ultralytics version check

# ----------------------------------------------------------------
# DEBUG LOGGING  (YOU CHOSE OPTION A — KEEP EXACT FILE PATH)
# ----------------------------------------------------------------
# Azure ML safe, cross-platform writable log file
LOG_PATH = Path(os.getenv("AZUREML_LOG_DIR", "/mnt/azureml/outputs")) / "debug_train.log"

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# MASTER_YAML_BLOB = "augmentation/augment_images/training_data/data.yaml"
MASTER_YAML_BLOB = "augmentation/training_data/data.yaml"
MASTER_YAML_LOCAL = Path("/mnt/azureml/outputs/dataset/training_data/data.yaml")



# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def sanitize_species_name(name: str) -> str:
    import re
    name = name.lower()
    name = name.replace("(", " ").replace(")", " ")
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return "_".join(name.split())



def get_container_client(config: dict):
    try:
        base_url = config["ConnectionString"]
        sas_token = config["SasToken"]
        return ContainerClient.from_container_url(f"{base_url}?{sas_token}")
    except Exception:
        logger.exception("❌ Failed to initialize Azure container client")
        raise



def list_blobs(prefix: str, config: dict):
    """
    Correct: Use Azure prefix filtering on server side.
    """
    try:
        container = get_container_client(config)

        # IMPORTANT: give prefix to Azure
        blobs = list(container.list_blobs(name_starts_with=prefix))

        return [b.name for b in blobs]

    except Exception:
        logger.exception(f"❌ Failed to list blobs with prefix {prefix}")
        return []


def download_blob(blob_name: str, dst_path: Path, config: dict):
    """
    Download one blob to local disk.
    """
    try:
        container = get_container_client(config)
        blob = container.get_blob_client(blob_name)

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dst_path, "wb") as f:
            f.write(blob.download_blob().readall())

        logger.info(f"⬇️ Downloaded {blob_name} → {dst_path}")

    except Exception:
        logger.exception(f"❌ Failed to download blob {blob_name}")


def clamp_bbox(bbox):
    """
    Ensure YOLO bbox floats are always inside [0–1].
    """
    cx, cy, bw, bh = bbox
    return (
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(max(bw, 0.0), 1.0),
        min(max(bh, 0.0), 1.0),
    )



def sanitize_species_name(name: str) -> str:
    """
    Converts: 'Red Drum (Sciaenops ocellatus)' 
    into:     'red_drum_sciaenops_ocellatus'
    """
    import re
    name = name.lower()
    name = name.replace("(", " ").replace(")", " ")
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = "_".join(name.split())
    return name

def load_dataset_yaml(path: Path):
    """Load sanitized names[] + raw class_map."""
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names", [])
    class_map_raw = data.get("class_map", {})

    # Build sanitized → classID map
    class_map = {}
    for cid, raw_name in class_map_raw.items():
        slug = sanitize_species_name(raw_name)
        class_map[slug] = int(cid)

    return names, class_map



def get_class_id(species_name: str, class_map: Dict[str, int]) -> int:
    slug = sanitize_species_name(species_name)
    return class_map.get(slug, 0)   # default 0 (but should not happen)



def update_training_data_yaml(input_species: list, yaml_path: Path):
    """
    Overwrite augmentation/training_data/data.yaml
    based ONLY on user-input species list.
    """
    import yaml
    from pathlib import Path

    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    names = []
    class_map = {}

    for i, sp in enumerate(input_species):
        original = sp["speciesName"]
        safe = sanitize_species_name(original)
        names.append(safe)
        class_map[i] = original

    data = {
        "names": names,
        "nc": len(names),
        "class_map": class_map
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    logger.info(f"✅ Updated training data.yaml → {yaml_path}")
    logger.info(f"Classes: {names}")


def download_yaml_from_blob():
    account_url = "https://gflstorageblob.blob.core.windows.net"
    container_name = "dotnetbackend-container"
    sas_token = "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2026-05-31T18:55:03Z&st=2025-12-24T10:40:03Z&spr=https&sig=42CYl4Y4TDTmbnPtV0kMwZ1YLzIL%2BwhUaoH5pglD4iE%3D"
    
	# account_url = AZURE_CONFIG["ConnectionString"]
	# container_name = AZURE_CONFIG["Container"]
	# sas_token = AZURE_CONFIG["SasToken"]

    blob_name = "augmentation/training_data/data.yaml"

    # Writable directory
    dataset_root = Path("/mnt/azureml/outputs/dataset/training_data")
    dataset_root.mkdir(parents=True, exist_ok=True)
    yaml_path = dataset_root / "data.yaml"

    # Correct Authentication using SAS token
    blob_service = BlobServiceClient(
        account_url=account_url,
        credential=sas_token
    )

    blob_client = blob_service.get_blob_client(
        container=container_name,
        blob=blob_name
    )

    with open(yaml_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    return yaml_path




# ---------------------------------------------------------------
# augment_pipeline.py  (PART 2 / 3)
# ---------------------------------------------------------------


# async def build_augmented_dataset(species_name: str, val_fraction: float = ANNO.VAL_FRACTION):

#     logger.info(f"🔄 Building augmented dataset for: {species_name}")

#     # Load unified YAML
#     yaml_path = download_yaml_from_blob()
#     names, class_map = load_dataset_yaml(yaml_path)

#     slug = sanitize_species_name(species_name)
#     cid = class_map.get(slug, 0)
#     logger.info(f"Class ID for {species_name} → {cid}")

#     # -----------------------------
#     # Download augmented images
#     # -----------------------------
#     safe = slug
#     dataset_root = ANNO.DATASET_ROOT / "augmentation" / "training_data"

#     train_img = dataset_root / "train" / "images"
#     train_lbl = dataset_root / "train" / "labels"
#     valid_img = dataset_root / "valid" / "images"
#     valid_lbl = dataset_root / "valid" / "labels"

#     # # reset on first species
#     # if not dataset_root.exists() or not any(dataset_root.iterdir()):
#     #     for d in [train_img, train_lbl, valid_img, valid_lbl]:
#     #         if d.exists():
#     #             import shutil
#     #             shutil.rmtree(d)
#     #         d.mkdir(parents=True, exist_ok=True)

#     # Ensure dataset directories exist (APPEND-ONLY, DO NOT DELETE OLD DATA)
#     for d in [train_img, train_lbl, valid_img, valid_lbl]:
#         d.mkdir(parents=True, exist_ok=True)


#     tmp = Path(tempfile.mkdtemp())
#     imgs_tmp = tmp / "images"
#     lbl_tmp = tmp / "labels"
#     ensure_dir(imgs_tmp)
#     ensure_dir(lbl_tmp)

#     # list & download blobs
#     container = ContainerClient.from_container_url(
#         f"{AZURE_CONFIG['ConnectionString']}?{AZURE_CONFIG['SasToken']}"
#     )

#     # img_blobs = [b.name for b in container.list_blobs(name_starts_with="augmentation/augment_images/images/")
#     #              if Path(b.name).stem.lower().startswith(safe)]

#     # lbl_blobs = [b.name for b in container.list_blobs(name_starts_with="augmentation/augment_images/labels/")
#     #              if Path(b.name).stem.lower().startswith(safe)]

#     img_blobs = [b.name for b in container.list_blobs(name_starts_with="augmentation/images/")
#                  if Path(b.name).stem.lower().startswith(safe)]

#     lbl_blobs = [b.name for b in container.list_blobs(name_starts_with="augmentation/labels/")
#                  if Path(b.name).stem.lower().startswith(safe)]


#     for blob in img_blobs:
#         download = container.get_blob_client(blob)
#         with open(imgs_tmp / Path(blob).name, "wb") as f:
#             f.write(download.download_blob().readall())

#     for blob in lbl_blobs:
#         download = container.get_blob_client(blob)
#         with open(lbl_tmp / Path(blob).name, "wb") as f:
#             f.write(download.download_blob().readall())

#     # -----------------------------
#     # Train/valid split
#     # -----------------------------
#     all_imgs = sorted(imgs_tmp.glob("*.jpg"))
#     all_lbls = {x.stem: x for x in lbl_tmp.glob("*.txt")}

#     total = len(all_imgs)
#     val_count = max(1, int(total * val_fraction))

#     valid_imgs = all_imgs[:val_count]
#     train_imgs = all_imgs[val_count:]

#     aug_pipes = make_aug_pipelines()

#     # -----------------------------
#     # Save clean & augmented
#     # -----------------------------
#     for split, items in [("train", train_imgs), ("valid", valid_imgs)]:
#         out_img_dir = train_img if split == "train" else valid_img
#         out_lbl_dir = train_lbl if split == "train" else valid_lbl

#         for img_path in items:
#             lbl_path = all_lbls.get(img_path.stem)
#             if not lbl_path:
#                 continue

#             img = load_image(img_path)
#             with open(lbl_path) as f:
#                 bboxes = [tuple(map(float, line.split()[1:])) for line in f if len(line.split()) == 5]

#             # write base image
#             save_image(out_img_dir / img_path.name, img)
#             write_label_file(out_lbl_dir / f"{img_path.stem}.txt",
#                              [f"{cid} {cx} {cy} {bw} {bh}" for (cx, cy, bw, bh) in bboxes])

#             await upload_training_file_async(out_img_dir / img_path.name, AZURE_CONFIG,
#                                              f"training_data/{split}/images")
#             await upload_training_file_async(out_lbl_dir / f"{img_path.stem}.txt", AZURE_CONFIG,
#                                              f"training_data/{split}/labels")

#             if split == "train":
#                 for tag, pipe in aug_pipes:
#                     try:
#                         out = pipe(image=img, bboxes=bboxes, class_ids=[cid] * len(bboxes))
#                         if not out["bboxes"]:
#                             continue

#                         new_img = out["image"]
#                         new_bboxes = out["bboxes"]

#                         aug_name = f"{slug}_{img_path.stem}_aug_{tag}.jpg"
#                         img_out = out_img_dir / aug_name
#                         lbl_out = out_lbl_dir / aug_name.replace(".jpg", ".txt")

#                         save_image(img_out, new_img)

#                         write_label_file(lbl_out,
#                                          [f"{cid} {cx} {cy} {bw} {bh}" for (cx, cy, bw, bh) in new_bboxes])

#                         await upload_training_file_async(img_out, AZURE_CONFIG, "training_data/train/images")
#                         await upload_training_file_async(lbl_out, AZURE_CONFIG, "training_data/train/labels")

#                     except Exception as e:
#                         logger.exception(f"Aug failed {img_path.name}: {e}")

#     logger.info("✅ Completed augmentation")
#     return {"status": "success", "species": species_name}



async def build_augmented_dataset(
    species_name: str,
    val_fraction: float = ANNO.VAL_FRACTION,
):
    logger.info(f"🔄 Building augmented dataset for: {species_name}")

    # ------------------------------------------------------------
    # Load species schema (GLOBAL SOURCE OF TRUTH)
    # ------------------------------------------------------------
    yaml_path = download_yaml_from_blob()
    names, class_map = load_dataset_yaml(yaml_path)

    slug = sanitize_species_name(species_name)
    if slug not in class_map:
        raise ValueError(f"Species '{species_name}' not found in data.yaml")

    cid = class_map[slug]
    logger.info(f"🆔 Class ID for {species_name}: {cid}")

    # ------------------------------------------------------------
    # Dataset paths (APPEND-ONLY)
    # ------------------------------------------------------------
    dataset_root = ANNO.DATASET_ROOT / "augmentation" / "training_data"

    train_img = dataset_root / "train" / "images"
    train_lbl = dataset_root / "train" / "labels"
    valid_img = dataset_root / "valid" / "images"
    valid_lbl = dataset_root / "valid" / "labels"

    for d in [train_img, train_lbl, valid_img, valid_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # TEMP DOWNLOAD DIR (MUST BE CLEANED)
    # ------------------------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="aug_"))
    imgs_tmp = tmp / "images"
    lbl_tmp = tmp / "labels"
    ensure_dir(imgs_tmp)
    ensure_dir(lbl_tmp)

    try:
        # --------------------------------------------------------
        # Download source images + labels from Azure (ONCE)
        # --------------------------------------------------------
        container = ContainerClient.from_container_url(
            f"{AZURE_CONFIG['ConnectionString']}?{AZURE_CONFIG['SasToken']}"
        )

        img_blobs = [
            b.name for b in container.list_blobs(name_starts_with="augmentation/images/")
            if Path(b.name).stem.lower().startswith(slug)
        ]

        lbl_blobs = [
            b.name for b in container.list_blobs(name_starts_with="augmentation/labels/")
            if Path(b.name).stem.lower().startswith(slug)
        ]

        if not img_blobs:
            logger.warning(f"⚠ No images found for species {species_name}")
            return {"status": "skipped", "species": species_name}

        for blob in img_blobs:
            dst = imgs_tmp / Path(blob).name
            if dst.exists():
                continue
            with open(dst, "wb") as f:
                f.write(container.get_blob_client(blob).download_blob().readall())

        for blob in lbl_blobs:
            dst = lbl_tmp / Path(blob).name
            if dst.exists():
                continue
            with open(dst, "wb") as f:
                f.write(container.get_blob_client(blob).download_blob().readall())

        # --------------------------------------------------------
        # Train / Validation split
        # --------------------------------------------------------
        all_imgs = sorted(imgs_tmp.glob("*.jpg"))
        all_lbls = {p.stem: p for p in lbl_tmp.glob("*.txt")}

        total = len(all_imgs)
        val_count = max(1, int(total * val_fraction))

        valid_imgs = all_imgs[:val_count]
        train_imgs = all_imgs[val_count:]

        aug_pipes = make_aug_pipelines()

        # --------------------------------------------------------
        # Save base + augmented data
        # --------------------------------------------------------
        for split, items in [("train", train_imgs), ("valid", valid_imgs)]:
            out_img_dir = train_img if split == "train" else valid_img
            out_lbl_dir = train_lbl if split == "train" else valid_lbl

            for img_path in items:
                lbl_path = all_lbls.get(img_path.stem)
                if not lbl_path:
                    continue

                out_img_path = out_img_dir / img_path.name
                out_lbl_path = out_lbl_dir / f"{img_path.stem}.txt"

                # Skip if already processed (idempotent)
                if out_img_path.exists() and out_lbl_path.exists():
                    continue

                img = load_image(img_path)

                with open(lbl_path, "r") as f:
                    bboxes = [
                        tuple(map(float, line.split()[1:]))
                        for line in f
                        if len(line.split()) == 5
                    ]

                # Save base image
                save_image(out_img_path, img)
                write_label_file(
                    out_lbl_path,
                    [f"{cid} {cx} {cy} {bw} {bh}" for (cx, cy, bw, bh) in bboxes],
                )

                # Upload base files
                await upload_training_file_async(
                    out_img_path, AZURE_CONFIG, f"training_data/{split}/images"
                )
                await upload_training_file_async(
                    out_lbl_path, AZURE_CONFIG, f"training_data/{split}/labels"
                )

                # --------------------------------------------------
                # AUGMENTATIONS (TRAIN ONLY, LIMITED)
                # --------------------------------------------------
                if split == "train":
                    for tag, pipe in aug_pipes[:3]:  # 🔒 LIMIT AUGS
                        try:
                            out = pipe(
                                image=img,
                                bboxes=bboxes,
                                class_ids=[cid] * len(bboxes),
                            )
                            if not out["bboxes"]:
                                continue

                            aug_name = f"{slug}_{img_path.stem}_aug_{tag}.jpg"
                            aug_img_path = out_img_dir / aug_name
                            aug_lbl_path = out_lbl_dir / aug_name.replace(".jpg", ".txt")

                            if aug_img_path.exists():
                                continue

                            save_image(aug_img_path, out["image"])
                            write_label_file(
                                aug_lbl_path,
                                [
                                    f"{cid} {cx} {cy} {bw} {bh}"
                                    for (cx, cy, bw, bh) in out["bboxes"]
                                ],
                            )

                            await upload_training_file_async(
                                aug_img_path, AZURE_CONFIG, "training_data/train/images"
                            )
                            await upload_training_file_async(
                                aug_lbl_path, AZURE_CONFIG, "training_data/train/labels"
                            )

                        except Exception as e:
                            logger.exception(
                                f"❌ Augmentation failed for {img_path.name}: {e}"
                            )

        logger.info(f"✅ Completed augmentation for {species_name}")
        return {"status": "success", "species": species_name}

    finally:
        # --------------------------------------------------------
        # 🚨 CRITICAL: CLEAN TEMP DIR (AVOID DISK FILL)
        # --------------------------------------------------------
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        logger.info(f"🧹 Cleaned temp directory: {tmp}")













# ---------------------------------------------------------------
# augment_yaml_builder.py   (PART 3 / 3)
# ---------------------------------------------------------------

def detect_species_in_dataset(dataset_root: Path) -> List[str]:
    """
    Scan dataset/train/images/* for species prefixes.
    This ensures YAML uses ONLY species that actually have images.
    """
    species = set()

    train_dir = dataset_root / "train" / "images"
    if not train_dir.exists():
        return []

    for img in train_dir.glob("*.jpg"):
        # strict species slug = first segment before first "_"
        stem = img.stem
        slug = stem.split("_")[0]
        species.add(slug)

    species = sorted(species)
    logger.info(f"🔍 Detected species in dataset: {species}")
    return species


def rebuild_labels_with_class_ids(dataset_root: Path, class_map: Dict[str, int]):
    """
    Rewrite all label files in train/labels and valid/labels to use
    the final class IDs from class_map.
    """
    for split in ["train", "valid"]:
        label_dir = dataset_root / split / "labels"
        for lbl_file in label_dir.glob("*.txt"):
            try:
                stem = lbl_file.stem
                species_slug = stem.split("_")[0]

                if species_slug not in class_map:
                    logger.warning(f"⚠️ Species slug '{species_slug}' not in class_map")
                    continue

                cid = class_map[species_slug]

                new_lines = []
                with open(lbl_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            # ignore old class id, rewrite with final
                            _, cx, cy, bw, bh = parts
                            new_lines.append(f"{cid} {cx} {cy} {bw} {bh}")

                with open(lbl_file, "w") as f:
                    f.write("\n".join(new_lines))

            except Exception as e:
                logger.exception(f"❌ Failed relabeling {lbl_file}: {e}")



