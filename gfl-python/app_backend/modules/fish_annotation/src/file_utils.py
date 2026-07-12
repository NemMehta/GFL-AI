



import os, re, logging, cv2, yaml
from pathlib import Path
from typing import List

# -------------------- Logger --------------------
logger = logging.getLogger(__name__)

# -------------------- File + Image Utilities --------------------
def ensure_dir(p: Path) -> None:
    """Ensure directory exists."""
    if not p.exists():
        logger.info(f"📂 Creating directory: {p}")
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.exception(f"❌ Failed to create directory {p}: {e}")
        raise

def load_image(path: Path):
    """Load image from disk using OpenCV."""
    try:
        img = cv2.imread(str(path))
        if img is None:
            logger.error(f"❌ Failed to load image: {path}")
            raise FileNotFoundError(f"Failed to load image: {path}")
        logger.debug(f"✅ Loaded image: {path}")
        return img
    except Exception as e:
        logger.exception(f"❌ Exception while loading image {path}: {e}")
        raise

def save_image(dst_path: Path, img) -> None:
    """Save image to disk."""
    try:
        ensure_dir(dst_path.parent)
        ok = cv2.imwrite(str(dst_path), img)
        if not ok:
            logger.error(f"❌ Failed to write image: {dst_path}")
            raise IOError(f"Failed to write image: {dst_path}")
        logger.debug(f"💾 Saved image: {dst_path}")
    except Exception as e:
        logger.exception(f"❌ Exception while saving image {dst_path}: {e}")
        raise

# def write_label_file(label_path: Path, lines: List[str]) -> None:
#     """Write YOLO label file."""
#     try:
#         ensure_dir(label_path.parent)
#         with open(label_path, "w", encoding="utf-8") as f:
#             f.write("\n".join(lines).strip() + "\n")
#         logger.debug(f"📝 Wrote label file: {label_path}")
#     except Exception as e:
#         logger.exception(f"❌ Failed to write label file {label_path}: {e}")
#         raise


def write_label_file(label_path: Path, lines: List[str], overwrite: bool = False) -> None:
    """Write YOLO label file. If overwrite=False, preserve existing files."""
    try:
        ensure_dir(label_path.parent)
        if label_path.exists() and not overwrite:
            logger.info(f"⚠️ Skipping write, label file already exists: {label_path}")
            return
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).strip() + "\n")
        logger.debug(f"📝 Wrote label file: {label_path}")
    except Exception as e:
        logger.exception(f"❌ Failed to write label file {label_path}: {e}")
        raise


def load_existing_class_names(dataset_root: Path) -> List[str]:
    """Load existing class names from dataset.yaml if available."""
    yaml_path = dataset_root / "dataset.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                old = yaml.safe_load(f) or {}
                names = [str(n) for n in old.get("names", []) or []]
                logger.info(f"🔍 Loaded {len(names)} class names from {yaml_path}")
                return names
        except Exception as e:
            logger.exception(f"⚠️ Failed to read dataset.yaml at {yaml_path}: {e}")
    return []

def normalize_name(name: str) -> str:
    """Normalize species/class name to lowercase alphanumeric with spaces."""
    s = re.sub(r"[:;,/\\()\[\]{}]+", " ", name)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def update_dataset_yaml(dataset_root: Path, new_class_names: List[str]) -> List[str]:
    """Update dataset.yaml with new classes (merge with existing)."""
    try:
        yaml_path = dataset_root / "dataset.yaml"
        existing = load_existing_class_names(dataset_root)

        cleaned_new = [normalize_name(n) for n in new_class_names]
        merged = list(existing)
        for n in cleaned_new:
            if n not in merged:
                merged.append(n)
                logger.info(f"➕ Added new class: {n}")

        train_images = dataset_root / "train" / "images"
        val_images   = dataset_root / "valid" / "images"

        def rel(p: Path) -> str:
            return os.path.normpath(os.path.relpath(p, start=yaml_path.parent)).replace("\\", "/")

        data = {
            "path": str(dataset_root.resolve()),
            "train": rel(train_images),
            "val": rel(val_images),
            "nc": len(merged),
            "names": merged,
        }

        ensure_dir(yaml_path.parent)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        logger.info(f"✅ Updated dataset.yaml at {yaml_path} with {len(merged)} classes")
        return merged
    except Exception as e:
        logger.exception(f"❌ Failed to update dataset.yaml in {dataset_root}: {e}")
        raise


















# import os, re, logging, cv2, yaml
# from pathlib import Path
# from typing import List

# def ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)

# def load_image(path: Path):
#     img = cv2.imread(str(path))
#     if img is None:
#         raise FileNotFoundError(f"Failed to load image: {path}")
#     return img

# def save_image(dst_path: Path, img) -> None:
#     ensure_dir(dst_path.parent)
#     ok = cv2.imwrite(str(dst_path), img)
#     if not ok:
#         raise IOError(f"Failed to write image: {dst_path}")

# def write_label_file(label_path: Path, lines: List[str]) -> None:
#     ensure_dir(label_path.parent)
#     with open(label_path, "w", encoding="utf-8") as f:
#         f.write("\n".join(lines).strip() + "\n")

# def load_existing_class_names(dataset_root: Path) -> List[str]:
#     yaml_path = dataset_root / "dataset.yaml"
#     if yaml_path.exists():
#         try:
#             with open(yaml_path, "r", encoding="utf-8") as f:
#                 old = yaml.safe_load(f) or {}
#                 return [str(n) for n in old.get("names", []) or []]
#         except Exception:
#             logging.exception("Failed to read dataset.yaml; starting fresh.")
#     return []

# def normalize_name(name: str) -> str:
#     s = re.sub(r"[:;,/\\()\[\]{}]+", " ", name)
#     s = re.sub(r"\s+", " ", s).strip()
#     return s.lower()

# def update_dataset_yaml(dataset_root: Path, new_class_names: List[str]) -> List[str]:
#     yaml_path = dataset_root / "dataset.yaml"
#     existing = load_existing_class_names(dataset_root)

#     cleaned_new = [normalize_name(n) for n in new_class_names]
#     merged = list(existing)
#     for n in cleaned_new:
#         if n not in merged:
#             merged.append(n)

#     train_images = dataset_root / "train" / "images"
#     val_images   = dataset_root / "valid" / "images"

#     def rel(p: Path) -> str:
#         return os.path.normpath(os.path.relpath(p, start=yaml_path.parent)).replace("\\", "/")

#     data = {
#         "path": str(dataset_root.resolve()),
#         "train": rel(train_images),
#         "val": rel(val_images),
#         "nc": len(merged),
#         "names": merged,
#     }
#     ensure_dir(yaml_path.parent)
#     with open(yaml_path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(data, f, sort_keys=False)
#     return merged

























# import os, re, logging, cv2, yaml
# from pathlib import Path
# from typing import List

# def ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)

# def load_image(path: Path):
#     img = cv2.imread(str(path))
#     if img is None:
#         raise FileNotFoundError(f"Failed to load image: {path}")
#     return img

# def save_image(dst_path: Path, img) -> None:
#     ensure_dir(dst_path.parent)
#     ok = cv2.imwrite(str(dst_path), img)
#     if not ok:
#         raise IOError(f"Failed to write image: {dst_path}")

# def write_label_file(label_path: Path, lines: List[str]) -> None:
#     ensure_dir(label_path.parent)
#     with open(label_path, "w", encoding="utf-8") as f:
#         f.write("\n".join(lines).strip() + "\n")

# def load_existing_class_names(dataset_root: Path) -> List[str]:
#     yaml_path = dataset_root / "dataset.yaml"
#     if yaml_path.exists():
#         try:
#             with open(yaml_path, "r", encoding="utf-8") as f:
#                 old = yaml.safe_load(f) or {}
#                 return [str(n) for n in old.get("names", []) or []]
#         except Exception:
#             logging.exception("Failed to read dataset.yaml; starting fresh.")
#     return []

# def normalize_name(name: str) -> str:
#     s = re.sub(r"[:;,/\\()\[\]{}]+", " ", name)
#     s = re.sub(r"\s+", " ", s).strip()
#     return s.lower()

# def update_dataset_yaml(dataset_root: Path, new_class_names: List[str]) -> List[str]:
#     yaml_path = dataset_root / "dataset.yaml"
#     existing = load_existing_class_names(dataset_root)

#     cleaned_new = [normalize_name(n) for n in new_class_names]
#     merged = list(existing)
#     for n in cleaned_new:
#         if n not in merged:
#             merged.append(n)

#     train_images = dataset_root / "train" / "images"
#     val_images   = dataset_root / "valid" / "images"

#     def rel(p: Path) -> str:
#         return os.path.normpath(os.path.relpath(p, start=yaml_path.parent)).replace("\\", "/")

#     data = {
#         "path": str(dataset_root.resolve()),
#         "train": rel(train_images),
#         "val": rel(val_images),
#         "nc": len(merged),
#         "names": merged,
#     }
#     ensure_dir(yaml_path.parent)
#     with open(yaml_path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(data, f, sort_keys=False)
#     return merged


# def draw_boxes(img_path, label_path, class_names):
#     img = cv2.imread(str(img_path))
#     h, w = img.shape[:2]

#     with open(label_path, "r") as f:
#         for line in f:
#             cls, cx, cy, bw, bh = map(float, line.strip().split())
#             x1 = int((cx - bw/2) * w)
#             y1 = int((cy - bh/2) * h)
#             x2 = int((cx + bw/2) * w)
#             y2 = int((cy + bh/2) * h)
#             cls = int(cls)
#             cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
#             cv2.putText(img, class_names[cls], (x1,y1-5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
#     return img





