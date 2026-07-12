# augmentations.py
import cv2
try:
    import albumentations as A
except ImportError:
    A = None

def make_aug_pipelines():
    if A is None:
        raise RuntimeError("albumentations not installed. pip install albumentations")
    bboxp = A.BboxParams(format="yolo", label_fields=["class_ids"])
    return [
        ("fliph",  A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bboxp)),
        ("bright", A.Compose([A.RandomBrightnessContrast(0.1, 0.0, p=1.0)], bbox_params=bboxp)),
        ("rot15",  A.Compose([A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=1.0)], bbox_params=bboxp)),
        ("clahe",  A.Compose([A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)], bbox_params=bboxp)),
    ]
