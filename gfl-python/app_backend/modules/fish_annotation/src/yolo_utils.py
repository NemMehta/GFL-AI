
from typing import List, Tuple
import numpy as np
import logging
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def yolo_predict_boxes(model: YOLO, img: np.ndarray) -> List[Tuple[float, float, float, float]]:
    results = model.predict(img, verbose=False)
    boxes: List[Tuple[float, float, float, float]] = []
    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        xywhn = getattr(r.boxes, "xywhn", None)
        if xywhn is not None:
            arr = xywhn.cpu().numpy()
            for cx, cy, bw, bh in arr:
                boxes.append((float(cx), float(cy), float(bw), float(bh)))
        else:
            xyxy = r.boxes.xyxy.cpu().numpy()
            h, w = img.shape[:2]
            for x1, y1, x2, y2 in xyxy:
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                cx = (x1 + x2) / (2 * w)
                cy = (y1 + y2) / (2 * h)
                boxes.append((float(cx), float(cy), float(bw), float(bh)))
    return boxes


def resolve_image_path(image_path: str, dataset_root: Path) -> Path:
    dataset_root = Path(dataset_root)

    # Case 1: URL
    if image_path.startswith("http://") or image_path.startswith("https://"):
        if "static/" in image_path:
            rel = image_path.split("static/")[-1]
            return dataset_root / "static" / rel
        else:
            logging.warning("URL without 'static/': %s", image_path)
            return dataset_root / Path(image_path).name

    # Case 2: Absolute path
    image_path = Path(image_path)
    if image_path.is_absolute():
        try:
            return image_path.relative_to(dataset_root)
        except ValueError:
            return image_path  # outside dataset_root

    # Case 3: Relative path
    if str(image_path).startswith(str(dataset_root.name)):
        return dataset_root.parent / image_path

    return dataset_root / image_path





















# from typing import List, Tuple
# import numpy as np
# import logging

# try:
#     from ultralytics import YOLO
# except ImportError:
#     YOLO = None

# def yolo_predict_boxes(model: YOLO, img: np.ndarray) -> List[Tuple[float, float, float, float]]:
#     results = model.predict(img, verbose=False)
#     boxes: List[Tuple[float, float, float, float]] = []
#     for r in results:
#         if not hasattr(r, "boxes") or r.boxes is None:
#             continue
#         xywhn = getattr(r.boxes, "xywhn", None)
#         if xywhn is not None:
#             arr = xywhn.cpu().numpy()
#             for cx, cy, bw, bh in arr:
#                 boxes.append((float(cx), float(cy), float(bw), float(bh)))
#         else:
#             xyxy = r.boxes.xyxy.cpu().numpy()
#             h, w = img.shape[:2]
#             for x1, y1, x2, y2 in xyxy:
#                 bw = (x2 - x1) / w
#                 bh = (y2 - y1) / h
#                 cx = (x1 + x2) / (2 * w)
#                 cy = (y1 + y2) / (2 * h)
#                 boxes.append((float(cx), float(cy), float(bw), float(bh)))
#     return boxes

# # def resolve_image_path(image_path: str, dataset_root) -> 'Path':
# #     from pathlib import Path
# #     if image_path.startswith("http://") or image_path.startswith("https://"):
# #         if "static/" in image_path:
# #             rel = image_path.split("static/")[-1]
# #             return dataset_root / "static" / rel
# #         else:
# #             logging.warning("URL without 'static/': %s", image_path)
# #             return dataset_root / Path(image_path).name
# #     else:
# #         return dataset_root / image_path


# def resolve_image_path(image_path: str, dataset_root) -> 'Path':
#     from pathlib import Path
#     import logging

#     dataset_root = Path(dataset_root)

#     # Case 1: URL
#     if image_path.startswith("http://") or image_path.startswith("https://"):
#         if "static/" in image_path:
#             # logging.info(f"static is present: {image_path}")
#             rel = image_path.split("static/")[-1]
#             # logging.info(f"dataset path is: {dataset_root}")
#             # logging.info(f"Relavant path is: {rel}")
#             return dataset_root / "static" / rel
#         else:
#             logging.warning("URL without 'static/': %s", image_path)
#             return dataset_root / Path(image_path).name

#     # Case 2: Absolute path
#     image_path = Path(image_path)
#     if image_path.is_absolute():
#         # Ensure it's inside dataset_root
#         try:
#             return image_path.relative_to(dataset_root)
#         except ValueError:
#             return image_path  # It's outside dataset_root, return as is

#     # Case 3: Relative path
#     # ⚡ Fix: if image_path already contains dataset_root parts, don't double it
#     if str(image_path).startswith(str(dataset_root.name)):
#         return dataset_root.parent / image_path

#     return dataset_root / image_path



