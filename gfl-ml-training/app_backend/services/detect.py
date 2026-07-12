import os
from ultralytics import YOLO
from app_backend.config import MODEL_PATH, YOLO_CONFIDENCE
import logging

model = YOLO(MODEL_PATH)

import os
import logging
from ultralytics import YOLO
from app_backend.config import MODEL_PATH, MODEL_PATH, YOLO_CONFIDENCE

# Load both models
# MODEL_PATH_FISH → binary fish detector model
# MODEL_PATH → fish species classifier
fish_detector = YOLO(MODEL_PATH)
species_detector = YOLO(MODEL_PATH)


def detect_fish(img_bgr):
    """
    Returns (confidence, fish_label, box_xyxy)
    box_xyxy as (x1, y1, x2, y2) ints
    """
    res = fish_detector(img_bgr)[0]
    if len(res.boxes) == 0:
        raise ValueError("No fish detected.")
    # pick the highest-confidence detection
    i = int(res.boxes.conf.argmax().item())
    conf = float(res.boxes.conf[i].item())
    cls_id = int(res.boxes.cls[i].item())
    fish_label = fish_detector.names[cls_id] if hasattr(fish_detector, "names") else str(cls_id)
    x1, y1, x2, y2 = map(int, res.boxes.xyxy[i].tolist())
    return conf, fish_label, (x1, y1, x2, y2)

def detect_fish_species(img_bgr):
    """
    Returns (confidence, specie_name, box_xyxy)
    """
    res = species_detector(img_bgr)[0]
    if len(res.boxes) == 0:
        raise ValueError("No species detection.")
    i = int(res.boxes.conf.argmax().item())
    conf = float(res.boxes.conf[i].item())
    cls_id = int(res.boxes.cls[i].item())
    specie_name = species_detector.names[cls_id] if hasattr(species_detector, "names") else str(cls_id)
    x1, y1, x2, y2 = map(int, res.boxes.xyxy[i].tolist())
    return conf, specie_name, (x1, y1, x2, y2)



# def detect_fish(img):
#     """
#     Detects whether a fish exists in the image (binary classification) 
#     and checks orientation (horizontal placement required).
#     """
#     results = fish_detector(img, conf=0.3)
#     names = fish_detector.names

#     if results[0].boxes is None or len(results[0].boxes.xyxy) == 0:
#         logging.warning("No fish detected in image.")
#         raise ValueError("No fish detected.")

#     # Taking the first detected fish
#     box = results[0].boxes.xyxy[0]
#     conf = float(results[0].boxes.conf[0])
#     # label = names[int(results[0].boxes.cls[0])]
#     label = "Unknown Fish"

#     # Orientation check
#     x1, y1, x2, y2 = box.tolist()
#     width = x2 - x1
#     height = y2 - y1

#     if height > width:
#         logging.info(f"Fish is vertically placed. Width: {width:.2f}, Height: {height:.2f}")
#         raise ValueError("Incorrect orientation: Please place fish horizontally aligned with the camera.")

#     logging.info(f"Fish detected with correct orientation. Label: {label}, Confidence: {conf:.2f}")
#     return img, label, box


# def detect_fish_species(img):
#     """
#     Detects the fish species (multi-class classification) without orientation check.
#     Assumes a fish is already present in the image.
#     """
#     results = species_detector(img, conf=YOLO_CONFIDENCE)
#     names = species_detector.names

#     if results[0].boxes is None or len(results[0].boxes.xyxy) == 0:
#         logging.warning("No fish species detected in image.")
#         raise ValueError("No fish species detected.")

#     box = results[0].boxes.xyxy[0]
#     conf = float(results[0].boxes.conf[0])
#     label = names[int(results[0].boxes.cls[0])]
    
#     ####### Extra ###########

#     # # Orientation check
#     # x1, y1, x2, y2 = box.tolist()
#     # width = x2 - x1
#     # height = y2 - y1

#     # if height > width:
#     #     logging.info(f"Fish is vertically placed. Width: {width:.2f}, Height: {height:.2f}")
#     #     raise ValueError("Incorrect orientation: Please place fish horizontally aligned with the phone camera.")

#     # logging.info(f"Fish detected with correct orientation. Label: {label}, Confidence: {conf:.2f}, "
#     #              f"Box: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")
#     # print(f"Fish detected with correct orientation. Label: {label}, Confidence: {conf:.2f}, ")

#     ###########################

#     logging.info(f"Fish species detected. Label: {label}, Confidence: {conf:.2f}")
#     return img, label, box


# ===== Function to get YOLO format bbox =====
def get_top_bbox_yolo_format(image):
    """Run YOLO inference and return top-confidence bbox in YOLO format."""
    result = fish_detector(image, conf=0.7)[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    if len(boxes) == 0:
        return None, None  # No detection

    # Get top confidence bbox
    top_idx = confs.argmax()
    x1, y1, x2, y2 = boxes[top_idx]
    top_conf = float(confs[top_idx])

    # Convert to YOLO format (pixels)
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    return [x_center, y_center, width, height], top_conf



# def detect_fish(img):
#     results = model(img, conf=YOLO_CONFIDENCE)
#     names = model.names

#     if results[0].boxes is None or len(results[0].boxes.xyxy) == 0:
#         logging.warning("No fish detected in image.")
#         raise ValueError("No fish detected")

#     box = results[0].boxes.xyxy[0]
#     conf = float(results[0].boxes.conf[0])
#     label = names[int(results[0].boxes.cls[0])]

#     # Orientation check
#     x1, y1, x2, y2 = box.tolist()
#     width = x2 - x1
#     height = y2 - y1

#     if height > width:
#         logging.info(f"Fish is vertically placed. Width: {width:.2f}, Height: {height:.2f}")
#         raise ValueError("Incorrect orientation: Please place fish horizontally aligned with the phone camera.")

#     logging.info(f"Fish detected with correct orientation. Label: {label}, Confidence: {conf:.2f}, "
#                  f"Box: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")
#     print(f"Fish detected with correct orientation. Label: {label}, Confidence: {conf:.2f}, ")
#     return img, label, box


# def detect_fish(img):
#     results = model(img, conf=YOLO_CONFIDENCE)
#     names = model.names

#     if results[0].boxes is None or len(results[0].boxes.xyxy) == 0:
#         raise ValueError("No fish detected")
    
#     box = results[0].boxes.xyxy[0]
#     label = names[int(results[0].boxes.cls[0])]
#     return img, label, box
