import os
import cv2
from datetime import datetime
from app_backend.config import OUTPUT_DIR, REFERENCE_OBJECT_WIDTH_INCH
from app_backend.utils.helpers import find_pixels_per_inch, find_head_tail, find_pixels_per_inch_v2, find_head_tail_v2
import time

os.makedirs(OUTPUT_DIR, exist_ok=True)

def measure_fish(img, box, label):
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2]

    # Get marker coordinates and calculate pixels per 
    start_time = time.time()
    x, y, w, h = find_pixels_per_inch(img)
    print("find pixels: ", time.time() - start_time)
    pixels_per_inch = w / REFERENCE_OBJECT_WIDTH_INCH

    head, tail, max_dist_fish = find_head_tail(crop)
    print("find head to tail: ", time.time() - start_time)
    if max_dist_fish == 0:
        raise ValueError("Unable to find head and tail")

    length_in_inches = max_dist_fish / pixels_per_inch

    # Prepare annotation
    head_abs = (x1 + head[0], y1 + head[1])
    tail_abs = (x1 + tail[0], y1 + tail[1])
    annotated = img.copy()

    # Draw fish bounding box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw head, tail, and connecting line
    cv2.circle(annotated, head_abs, 6, (255, 0, 0), -1)
    cv2.circle(annotated, tail_abs, 6, (0, 0, 255), -1)
    cv2.line(annotated, head_abs, tail_abs, (0, 255, 255), 2)
    
    # Add label at top-left of fish bounding box
    label_y_offset = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(annotated, f"{label}", (x1, label_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Add fish length in inches
    mid_x, mid_y = int((head_abs[0] + tail_abs[0]) / 2), int((head_abs[1] + tail_abs[1]) / 2) - 20
    cv2.putText(annotated, f"{max_dist_fish:.2f}px == {length_in_inches:.2f} in", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw marker bounding box (used for pixel-per-inch reference)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Annotate marker width in both inches and pixels
    marker_text = f"{w}px == {REFERENCE_OBJECT_WIDTH_INCH:.2f} in"
    marker_text_pos = (x, y - 10 if y - 10 > 10 else y + h + 20)
    cv2.putText(annotated, marker_text, marker_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    # print("response back from measurement: ", time.time() - start_time)
    # # Save image
    # filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    # path = os.path.join(OUTPUT_DIR, filename)
    # cv2.imwrite(path, annotated)

    return length_in_inches, annotated, crop



def measure_fish_v2(img, box, label):
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2]
 
    # Get marker coordinates and calculate pixels per
    start_time = time.time()
    x, y, w, h = find_pixels_per_inch_v2(img)
    print("find pixels: ", time.time() - start_time)
    pixels_per_inch = w / REFERENCE_OBJECT_WIDTH_INCH
 
    head, tail, max_dist_fish = find_head_tail_v2(crop)
    print("find head to tail: ", time.time() - start_time)
    if max_dist_fish == 0:
        raise ValueError("Unable to find head and tail")
 
    length_in_inches = max_dist_fish / pixels_per_inch
 
    # Prepare annotation
    head_abs = (x1 + head[0], y1 + head[1])
    tail_abs = (x1 + tail[0], y1 + tail[1])
    annotated = img.copy()
 
    # Draw fish bounding box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
   
    # Draw head, tail, and connecting line
    cv2.circle(annotated, head_abs, 6, (255, 0, 0), -1)
    cv2.circle(annotated, tail_abs, 6, (0, 0, 255), -1)
    cv2.line(annotated, head_abs, tail_abs, (0, 255, 255), 2)
   
    # Add label at top-left of fish bounding box
    label_y_offset = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(annotated, f"{label}", (x1, label_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
   
    # Add fish length in inches
    mid_x, mid_y = int((head_abs[0] + tail_abs[0]) / 2), int((head_abs[1] + tail_abs[1]) / 2) - 20
    cv2.putText(annotated, f"{max_dist_fish:.2f}px == {length_in_inches:.2f} in", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 
    # Draw marker bounding box (used for pixel-per-inch reference)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
   
    # Annotate marker width in both inches and pixels
    marker_text = f"{w}px == {REFERENCE_OBJECT_WIDTH_INCH:.2f} in"
    marker_text_pos = (x, y - 10 if y - 10 > 10 else y + h + 20)
    cv2.putText(annotated, marker_text, marker_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    # print("response back from measurement: ", time.time() - start_time)
    # # Save image
    # filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    # path = os.path.join(OUTPUT_DIR, filename)
    # cv2.imwrite(path, annotated)
 
    return length_in_inches, annotated, crop
