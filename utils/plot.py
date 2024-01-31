import numpy as np
import cv2

def visualize_bbox(image: np.ndarray, bboxes: np.ndarray, color: tuple=(255, 30, 30)) -> np.ndarray:
    try:
        bboxes = np.array(bboxes, dtype=np.int32)
    except Exception as e:
        exit()
        
    for bbox in bboxes:
        image = cv2.polylines(image, [bbox], True, color, 2)
    return image