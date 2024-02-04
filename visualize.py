
import numpy as np

import json
import os
import cv2
import glob

from utils.plot import visualize_bbox

ignore_tags = ['masked', 'excluded-region', 'maintable', 'stamp']
img_dir = "../data/medical/img/train"
json_path = "../data/medical/ufo/split/train_fold_0.json"
img_list = glob.glob(os.path.join(img_dir, '*.jpg'))
img_save_path = "./visualization/"
os.makedirs(img_save_path, exist_ok=True)

with open(json_path, 'r', encoding="utf-8-sig") as json_reader:
    dataset = json.load(json_reader)

for img_path in img_list:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Image Processing
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Histogram Equal.
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # img = clahe.apply(img)
    
    # Binarization
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 5)
    
    if img_path.split('/')[-1] not in dataset["images"]:
        continue
    bboxes = []
    for words in dataset['images'][img_path.split('/')[-1]]['words'].values():
        if any(elem for elem in words['tags'] if elem in ignore_tags) or len(words["points"]) != 4:
            continue
        else:
            bboxes.append(words['points'])
            
    visualized_img = visualize_bbox(img.copy(), bboxes, color=(0, 0, 0))
    cv2.imwrite(os.path.join(img_save_path, img_path.split('/')[-1]), visualized_img)
