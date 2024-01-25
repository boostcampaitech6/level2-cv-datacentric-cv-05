import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os
import json

json_path = (
    "/data/ephemeral/home/level2-cv-datacentric-cv-05/data/medical/ufo/train.json"
)
save_path = "/data/ephemeral/home/level2-cv-datacentric-cv-05/data/medical/ufo/split"
os.makedirs(save_path, exist_ok=True)
random_seed = 207

with open(json_path, "r") as f:
    json_data = json.load(f)

json_images = [key for key in json_data["images"]]
kf = KFold(n_splits=2, random_state=207, shuffle=True)

json_images = np.array(json_images)
cnt = 0
for train_idx, test_idx in kf.split(range(100)):
    train_json = json_data.copy()
    val_json = json_data.copy()

    train_json["images"] = {
        image_name: json_data["images"][image_name]
        for image_name in json_images[train_idx]
    }
    val_json["images"] = {
        image_name: json_data["images"][image_name]
        for image_name in json_images[test_idx]
    }

    train_filename = os.path.join(save_path, "train_fold_{}.json".format(cnt))
    with open(train_filename, "w+", encoding="utf-8-sig") as f:
        json.dump(train_json, f, indent=4, ensure_ascii=False)

    val_filename = os.path.join(save_path, "val_fold_{}.json".format(cnt))
    with open(val_filename, "w+", encoding="utf-8-sig") as f:
        json.dump(val_json, f, indent=4, ensure_ascii=False)
    cnt += 1
