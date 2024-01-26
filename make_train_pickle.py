import os.path as osp
import json
from tqdm import tqdm
import pickle
from PIL import Image

import numpy as np

from data.dataset import filter_vertices, resize_img, adjust_height, rotate_img, crop_img

def make_pickle(root_dir, json_name, 
                image_size=2048, 
                crop_size=1024, 
                ignore_tags=["masked", "excluded-region", "maintable", "stamp"], 
                ignore_under_threshold=10, 
                drop_under_threshold=1):
    
    if crop_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")
    
    json_dir = osp.join(root_dir, "ufo/{}".format(json_name))
    with open(json_dir, "r", encoding='utf-8-sig') as f:
        anno = json.load(f)

    image_fnames = sorted(anno["images"].keys())
    image_dir = osp.join(root_dir, "img", "train")
    
    total = dict(images = [], vertices = [], labels = [])
    for idx in tqdm(range(len(image_fnames))):
        image_fname = image_fnames[idx]
        image_fpath = osp.join(image_dir, image_fname)

        vertices, labels = [], []
        for word_info in anno["images"][image_fname]["words"].values():
            word_tags = word_info["tags"]
            ignore_sample = any(
                elem for elem in word_tags if elem in ignore_tags
            )
            num_pts = np.array(word_info["points"]).shape[0]

            if ignore_sample or num_pts > 4:
                continue

            vertices.append(np.array(word_info["points"]).flatten())
            labels.append(int(not word_info["illegibility"]))
        
        
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)
        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=ignore_under_threshold,
            drop_under=drop_under_threshold,
        )
        
        image = Image.open(image_fpath)

        image, vertices = resize_img(image, vertices, image_size)
        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices)
        image, vertices = crop_img(image, vertices, labels, crop_size)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        total["images"].append(np.array(image))
        total["vertices"].append(vertices)
        total["labels"].append(labels)
    
    with open(osp.splitext(json_dir)[0] + ".pkl", "wb") as fw:
        pickle.dump(total, fw)
            
if __name__ == '__main__':
    make_pickle(
        root_dir = '../data/medical',
        json_name= 'split/train_fold_0.json',
        image_size=2048,
        crop_size=1024,
        ignore_tags=["masked", "excluded-region", "maintable", "stamp"],
        ignore_under_threshold=10,
        drop_under_threshold=1,
    )
