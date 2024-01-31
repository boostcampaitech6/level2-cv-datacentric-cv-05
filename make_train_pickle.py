import os.path as osp
import json
from tqdm import tqdm
import pickle
from PIL import Image

import numpy as np
import cv2
import torch

from data.dataset import filter_vertices, resize_img, adjust_height, rotate_img, crop_img
from data.preprocess import generate_score_geo_maps, generate_roi_mask

def make_pickle(root_dir, json_name, out_path,
                image_size=2048, 
                crop_size=1024, 
                ignore_tags=["masked", "excluded-region", "maintable", "stamp"], 
                ignore_under_threshold=10, 
                drop_under_threshold=1,
                num_crop=1):
    
    if crop_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")
    
    json_dir = osp.join(root_dir, "ufo/{}".format(json_name))
    with open(json_dir, "r", encoding='utf-8-sig') as f:
        anno = json.load(f)

    image_fnames = sorted(anno["images"].keys())
    image_dir = osp.join(root_dir, "img", "train")
    
    total = dict(images=[], scores=[], geos=[], rois=[])
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
        for idx in range(num_crop):
            crop_image, crop_vertices = crop_img(image, vertices, labels, crop_size)
            
            if crop_image.mode != 'RGB':
                crop_image = crop_image.convert('RGB')
            
            word_bboxes = np.reshape(crop_vertices, (-1, 4, 2))
            crop_image = np.array(crop_image)
            roi_mask = generate_roi_mask(crop_image, crop_vertices, labels)
            
            score_map, geo_map = generate_score_geo_maps(crop_image, word_bboxes, map_scale=0.5)
            
            mask_size = int(crop_image.shape[0] * 0.5), int(crop_image.shape[1] * 0.5)
            roi_mask = cv2.resize(roi_mask, dsize=mask_size)
            
            if roi_mask.ndim == 2:
                roi_mask = np.expand_dims(roi_mask, axis=2)
            
            score_map = torch.Tensor(score_map).permute(2, 0, 1)
            geo_map = torch.Tensor(geo_map).permute(2, 0, 1)
            roi_mask = torch.Tensor(roi_mask).permute(2, 0, 1)
            
            total["images"].append(np.array(crop_image))
            total["scores"].append(score_map)
            total["geos"].append(geo_map)
            total["rois"].append(roi_mask)
    
    with open(osp.join(root_dir, "ufo", out_path), "wb") as fw:
        pickle.dump(total, fw)
            
if __name__ == '__main__':
    make_pickle(
        root_dir = '../data/medical',
        json_name= 'split/train_fold_0.json',
        out_path="split/train_fold_0_c4.pkl",
        image_size=2048,
        crop_size=1024,
        ignore_tags=["masked", "excluded-region", "maintable", "stamp"],
        ignore_under_threshold=10,
        drop_under_threshold=1,
        num_crop=4
    )
