import os

import pickle
import numpy as np
import torch
import cv2

from torch.utils.data import Dataset

from data.preprocess import generate_score_geo_maps
from data.augmentation import TrainTransform

class EASTDataset(Dataset):
    def __init__(self, dataset, map_scale=0.5, to_tensor=True):
        self.dataset = dataset
        self.map_scale = map_scale
        self.to_tensor = to_tensor

    def __getitem__(self, idx):
        image, word_bboxes, roi_mask = self.dataset[idx]
        score_map, geo_map = generate_score_geo_maps(image, word_bboxes, map_scale=self.map_scale)

        mask_size = int(image.shape[0] * self.map_scale), int(image.shape[1] * self.map_scale)
        roi_mask = cv2.resize(roi_mask, dsize=mask_size)
        if roi_mask.ndim == 2:
            roi_mask = np.expand_dims(roi_mask, axis=2)

        if self.to_tensor:
            image = torch.Tensor(image).permute(2, 0, 1)
            score_map = torch.Tensor(score_map).permute(2, 0, 1)
            geo_map = torch.Tensor(geo_map).permute(2, 0, 1)
            roi_mask = torch.Tensor(roi_mask).permute(2, 0, 1)

        return image, score_map, geo_map, roi_mask

    def __len__(self):
        return len(self.dataset)

class EASTDatasetWithPickle(Dataset):
    def __init__(self, data_dir, pickle_path, use_transform=True):
        with open(os.path.join(data_dir, 'ufo', pickle_path), "rb") as fr:
            total = pickle.load(fr)
            
        self.images = total["images"]
        self.score_maps = total["scores"]
        self.geo_maps = total["geos"]
        self.roi_masks = total["rois"]
        
        self.transform = TrainTransform() if use_transform else None
        
    def __getitem__(self, idx):
        image, score_map, geo_map, roi_mask = \
            self.images[idx], self.score_maps[idx], self.geo_maps[idx], self.roi_masks[idx]
            
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, score_map, geo_map, roi_mask
    
    def __len__(self):
        return len(self.images)