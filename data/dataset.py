import os.path as osp
import json
import pickle

import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset

from data.preprocess import *
from data.augmentation import TestTransform
from utils.detect import get_bboxes

class SceneTextDataset(Dataset):
    def __init__(self, root_dir,
                 json_path,
                 image_size=2048,
                 crop_size=1024,
                 ignore_tags=[],
                 ignore_under_threshold=10,
                 drop_under_threshold=1,
                 color_jitter=True,
                 normalize=True):
        with open(osp.join(root_dir, 'ufo/{}'.format(json_path)), 'r', encoding='utf-8-sig') as f:
            anno = json.load(f)

        self.anno = anno
        self.image_fnames = sorted(anno['images'].keys())
        self.image_dir = osp.join(root_dir, 'img', 'train')

        self.image_size, self.crop_size = image_size, crop_size
        self.color_jitter, self.normalize = color_jitter, normalize

        self.ignore_tags = ignore_tags

        self.drop_under_threshold = drop_under_threshold
        self.ignore_under_threshold = ignore_under_threshold

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_fpath = osp.join(self.image_dir, image_fname)

        vertices, labels = [], []
        for word_info in self.anno['images'][image_fname]['words'].values():
            word_tags = word_info['tags']

            ignore_sample = any(elem for elem in word_tags if elem in self.ignore_tags)
            num_pts = np.array(word_info['points']).shape[0]

            if ignore_sample or num_pts > 4:
                continue

            vertices.append(np.array(word_info['points']).flatten())
            labels.append(int(not word_info['illegibility']))
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )

        image = Image.open(image_fpath)

        image, vertices = resize_img(image, vertices, self.image_size)
        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices)
        image, vertices = crop_img(image, vertices, labels, self.crop_size)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        funcs = []
        if self.color_jitter:
            funcs.append(A.ColorJitter(0.5, 0.5, 0.5, 0.25))
        if self.normalize:
            funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = A.Compose(funcs)

        image = transform(image=image)['image']
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)
        
        return image, word_bboxes, roi_mask

class TestDataset(Dataset):
    def __init__(self, root_dir: str, json_path: str, input_size: int=2048, ignore_tags: list=[], 
                 ignore_under_threshold=10, drop_under_threshold=1):
        with open(osp.join(root_dir, 'ufo/{}'.format(json_path)), 'r', encoding='utf-8-sig') as f:
            anno = json.load(f)
        self.anno = anno
        self.image_fnames = sorted(anno['images'].keys())
        self.image_dir = osp.join(root_dir, 'img', 'train')
        self.input_size = input_size
        
        self.transform = TestTransform(input_size)
        self.ignore_tags = ignore_tags

        self.drop_under_threshold = drop_under_threshold
        self.ignore_under_threshold = ignore_under_threshold
        
    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_fpath = osp.join(self.image_dir, image_fname)

        vertices, labels = [], []
        for word_info in self.anno['images'][image_fname]['words'].values():
            word_tags = word_info['tags']

            ignore_sample = any(elem for elem in word_tags if elem in self.ignore_tags)
            num_pts = np.array(word_info['points']).shape[0]

            if ignore_sample or num_pts > 4:
                continue

            vertices.append(np.array(word_info['points']).flatten())
            labels.append(int(not word_info['illegibility']))
        
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )

        image = Image.open(image_fpath)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image = np.array(image)
        orig_size = image.shape[:2]
        image = self.transform(image=image)['image']
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        
        return image, image_fname, word_bboxes, orig_size
    
    def __len__(self):
        return len(self.image_fnames)
    
    @staticmethod
    def restore_bbox(input_size, score_maps, geo_maps, orig_sizes):
        score_maps, geo_maps = score_maps.cpu().numpy(), geo_maps.cpu().numpy()

        by_sample_bboxes = []
        for score_map, geo_map, orig_size in zip(score_maps, geo_maps, [orig_sizes]):
            map_margin = int(abs(orig_size[0] - orig_size[1]) * 0.5 * input_size / max(orig_size))
            if orig_size[0] == orig_size[1]:
                score_map, geo_map = score_map, geo_map        
            elif orig_size[0] > orig_size[1]:
                score_map, geo_map = score_map[:, :, :-map_margin], geo_map[:, :, :-map_margin]
            else:
                score_map, geo_map = score_map[:, :-map_margin, :], geo_map[:, :-map_margin, :]

            bboxes = get_bboxes(score_map, geo_map)
            if bboxes is None:
                bboxes = np.zeros((0, 4, 2), dtype=np.float32)
            else:
                bboxes = bboxes[:, :8].reshape(-1, 4, 2)
                bboxes *= int(max(orig_size)) / input_size

            by_sample_bboxes.append(bboxes)
            
        return by_sample_bboxes
    
class SceneTextDatasetWithPickle(Dataset):
    def __init__(self, data_dir, pickle_path,
                 color_jitter=True,
                 normalize=True):
        with open(osp.join(data_dir, 'ufo', pickle_path), "rb") as fr:
            total = pickle.load(fr)
            
        self.images = total["images"]
        self.vertices = total["vertices"]
        self.labels = total["labels"]
        
        self.color_jitter, self.normalize = color_jitter, normalize
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        vertices = self.vertices[idx]
        labels = self.labels[idx]

        funcs = []
        if self.color_jitter:
            funcs.append(A.ColorJitter(0.5, 0.5, 0.5, 0.25))
        if self.normalize:
            funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = A.Compose(funcs)

        image = transform(image=image)["image"]
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask
    
class TestDatasetWithPickle(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, "rb") as fr:
            total = pickle.load(fr)
            
        self.images = total["images"]
        self.image_fnames = total["image_fnames"]
        self.labels = total["word_bboxes"]
        self.orig_sizes = total["orig_sizes"]
    
    def __getitem__(self, idx):
        return self.images[idx], self.image_fnames[idx], self.labels[idx], self.orig_sizes[idx]