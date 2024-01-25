import os.path as osp
from detect import *
from tqdm import tqdm
import cv2
import json


def get_gt_bboxes(root_dir, ufo_dir, valid_images):
    """
    ground truth bboxes dictionary를 구하는 함수 입니다.
    """

    gt_bboxes = dict()
    ufo_file_root = root_dir + ufo_dir

    with open(ufo_file_root, "r", encoding="utf-8-sig") as f:
        ufo_file = json.load(f)

    ufo_file_images = ufo_file["images"]
    for valid_image in tqdm(valid_images):
        gt_bboxes[valid_image] = []
        for idx in ufo_file_images[valid_image]["words"].keys():
            gt_bboxes[valid_image].append(
                ufo_file_images[valid_image]["words"][idx]["points"]
            )
    return gt_bboxes


def get_pred_bboxes(model, data_dir, valid_images, input_size, batch_size, fold):
    """
    모델을 통해 예측된 bboxes dictionary를 구하는 함수 입니다.
    """

    image_fnames, by_sample_bboxes = [], []

    images = []
    for valid_image in tqdm(valid_images):
        image_fpath = osp.join(data_dir, f"img/train/{valid_image}")
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    pred_bboxes = dict()
    for idx in range(len(image_fnames)):
        image_fname = image_fnames[idx]
        sample_bboxes = by_sample_bboxes[idx]
        pred_bboxes[image_fname] = sample_bboxes

    return pred_bboxes
