# validation set을 기준으로 모델 accuracy를 확인합니다.
import json
import os.path as osp
import time
from datetime import timedelta
from argparse import ArgumentParser

# from tqdm import tqdm

import torch
from torch import cuda
from model import EAST

from detect import *
from deteval import *
from utils import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/ephemeral/home/level2-cv-datacentric-cv-05/data/medical",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--model_pth",
        type=str,
        default="/data/ephemeral/home/level2-cv-datacentric-cv-05/code/pths/exp2/best.pth",
    )
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


def val(
    data_dir,
    fold,
    model_pth,
    input_size,
    batch_size,
):
    with open(
        osp.join(data_dir, f"ufo/split/val_fold_{fold}.json"), "r", encoding="utf-8-sig"
    ) as f:
        valid_anno = json.load(f)
    valid_images = [key for key in valid_anno["images"]]

    print("[INFO] Dataset loaded")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_pth, map_location=device))
    print("[INFO] Model loaded.")

    print("[INFO] Start evaluation.")
    model.eval()
    start = time.time()
    print(f"Calculating validation results for {model_pth}...")
    with torch.no_grad():
        gt_bboxes_dict = get_gt_bboxes(
            data_dir,
            ufo_dir=f"/ufo/split/val_fold_{fold}.json",
            valid_images=valid_images,
        )
        pred_bboxes_dict = get_pred_bboxes(
            model, data_dir, valid_images, input_size, batch_size, fold
        )

        result = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)
        total_result = result["total"]
        precision, recall, hmean = (
            total_result["precision"],
            total_result["recall"],
            total_result["hmean"],
        )
        f1_score = hmean
        print("F1 Score : {:.4f}".format(f1_score))
        print("Precision : {:.4f}".format(precision))
        print("Recall : {:.4f}".format(recall))
    print("Elapsed time: {}".format(timedelta(seconds=time.time() - start)))


def main(args):
    val(**args.__dict__)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    print(args)
    main(args)
