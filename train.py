import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
from detect import *
from deteval import *
from utils import *
import wandb
import random


def seed_everything(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/ephemeral/home/level2-cv-datacentric-cv-05/data/medical",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/ephemeral/home/level2-cv-datacentric-cv-05/code/pths",
    )
    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epoch", type=int, default=150)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--val_start", type=int, default=30)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument(
        "--ignore_tags",
        type=list,
        default=["masked", "excluded-region", "maintable", "stamp"],
    )

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


def train(
    name,
    seed,
    data_dir,
    fold,
    model_dir,
    device,
    image_size,
    input_size,
    num_workers,
    batch_size,
    learning_rate,
    max_epoch,
    save_interval,
    val_start,
    val_interval,
    ignore_tags,
):
    # wandb 초기 설정
    wandb.init(
        name=name,  # exp_name
        project="ocr",
        config=args,
    )
    model_save_dir = osp.join(model_dir, name)
    os.makedirs(model_save_dir, exist_ok=True)

    dataset = SceneTextDataset(
        data_dir,
        split="train",
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        fold=fold,
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    with open(
        osp.join(data_dir, f"ufo/split/val_fold_{fold}.json"), "r", encoding="utf-8-sig"
    ) as f:
        valid_anno = json.load(f)
    valid_images = [key for key in valid_anno["images"]]

    print("[INFO] Dataset loaded")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[max_epoch // 2], gamma=0.1
    )
    # scaler = torch.cuda.amp.GradScaler() # TODO: amp
    print("[INFO] Model loaded.")

    print("[INFO] Start training.")
    best_f1_score = 0
    for epoch in range(max_epoch):
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(
                train_loader
            ):
                pbar.set_description("[Epoch {}]".format(epoch + 1))

                img = img.to(device)
                gt_score_map = gt_score_map.to(device)
                gt_geo_map = gt_geo_map.to(device)
                roi_mask = roi_mask.to(device)

                # loss, extra_info = model.train_step(
                #     img, gt_score_map, gt_geo_map, roi_mask
                # )
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                with torch.cuda.amp.autocast():
                    loss, extra_info = model.train_step(
                        img, gt_score_map, gt_geo_map, roi_mask
                    )
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    "train_cls loss": extra_info["cls_loss"],
                    "train_angle loss": extra_info["angle_loss"],
                    "train_IoU loss": extra_info["iou_loss"],
                }
                wandb.log(val_dict, step=epoch)
                pbar.set_postfix(val_dict)

        scheduler.step()

        # validation
        if (epoch + 1) % val_interval == 0 and epoch >= val_start:
            model.eval()
            start = time.time()
            print(f"Calculating validation results for epoch {epoch+1}...")
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
                # f1_score = 2*precision*recall/(precision+recall)
                f1_score = hmean
                print("F1 Score : {:.4f}".format(f1_score))

                wandb.log({"valid_precision": precision}, step=epoch)
                wandb.log({"valid_recall": recall}, step=epoch)
                wandb.log({"valid_f1_score": f1_score}, step=epoch)

                if best_f1_score < f1_score:
                    print(
                        f"New best model for f1 score : {f1_score}! saving the best model.."
                    )
                    bestpt_fpath = osp.join(model_save_dir, "best.pth")
                    torch.save(model.state_dict(), bestpt_fpath)
                    best_f1_score = f1_score

            print("Elapsed time: {}".format(timedelta(seconds=time.time() - start)))

        print(
            "Mean loss: {:.4f} | Elapsed time: {}".format(
                epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)
            )
        )

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_save_dir, "latest.pth")
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    train(**args.__dict__)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    seed_everything(args.seed)
    print(args)
    main(args)
