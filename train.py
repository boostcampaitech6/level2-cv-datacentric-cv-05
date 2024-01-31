import os
import os.path as osp
import time
from importlib import import_module

import numpy as np
import cv2
import json

from datetime import timedelta

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.east_dataset import EASTDataset
from data.dataset import TestDataset
from model.model import EAST
from utils.util import seed_everything, setup_paths
from utils.argparsers import Parser
from utils.logger import Logger, WeightAndBiasLogger
from utils.create import *
from utils.deteval import calc_deteval_metrics
from utils.plot import visualize_bbox

txt_logger = None
wb_logger = None
device = None
VAL_START_IDX = 50

def validate(model: torch.nn.Module, val_loader: DataLoader) -> dict:
    with torch.no_grad():
        model.eval()
        
        image_fnames = []
        gt_bboxes = []
        pred_bboxes = []
        pred_bbox = []
        print("Calculate validation set.....")
        
        for val_batch in val_loader:
            images, image_fname, gt_bbox, orig_sizes = val_batch
            images = images.to(device)
            score_maps, geo_maps = model(images)
            pred_bbox = TestDataset.restore_bbox(2048, score_maps, geo_maps, orig_sizes)
            
            image_fnames.append(image_fname[0])
            pred_bboxes.append(pred_bbox[0])
            gt_bboxes.append(gt_bbox.cpu().numpy()[0])
            
        
        gt_bboxes = {image_fname : gt_bbox for image_fname, gt_bbox in zip(image_fnames, gt_bboxes)}
        pred_bboxes = {image_fname : pred_bbox for image_fname, pred_bbox in zip(image_fnames, pred_bboxes)}
            
        result = calc_deteval_metrics(pred_bboxes, gt_bboxes, verbose=False)
        precision, recall, f1score = result['total']['precision'], result['total']['recall'], result['total']['hmean']
        #per_sample_log = result['per_sample']
         
        validation_desc = \
            "Precision: {:3.4f}, Recall: {:3.4f}, F1 Score: {:3.4f}".\
            format(precision, recall, f1score)
        
        print(validation_desc)
        
        # for image_fname, value in per_sample_log.items():
        #     validation_desc += "\n[{}] Precision: {:3.4f}, Recall: {:3.4f}, F1 Score: {:3.4f}, Log: {}\n".\
        #         format(image_fname, value["precision"], value["recall"], value["hmean"], value["evaluation_log"])

        txt_logger.update_string(validation_desc)

        return {
                "Val Recall": recall,
                "Val Precision": precision,
                "Val F1_Score": f1score,
            }, image_fname[0], pred_bbox[0] if len(pred_bbox) else None, gt_bbox[0]
            
def train(args):
    seed_everything(args.seed)
    save_path, weight_path = setup_paths(args.save_dir, args.exp_name)
    
    use_cuda = torch.cuda.is_available()
    global device
    device = torch.device("cuda" if use_cuda else "cpu")
    
    try:
        dataset_module = getattr(import_module("data.east_dataset"), args.dataset + "WithPickle" if args.with_pickle else args.dataset)
        train_set = dataset_module(args.data_dir, args.pickle_path)
    except Exception as e:
        raise Exception("Be sure to use the East dataset pickle dataset")
        
    val_set = TestDataset(args.data_dir, args.val_json_path, args.image_size, args.ignore_tags)
        
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)
    
    model = EAST(pretrained=True)
    model.to(device)
    
    optimizer = create_optimizer(args.optimizer, model.parameters(), float(args.lr), 5e-4)
    scheduler = create_scheduler(args.scheduler, optimizer, args.max_epochs)
    
    with open(osp.join(save_path, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    global txt_logger
    global wb_logger
    
    txt_logger = Logger(save_path)
    wb_logger = WeightAndBiasLogger(args, save_path.split("/")[-1], args.project_name)
    txt_logger.update_string(str(args))
    
    best_f1_score = 0.
    num_batches = len(train_loader)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(args.max_epochs):
        model.train()
        epoch_start = time.time()
        train_desc_format = "Epoch[{:03d}/{:03d}] - Cls Loss: {:3.4f}, Angle Loss.: {:3.4f}, IoU Loss.: {:3.4f}"
        train_process_bar = tqdm(train_loader, desc=train_desc_format.format(epoch, args.max_epochs, 0., 0., 0.), mininterval=0.01)
        train_loss = 0.
        
        for train_batch in train_process_bar:
            img, gt_score_map, gt_geo_map, roi_mask = train_batch
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                    loss, extra_info = model.train_step(
                        img, gt_score_map, gt_geo_map, roi_mask
                    )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_val = loss.item()
            train_loss += loss_val
            train_desc = train_desc_format.format(epoch, args.max_epochs, extra_info['cls_loss'],
                                                  extra_info['angle_loss'], extra_info['iou_loss'])
            train_process_bar.set_description(train_desc)
            
        scheduler.step()
        
        train_desc = '[Train] Total loss: {:.4f} | Elapsed time: {}'.format(
            train_loss / num_batches, timedelta(seconds=time.time() - epoch_start))
        print(train_desc)
        txt_logger.update_string(train_desc)
        train_process_bar.close()
        
        if epoch >= VAL_START_IDX:
            metric, vis_img_name, pred_bboxes, gt_bboxes = validate(model, val_loader)
            image_fpath = osp.join(args.data_dir, "img", "train", vis_img_name)
        
            img = cv2.imread(image_fpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt_img = visualize_bbox(img.copy(), gt_bboxes)
            pred_img = visualize_bbox(img.copy(), pred_bboxes)
            
            vis_img = np.hstack([pred_img, gt_img])
            
            metric["Image"] = wb_logger.make_image_with_label(vis_img)
            metric["Train Loss"] = train_loss / num_batches
            wb_logger.log(metric)
            
            if metric["Val F1_Score"] > best_f1_score:
                torch.save(model.state_dict(), os.path.join(weight_path, 'best.pt'))
                best_f1_score = metric["Val F1_Score"]
        
        torch.save(model.state_dict(), os.path.join(weight_path, 'last.pt'))
    
    best_weight = torch.load(os.path.join(weight_path, 'best.pt'))
    
    model.load_state_dict(best_weight)
    
    _, vis_img_name, pred_bboxes, gt_bboxes = validate(model, val_loader)
    
    image_fpath = osp.join(args.data_dir, "img", "train", vis_img_name)
    img = cv2.imread(image_fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt_img = visualize_bbox(img.copy(), gt_bboxes)
    pred_img = visualize_bbox(img.copy(), pred_bboxes)
    vis_img = np.hstack([pred_img, gt_img])
    cv2.imwrite(osp.join(save_path, "result_img.jpg"), vis_img)
    
if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise AttributeError("CUDA is not ready")
        
    start_time = time.time()
    p = Parser()
    p.create_parser()
    
    import yaml
    pargs = p.parser.parse_args()
    try:
        with open(pargs.config, 'r') as fp:
            load_args = yaml.load(fp, Loader=yaml.FullLoader)
        key = vars(pargs).keys()
        for k in load_args.keys():
            if k not in key:
                print("Wrong argument: ", k)
                assert(k in key)
            p.parser.set_defaults(**load_args)
    except FileNotFoundError as error:
        raise error("Invalid filename. Check your file path or name.")
    
    args = p.parser.parse_args() 
    
    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')
    
    p.print_args(args)
    os.makedirs(args.save_dir, exist_ok=True)
    
    train(args)
        
    print("--- %s seconds ---" % (time.time() - start_time))