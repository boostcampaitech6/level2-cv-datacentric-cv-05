import os.path as osp
import json

import matplotlib.pyplot as plt
import numpy as np
import cv2

def draw_bbox(image, bbox, color=(0, 0, 255), thickness=1, thickness_sub=None, double_lined=False,
              write_point_numbers=False):
    """이미지에 하나의 bounding box를 그려넣는 함수
    """
    thickness_sub = thickness_sub or thickness * 3
    basis = max(image.shape[:2])
    fontsize = basis / 4000
    x_offset, y_offset = int(fontsize * 12), int(fontsize * 10)
    color_sub = (255 - color[0], 255 - color[1], 255 - color[2])

    points = [(int(np.rint(p[0])), int(np.rint(p[1]))) for p in bbox]

    for idx in range(len(points)):
        if double_lined:
            cv2.line(image, points[idx], points[(idx + 1) % len(points)], color_sub,
                     thickness=thickness_sub)
        cv2.line(image, points[idx], points[(idx + 1) % len(points)], color, thickness=thickness)


def draw_bboxes(image, bboxes, color=(0, 0, 255), thickness=1, thickness_sub=None,
                double_lined=False, write_point_numbers=False):
    """이미지에 다수의 bounding box들을 그려넣는 함수
    """
    for bbox in bboxes:
        draw_bbox(image, bbox, color=color, thickness=thickness, thickness_sub=thickness_sub,
                  double_lined=double_lined, write_point_numbers=write_point_numbers)
        

if __name__ == "__main__":
    DATASET_DIR = '../data/medical/'
    
    ufo_fpath = osp.join(DATASET_DIR, 'ufo/train.json')
    with open(ufo_fpath, 'r') as f:
        ufo_anno = json.load(f)
    
    sample_ids = sorted(ufo_anno['images'])
    
    SAMPLE_IDX = 79

    sample_id = sample_ids[SAMPLE_IDX]  # `sample_id`가 곧 이미지 파일명
    image_fpath = osp.join(DATASET_DIR, 'img/train', sample_id)
    print(image_fpath)
    image = cv2.imread(image_fpath)

    print('Image:\t{}'.format(image.shape))

    ignore_tags = ['masked', 'excluded-region', 'maintable', 'stamp']

    bboxes, labels = [], []
    for word_info in ufo_anno['images'][sample_id]['words'].values():
        word_tags = word_info['tags']
        ignore_sample = any(elem for elem in word_tags if elem in ignore_tags)
        if ignore_sample:
            continue
        
        if len(word_info['points']) > 4:
            continue
            
        bboxes.append(np.array(word_info['points']))
        labels.append(int(not word_info['illegibility']))
    bboxes, labels = np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.float32)

    print('Bounding boxes:\t{}'.format(bboxes.shape))
    print('Labels:\t{}'.format(labels.shape))
    
    vis = image.copy()
    draw_bboxes(vis, bboxes, double_lined=True, thickness=2, thickness_sub=5, write_point_numbers=True)

    plt.figure(figsize=(16, 16))
    plt.imshow(vis)