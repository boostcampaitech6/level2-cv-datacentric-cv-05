import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2

def visualize_bbox(image: np.ndarray, bboxes: np.ndarray, color: tuple=(255, 30, 30)) -> np.ndarray:
    bboxes = np.array(bboxes, dtype=np.int32)
    for bbox in bboxes:
        image = cv2.polylines(image, [bbox], True, color, 2)
    return image

def save_confusion_matrix(target: np.ndarray, prediction: np.ndarray, label_num: int, save_path: str):
    """
    주어진 대상 및 예측 레이블을 기반으로 혼동 행렬을 생성하고 저장한다.

    Args:
        target (np.ndarray): 실제 레이블 배열.
        prediction (np.ndarray): 예측 레이블 배열.
        label_num (int): 레이블의 개수.
        save_path (str): 혼동 행렬을 저장할 경로.

    Returns:
        None
    """
    matrix = confusion_matrix(target, prediction)
    plt.figure(figsize=(12, 9)) 
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.jpg'))