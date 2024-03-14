import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def IOU(outputs, labels):
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    matrix = confusion_matrix(y_true=np.array(outputs).flatten(), y_pred=np.array(labels).flatten())
    intersection = np.diag(matrix)

    ground_truth_set = matrix.sum(axis=1)

    predicted_set = matrix.sum(axis=0)

    union = ground_truth_set + predicted_set - intersection + 1e-7
    IoU = intersection / union.astype(np.float32)

    #union_dice = ground_truth_set + predicted_set + 1e-7
    #DICE = 2 * intersection / union_dice.astype(np.float32)

    #return np.mean(IoU)#, np.mean(DICE)
    return np.mean(IoU)
def compute_acc(outputs, labels):
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    matrix = confusion_matrix(y_true=np.array(outputs).flatten(), y_pred=np.array(labels).flatten())
    acc = np.diag(matrix).sum() / matrix.sum()
    return np.mean(acc)

