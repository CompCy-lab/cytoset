import os
import sys
import math
import time
import shutil
import pickle
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

from model import CytoSetModel
from data import CytoDatasetFromFCS
from utils import (
    EarlyStopping, load_fcs_dataset, train_valid_split, combine_samples, down_rsampling
)


def test_model(test_samples, test_phenotypes, model, device):
    model.eval()
    correct_num, total_num = 0, 0
    y_pred, y_true = [], []
    losses = []

    for sample, label in zip(test_samples, test_phenotypes):
        with torch.no_grad():
            sample = torch.from_numpy(sample).to(device)
            true_label = torch.tensor([label], dtype=torch.float32).to(device)
            prob = model(sample)
            loss = F.binary_cross_entropy(prob, true_label, reduction='mean')
            pred_label = torch.ge(prob, 0.5)

        losses.append(loss.item())
        v = (pred_label == label).sum()

        y_true.append(label)
        y_pred.append(prob.detach().cpu().numpy())

        correct_num += v.item()
        total_num += 1

    acc = float(correct_num) / total_num

    y_true, y_pred = np.array(y_true), np.hstack(y_pred)
    auc = roc_auc_score(y_true, y_pred)
    eval_loss = np.mean(np.array(losses))

    return eval_loss, acc, auc


def test(args):
    # load the pretrained model
    model = CytoSetModel.from_pretrained(model_path=args.model, config_path=args.config_file, cache_dir='./cache')
    model = model.to(args.device)

    # read the test dataset
    with open(args.test_pkl, 'rb') as f:
        test_data = pickle.load(f)
    test_samples, test_phenotypes = test_data['test_sample'], test_data['test_phenotype']

    # test model
    _, test_acc, test_auc = test_model(test_samples, test_phenotypes, model, args.device)
    print("Testing Acc: {:.3f}, Testing Auc: {:.3f}".format(test_acc, test_auc))

    # Finished the testing process
    print("Testing finished, Done")


def main():
    parser = argparse.ArgumentParser("Cytometry Set Model")

    # data
    parser.add_argument('--test_pkl', type=str, help='path or url to the test pickled file')

    # model
    parser.add_argument('--model', type=str, help='the path to the pretrained model')
    parser.add_argument('--config_file', type=str, help='the path to model configuration')
    parser.add_argument('--device', type=str, default='cuda', help='device to use')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'

    # test the model
    test(args)


if __name__ == "__main__":
    main()
