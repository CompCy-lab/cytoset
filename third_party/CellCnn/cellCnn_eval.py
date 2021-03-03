# coding: utf-8
import os
import sys
import errno
import glob
import argparse
import pickle
import numpy as np
import pandas as pd

import cellCnn
from cellCnn.utils import loadFCS, ftrans, mkdir_p, get_items
from cellCnn.model import CellCnn
from sklearn.metrics import roc_auc_score, roc_curve
import sklearn.utils as utils


def load_fcs_dataset(fcs_info_file, marker_file, co_factor=5):
    """
    Args:
        - fcs_info_file (str) :
          Path to fcs info file that contains the fcs file name and phenotypes.
          The format of this fcs info file looks like: `fcs file name (str)`, `label (int)`.
        - marker_file (str) :
          path to the marker file that contains the name of markers.
        - co_factor (float) :
          the coefficient factor of arcsinh: `x_normalized = arcsinh(co_factor * x)`.
    """
    fcs_info = np.array(pd.read_csv(fcs_info_file, sep=','))
    marker_names = list(pd.read_csv(marker_file, sep=',').columns)
    sample_ids, sample_labels = fcs_info[:, 0], fcs_info[:, 1].astype(int)
    samples, phenotypes = [], []

    fcs_dir = os.path.dirname(fcs_info_file)
    for fcs_file, label in zip(sample_ids, sample_labels):
        fname = os.path.join(fcs_dir, fcs_file)
        fcs = loadFCS(fname, transform=None, auto_comp=False)
        marker_idx = [fcs.channels.index(name) for name in marker_names]
        x = np.asarray(fcs)[:, marker_idx]
        x = ftrans(x, co_factor)
        samples.append(x)
        phenotypes.append(label)
    return samples, phenotypes


def main(args):
    # define input and output directories
    WDIR = os.path.join('.')
    FCS_DATA_PATH = args.data_root

    # define output directory
    OUTDIR = os.path.join(WDIR, f'output_{args.data_name}_{args.ncell}_{args.seed}')
    mkdir_p(OUTDIR)

    train_csv_file = os.path.join(FCS_DATA_PATH, 'train', 'train_labels.csv')
    test_csv_file = os.path.join(FCS_DATA_PATH, 'test', 'test_labels.csv')
    marker_file = os.path.join(FCS_DATA_PATH, 'marker.csv')

    # set random seed for reproducible results
    co_factor = 5.0
    np.random.seed(args.seed)

    if args.pkl:
        with open(os.path.join(FCS_DATA_PATH, 'train_HIV.pkl'), 'rb') as f:
            _data = pickle.load(f)
            train_samples, train_phenotypes = _data['sample'], _data['phenotype']
        with open(os.path.join(FCS_DATA_PATH, 'test_HIV.pkl'), 'rb') as f:
            _data = pickle.load(f)
            test_samples, test_phenotypes = _data['sample'], _data['phenotype']
    else:
        train_samples, train_phenotypes = load_fcs_dataset(train_csv_file, marker_file, co_factor)
        test_samples, test_phenotypes = load_fcs_dataset(test_csv_file, marker_file, co_factor)

    print("data io finished")

    # run a CellCnn analysis
    cellcnn = CellCnn(ncell=args.ncell, nsubset=args.nsubset, max_epochs=args.max_epochs, nrun=3, verbose=0)
    cellcnn.fit(train_samples=train_samples, train_phenotypes=train_phenotypes, outdir=OUTDIR)

    # make predictions on the test cohort
    test_pred_cellcnn = cellcnn.predict(test_samples)
    test_pred_label_cellcnn = [1 if p > 0.5 else 0 for p in test_pred_cellcnn[:, 1]]

    # look at the test set predictions
    # print('\nModel predictions:\n', test_pred_cellcnn)

    # and the true phenotypes of the test samples
    print('\nPred phenotypes:\n', test_pred_label_cellcnn)
    print('\nTrue phenotypes:\n', test_phenotypes)

    # calculate area under the ROC curve for the test set
    test_acc_cellcnn = sum(np.array(test_pred_label_cellcnn) == np.array(test_phenotypes)) / len(test_phenotypes)
    test_fpr, test_tpr, _ = roc_curve(test_phenotypes, test_pred_cellcnn[:, 1], pos_label=1)
    test_auc_cellcnn = roc_auc_score(test_phenotypes, test_pred_cellcnn[:, 1])
    print("test acc of cellcnn: ", test_acc_cellcnn)
    print("test auc of cellcnn: ", test_auc_cellcnn)

    test_stat = {
        'test_acc': test_acc_cellcnn,
        'test_auc': test_auc_cellcnn,
        'fpr': test_fpr,
        'tpr': test_tpr
    }

    with open(os.path.join(OUTDIR, 'test_result.pkl'), 'wb') as f:
        pickle.dump(test_stat, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cellCnn")

    parser.add_argument(
        '--data_root',
        type=str,
        help='the root dir of the .fcs files'
    )
    parser.add_argument(
        '--data_name',
        type=str,
        help='the dataset name'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        help='number of epochs to train'
    )
    parser.add_argument(
        '--nsubset',
        type=int,
        help='number of subsets to use'
    )
    parser.add_argument(
        '--ncell',
        type=int,
        help='number of cells'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='random seed to use'
    )
    parser.add_argument(
        '--pkl',
        action='store_true',
        help='whether to read from the pickled file'
    )

    args = parser.parse_args()

    main(args)
