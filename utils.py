import os
import gc
import psutil
import numpy as np
import pandas as pd
import torch
import flowio

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle


class EarlyStopping(object):
    """
    Early stops the training if validation loss
    doesn't improve after a given patience
    """
    def __init__(
        self,
        patience=5,
        verbose=False,
        delta=0
    ):
        """
        Args:
            patience (int): how long to wait after last validation loss improved.
            verbose (bool): whether to print a message for each validation loss.
            delta (float): minimum change in the monitored quantity to qualify
                        as an improvement.
            trace_func (function): trace print function.
        """
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        # self.trace_func = trace_func

    def __call__(self, val_loss):
        score = - val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def ftrans(x, c):
    return np.arcsinh(1. / c * x)


class FcmData(object):
    """ tool class from (Cellcnn) """
    def __init__(self, events, channels):
        self.channels = channels
        self.events = events
        self.shape = events.shape

    def __array__(self):
        return self.events


def loadFCS(filename, *args, **kwargs):
    """ read .fcs file from (Cellcnn) """
    f = flowio.FlowData(filename)
    events = np.reshape(f.events, (-1, f.channel_count))
    channels = []
    for i in range(1, f.channel_count + 1):
        key = str(i)
        if 'PnS' in f.channels[key] and f.channels[key]['PnS'] != u' ':
            channels.append(f.channels[key]['PnS'])
        elif 'PnN' in f.channels[key] and f.channels[key]['PnN'] != u' ':
            channels.append(f.channels[key]['PnN'])
        else:
            channels.append('None')
    return FcmData(events, channels)


def combine_samples(data_list, sample_id):
    accum_x, accum_y = [], []
    for x, y in zip(data_list, sample_id):
        accum_x.append(x)
        accum_y.append(y * np.ones(x.shape[0], dtype=int))
    return np.vstack(accum_x), np.hstack(accum_y)


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


def train_valid_split(samples, sample_ids, k_fold=5):
    """
    Args:
        - samples (List[np.array]) : a list of feature matrices of samples
        - sample_ids (List[int]) : the sample ids
        - k_fold (int): the k-fold parameter to split the data to the train and valid
    """
    X, sample_id = combine_samples(samples, sample_ids)
    kf = StratifiedKFold(n_splits=k_fold)

    train_indices, valid_indices = next(kf.split(X, sample_id))
    X_train, id_train = X[train_indices], sample_id[train_indices]
    X_valid, id_valid = X[valid_indices], sample_id[valid_indices]

    return X_train, id_train, X_valid, id_valid


def down_rsampling(arr: np.array, sample_size, axis=0):
    selected_idx = np.random.choice(arr.shape[axis], sample_size, replace=False)
    return arr.take(indices=selected_idx, axis=axis)


def generate_subsets(X, pheno_map, sample_id, nsubsets, ncell,
                     per_sample=False, k_init=False):
    S = dict()
    n_out = len(np.unique(sample_id))

    for ylabel in range(n_out):
        X_i = filter_per_class(X, sample_id, ylabel)
        if per_sample:
            S[ylabel] = per_sample_subsets(X_i, nsubsets, ncell, k_init)
        else:
            n = nsubsets[pheno_map[ylabel]]
            S[ylabel] = per_sample_subsets(X_i, n, ncell, k_init)

    # mix them
    Xt, yt = [], []
    for y_i, x_i in S.items():
        Xt.append(x_i)
        yt.append(pheno_map[y_i] * np.ones(x_i.shape[0], dtype=int))
    del S
    gc.collect()

    Xt = np.vstack(Xt)
    yt = np.hstack(yt)
    Xt, yt = shuffle(Xt, yt)
    
    return Xt, yt


def filter_per_class(X, y, ylabel):
    return X[np.where(y == ylabel)]


def per_sample_subsets(X, nsubsets, ncell_per_subset, k_init=False):
    nmark = X.shape[1]
    shape = (nsubsets, nmark, ncell_per_subset)
    Xres = np.zeros(shape, dtype=np.float32)

    if not k_init:
        for i in range(nsubsets):
            X_i = random_subsample(X, ncell_per_subset)
            Xres[i] = X_i.T
    else:
        for i in range(nsubsets):
            X_i = random_subsample(X, 2000)
            X_i = kmeans_subsample(X_i, ncell_per_subset, random_state=i)
            Xres[i] = X_i.T
    return Xres

def random_subsample(X, target_nobs, replace=True):

    """ Draws subsets of cells uniformly at random. """

    nobs = X.shape[0]
    if (not replace) and (nobs <= target_nobs):
        return X
    else:
        indices = np.random.choice(nobs, size=target_nobs, replace=replace)
        return X[indices, :]