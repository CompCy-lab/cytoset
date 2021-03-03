from typing import List, Optional

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from utils import (
    generate_subsets, combine_samples, loadFCS, ftrans
)


class CytoDatasetFromCSV(Dataset):
    def __init__(
        self,
        csv_file,
        ncell=1024,
        nsubset=1000,
        per_sample=False
    ):
        """
        Args:
            - csv_file (str) :
                path to the .csv data file that contains the markers, sample_ids
                and phenotypes (labels). The columns of the .csv file is:
                `marker_1, marker_2, ..., marker_m, sample_id (int), label (str)`.
                Note: 1. the csv must have `sample_id` and `label` columns.
                      2. when using this csv dataloader, make sure the feature matrix is well
                         pre-processed and orgainized, e.g. marker selection and normalization.
            - ncell (int) :
                the number of cells per multi-cell input.
            - nsubset (int) :
                per_sample (bool): whether the `nsubset` argument refers to
                each class or each input sample.
        """
        samples = pd.read_csv(csv_file, index_col=0)
        id2pheno = pd.Series(samples['label'].values, index=samples['sample_id']).to_dict()
        sample_id = np.asarray(samples['sample_id']).astype(int)

        X_sample = samples.drop(columns=['sample_id', 'label']).to_numpy(dtype=np.float32)
        X_sample, sample_id = shuffle(X_sample, sample_id)

        if per_sample:
            self.data, self.label = generate_subsets(
                X_sample, id2pheno, sample_id, nsubset, ncell, per_sample
            )
        else:
            nsubset_list = []
            pheno_list = np.array([v for _, v in id2pheno.items()])
            for pheno in range(len(np.unique(pheno_list))):
                nsubset_list.append(nsubset // np.sum(pheno == pheno_list))
            self.data, self.label = generate_subsets(
                X_sample, id2pheno, sample_id, nsubset_list, ncell, False
            )
        self.data = np.transpose(self.data, (0, 2, 1)).astype(np.float32)
        self.label = self.label.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class CytoDatasetFromFCS(Dataset):
    def __init__(
        self,
        X_sample,
        id_sample,
        phenotypes,
        ncell=1024,
        nsubset=1000,
        per_sample=False
    ):
        """
        Args:
            - X_sample (np.array: [total_cells x nmarks]) :
                the data matrix containing multiple samples
            - id_sample  (np.array: [total_cells]) :
                the sample id each cell belongs to
            - phenotypes (list: [number of samples]) :
                the `phenotype` each sample belongs to
        """
        phenotypes = np.asarray(phenotypes)
        if per_sample:
            self.data, self.label = generate_subsets(X_sample, phenotypes, id_sample, nsubset, ncell, per_sample)
        else:
            nsubset_list = []
            for pheno in range(len(np.unique(phenotypes))):
                nsubset_list.append(nsubset // np.sum(phenotypes == pheno))
            self.data, self.label = generate_subsets(X_sample, phenotypes, id_sample, nsubset_list, ncell, per_sample)
        self.data = np.transpose(self.data, (0, 2, 1))
        self.label = self.label.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
