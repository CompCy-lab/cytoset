from typing import List, Optional

import os
import argparse
import numpy as np
import pandas as pd
import flowkit as fk
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from cellCnn.utils import (
    generate_subsets, combine_samples, loadFCS, ftrans
)


# class fcsReader(object):
#     """ fcs files reader """
#     def __init__(self):
#         pass

#     @classmethod
#     def fcs2df(
#         cls,
#         fcsFiles: List[str],
#         b: float = 1. / 5,
#         fileSampleSize: int = None,
#         excludeCols: List[str] = None,
#         excludeTransformCols: List[str] = None,
#         seed: int = 1,
#     ):
#         fcsDFs = []
#         for fcs_file in fcsFiles:
#             # read fcs data
#             fcs_df = fk.Sample(fcs_file).as_dataframe('raw')
#             # subsampling if enabled
#             if fileSampleSize is not None:
#                 fcs_df = fcs_df.sample(n=fileSampleSize, random_state=seed)

#             # filter and transform data
#             if excludeCols is not None:
#                 fcs_df.drop(columns=excludeCols, inplace=True)
#             if excludeTransformCols is not None:
#                 fcs_df = fcs_df.apply(
#                     lambda x: x if x.name in excludeTransformCols else np.arcsinh(b * x),
#                     axis=1
#                 )

#             fcsDFs.append(fcs_df)

#         return fcsDFs


# class CytoDatasetRaw(Dataset):
#     """ Cytometry Dataset Class """
#     def __init__(self, args):
#         self.args = args

#         self.fcs_info = pd.read_csv(args.fcs_metafile)
#         assert (
#             'new_name' in self.fcs_info.columns
#         ), "The path to fcs files is not specified in the meta data."

#         self.data = fcsReader.fcs2df(
#             fcsFiles=list(self.fcs_info['new_name']),
#             b=args.b,
#             fileSampleSize=args.fileSampleSize,
#             excludeTransformCols=args.excludeCols,
#             seed=args.seed
#         )

#         # transform to numpy array
#         self.data = [fcs_df.to_numpy(dtype=np.float32) for fcs_df in self.data]
#         self.target = np.array(self.fcs_info['CMV_Status'])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         data = self.data[idx]
#         # random sample a set
#         rand_idx = np.random.randint(data.shape[0], size=self.args.set_size)
#         x, y = data[rand_idx, :], self.target[idx]

#         return x, y  # data matrix and target


# class CytoDatasetFromCellCNN(Dataset):
#     """ Cytometry Dataset Class from preprocessed data using R"""
#     def __init__(self, data_file, args):
#         self.args = args
        
#         data = pd.read_csv(data_file, index_col=0)
#         id2pheo = pd.Series(data.label.values, index=data.sample_id).to_dict()
#         shuffle(data)

#         ids = np.asarray(data['sample_id']).astype(np.int32)
#         self.X = data.drop(columns=['sample_id', 'label']).to_numpy(dtype=np.float32)

#         if args.per_sample:
#             self.X, self.y = generate_subsets(
#                 self.X, id2pheo, ids, args.nsubset, args.ncell, args.per_sample
#             )
#         else:
#             nsubset_list = []
#             pheno_list = np.array([v for _, v in id2pheo.items()])
#             for pheno in range(len(np.unique(pheno_list))):
#                 nsubset_list.append(args.nsubset // np.sum(pheno == pheno_list))
#             self.X, self.y = generate_subsets(
#                 self.X, id2pheo, ids, nsubset_list, args.ncell, False
#             )
#         self.X = np.transpose(self.X, (0, 2, 1)).astype(np.float32)
#         self.y = self.y.astype(np.float32)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]


# class CytoDatasetFromCSV(Dataset):
#     def __init__(self, args, split='train'):
#         self.args = args
#         assert (
#             split in ['train', 'test']
#         ), 'split only contain train or test'

#         if split == 'train':
#             self.data = pd.read_csv(args.train_data, index_col=0)
#         else:
#             self.data = pd.read_csv(args.test_data, index_col=0)
#         self.split = split
        
#         self.data = self.data.groupby(self.data.sample_id)
#         self.data = [
#             x.drop(columns=['sample_id']).to_numpy(dtype=np.float32)
#             for _, x in list(self.data)
#         ]

#         self.n_sample = len(self.data)
#         self.set_size = args.set_size if split=='train' else args.test_size
#         if split == 'train':
#             n_cells = [len(x) for x in self.data]
#             assert (
#                 self.set_size <= min(n_cells)
#             ), "#cells in some samples is not enough to train the model"

#     def __len__(self):
#         return self.n_sample

#     def __getitem__(self, idx):
#         data = self.data[idx]
#         if self.split == 'train':
#             rand_idx = np.random.randint(data.shape[0], size=self.set_size)
#             x, y = data[rand_idx, :-1], data[rand_idx, -1][0]
#         else:
#             x, y = data[:, :-1], data[:, -1][0]
#         return x, y

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
        self.data = np.transpose(self.data, (0, 2, 1)).astype(np.float32)
        self.label = self.label.astype(np.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
