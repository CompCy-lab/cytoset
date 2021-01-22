import os
import argparse
import flowio
import shutil

import numpy as np
import pandas as pd


def train_test_split(pos_idx, neg_idx, tr_prop):
    tr_pos_idx = list(np.random.choice(pos_idx, size=int(len(pos_idx) * tr_prop), replace=False))
    te_pos_idx = [i for i in pos_idx if i not in tr_pos_idx]
    tr_neg_idx = list(np.random.choice(neg_idx, size=int(len(neg_idx) * tr_prop), replace=False))
    te_neg_idx = [i for i in neg_idx if i not in tr_neg_idx]

    return tr_pos_idx, tr_neg_idx, te_pos_idx, te_neg_idx


def write_fcs(fcs_file, marker_map, out_file):
    f = flowio.FlowData(fcs_file)
    for i in range(1, f.channel_count + 1):
        key = str(i)
        if 'PnS' in f.channels[key] and f.channels[key]['PnS'] != u' ':
            f.channels[key]['PnS'] = marker_map[key]
        elif 'PnN' in f.channels[key] and f.channels[key]['PnN'] != u' ':
            f.channels[key]['PnN'] = marker_map[key]
        else:
            raise NotImplementedError('Both PnS and PnN is not the key in .fcs file')
    f.write_fcs(out_file)


def main(args):
    # read the .fcs meta file
    fcs_info = pd.read_csv(args.fcs_info_file, sep=',')
    
    # select the id and label column
    fcs_info = np.array(fcs_info[[args.id_name, args.label_name]])
    sample_ids = fcs_info[:, 0]
    sample_labels = fcs_info[:, 1]
    # assign label with name
    sample_labels[np.where(sample_labels == args.pos)] = 1
    sample_labels[np.where(sample_labels == args.neg)] = 0
    sample_labels = sample_labels.astype(int)

    # set random seed for reproducible results
    np.random.seed(12345)
    os.makedirs(args.out_dir, exist_ok=True)
    
    markers = args.marker.strip().split(',')
    marker_dic = {}
    for i, marker in enumerate(markers):
        marker_dic[str(i+1)] = marker
    # train and test split
    assert 0 <= args.train_prop <= 1, "train split should be between 0 and 1"
    group_pos = np.where(sample_labels == 1)[0]
    group_neg = np.where(sample_labels == 0)[0]

    train_pos_idx, train_neg_idx, test_pos_idx, test_neg_idx = train_test_split(
        group_pos, group_neg, tr_prop=args.train_prop
    )
    
    # write processed fcs file to output dir
    train_dir = os.path.join(args.out_dir, 'train')
    test_dir = os.path.join(args.out_dir, 'test')
    os.makedirs(train_dir, exist_ok=False)
    os.makedirs(test_dir, exist_ok=False)

    with open(os.path.join(train_dir, 'train_labels.csv'), 'w') as f:
        f.write('fcs_file,label\n')
        for idx in train_pos_idx:
            fname = os.path.join(args.fcs_data_dir, sample_ids[idx])
            write_fcs(fname, marker_dic, os.path.join(train_dir, sample_ids[idx]))
            f.write(f'{sample_ids[idx]},1\n')
        print('train positive finished')
    
        for idx in train_neg_idx:
            fname = os.path.join(args.fcs_data_dir, sample_ids[idx])
            write_fcs(fname, marker_dic, os.path.join(train_dir, sample_ids[idx]))
            f.write(f'{sample_ids[idx]},0\n')
        print('train negative finished')
    
    with open(os.path.join(test_dir, 'test_labels.csv'), 'w') as f:
        f.write('fcs_file,label\n')
        for idx in test_pos_idx:
            fname = os.path.join(args.fcs_data_dir, sample_ids[idx])
            write_fcs(fname, marker_dic, os.path.join(test_dir, sample_ids[idx]))
            f.write(f'{sample_ids[idx]},1\n')
        print('test positive finished')
        
        for idx in test_neg_idx:
            fname = os.path.join(args.fcs_data_dir, sample_ids[idx])
            write_fcs(fname, marker_dic, os.path.join(test_dir, sample_ids[idx]))
            f.write(f'{sample_ids[idx]},0\n')
        print('test negative finished')

    # write marker_file
    with open(os.path.join(args.out_dir, 'marker.csv'), "w") as f:
        f.write(args.marker)
    print('Done')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("preprocessing fcs dataset to csv")
    parser.add_argument(
        '--fcs_data_dir',
        type=str,
        help='the directory of the .fcs files'
    )
    parser.add_argument(
        '--fcs_info_file',
        type=str,
        help="path to the meta file of fcs dataset"
    )
    parser.add_argument(
        '--marker',
        type=str,
        help='relevant markers for analysis ("," seperated)'
    )
    parser.add_argument(
        '--id_name',
        type=str,
        help='id name in the fcs info file'
    )
    parser.add_argument(
        '--label_name',
        type=str,
        help="label name in the fcs info file"
    )
    parser.add_argument(
        '--pos',
        type=str,
        help='positive label name'
    )
    parser.add_argument(
        '--neg',
        type=str,
        help='negative label name'
    )
    parser.add_argument(
        '--train_prop',
        type=float,
        help='the proportion of the dataset to include in the train split'       
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        help='the output directory to hold the pre_processed '
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='random seed used for data pre-processing'
    )

    
    args = parser.parse_args()
    

    main(args)
