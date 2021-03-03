import os
import argparse
import flowio
import parc

import numpy as np
import pandas as pd
from rich.console import Console


def train_test_split(pos_idx, neg_idx, tr_prop):
    tr_pos_idx = list(np.random.choice(pos_idx, size=int(len(pos_idx) * tr_prop), replace=False))
    te_pos_idx = [i for i in pos_idx if i not in tr_pos_idx]
    tr_neg_idx = list(np.random.choice(neg_idx, size=int(len(neg_idx) * tr_prop), replace=False))
    te_neg_idx = [i for i in neg_idx if i not in tr_neg_idx]

    return tr_pos_idx, tr_neg_idx, te_pos_idx, te_neg_idx


def down_sampling(f, down_rate=0.1, clust_id=None, co_factor=5, jac_std_global=0.15, seed=12345):
    """
    Args:
        - f (file handler): fcs file handler returned by flowio.FlowData()
        - down_rate (float): down sampling rate
        - co_factor: coefficient factor to normalize the data
        - jac_std_global: jac_std_global parameter to run clustering
        - seed: random seed for reproducibility
    """
    npy_events = np.reshape(f.events, (-1, f.channel_count))
    events = np.arcsinh(1. / co_factor * npy_events[:, clust_id])
    # clustering with parc
    parc_obj = parc.PARC(
        events,
        jac_std_global=jac_std_global,
        num_threads=6,
        random_seed=seed,
        small_pop=50
    )
    parc_obj.run_PARC()
    clust_label = np.asarray(parc_obj.labels)
    n_cluster = len(np.unique(clust_label))

    # downsampling cells from each cluster
    downsampled_events = []
    for i in range(n_cluster):
        idx = np.where(clust_label == i)[0]
        selected_idx = np.random.choice(idx, int(down_rate * len(idx)), replace=False)
        downsampled_events.append(npy_events[selected_idx])
    downsampled_events = np.vstack(downsampled_events)

    return downsampled_events


def random_sampling(f, sample_size):
    """
    Args:
        - f (file handler): fcs file handler returned by flowio.FlowData
        - sample_size(int): the number of cells to sample
    """
    npy_events = np.reshape(f.events, (-1, f.channel_count))

    # random sampling cells from each individual
    selected_idx = np.random.choice(f.event_count, sample_size, replace=False)
    sampled_events = npy_events[selected_idx, :]
    return sampled_events


def write_fcs(fcs_file, marker_map, out_file, downsampling=None, clust_id=None, co_factor=5, jac_std=0.15, seed=12345):
    f = flowio.FlowData(fcs_file)

    for i in range(1, f.channel_count + 1):
        key = str(i)
        if 'PnS' in f.channels[key] and f.channels[key]['PnS'] != u' ':
            f.channels[key]['PnS'] = marker_map[key]
        elif 'PnN' in f.channels[key] and f.channels[key]['PnN'] != u' ':
            f.channels[key]['PnN'] = marker_map[key]
        else:
            raise NotImplementedError('Both PnS and PnN is not the key in .fcs file')

    if downsampling:
        if downsampling > 0 and downsampling < 1:
            if clust_id is not None:
                clust_id = [int(x) for x in clust_id.split(',')]
            else:
                clust_id = [i for i in range(f.channel_count)]

            downsampled_events = down_sampling(
                f, downsampling, clust_id, co_factor, jac_std, seed)
        else:
            downsampled_events = random_sampling(f, down_sampling)

        fh = open(out_file, 'wb')
        flowio.create_fcs(
            downsampled_events.flatten().tolist(),
            channel_names=[chn['PnN'] for _, chn in f.channels.items()],
            opt_channel_names=[chn['PnS'] for _, chn in f.channels.items()],
            file_handle=fh
        )
        fh.close()
    else:
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
        marker_dic[str(i + 1)] = marker
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

    console = Console()
    console.print("start generating training samples :rocket:")

    with console.status("[bold green]Working on tasks...") as status:
        with open(os.path.join(train_dir, 'train_labels.csv'), 'w') as f:
            f.write('fcs_file,label\n')
            for idx in train_pos_idx:
                fname = os.path.join(args.fcs_data_dir, sample_ids[idx])
                write_fcs(fname, marker_dic, os.path.join(train_dir, sample_ids[idx]), args.downsampling,
                    args.cluster_marker_id, args.co_factor, args.jac_std, args.seed)
                f.write(f'{sample_ids[idx]},1\n')
            console.log('train positive finished')

            for idx in train_neg_idx:
                fname = os.path.join(args.fcs_data_dir, sample_ids[idx])
                write_fcs(fname, marker_dic, os.path.join(train_dir, sample_ids[idx]), args.downsampling,
                    args.cluster_marker_id, args.co_factor, args.jac_std, args.seed)
                f.write(f'{sample_ids[idx]},0\n')
            console.log('train negative finished')

        with open(os.path.join(test_dir, 'test_labels.csv'), 'w') as f:
            f.write('fcs_file,label\n')
            for idx in test_pos_idx:
                fname = os.path.join(args.fcs_data_dir, sample_ids[idx])
                write_fcs(fname, marker_dic, os.path.join(test_dir, sample_ids[idx]), args.downsampling,
                    args.cluster_marker_id, args.co_factor, args.jac_std, args.seed)
                f.write(f'{sample_ids[idx]},1\n')
            console.log('test positive finished')

            for idx in test_neg_idx:
                fname = os.path.join(args.fcs_data_dir, sample_ids[idx])
                write_fcs(fname, marker_dic, os.path.join(test_dir, sample_ids[idx]), args.downsampling,
                    args.cluster_marker_id, args.co_factor, args.jac_std, args.seed)
                f.write(f'{sample_ids[idx]},0\n')
            console.log('test negative finished')

        # write marker_file
        with open(os.path.join(args.out_dir, 'marker.csv'), "w") as f:
            f.write(args.train_marker)
    console.print('finished :tada:')


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
        help='markers in the dataset'
    )
    parser.add_argument(
        '--train_marker',
        type=str,
        help='the markers that are meaningful for training the model'
    )
    parser.add_argument(
        '--cluster_marker_id',
        type=str,
        default=None,
        help='the marker idx used to cluster cells'
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
        '--downsampling',
        type=float,
        default=None,
        help="downsampling rate"
    )
    parser.add_argument(
        '--co_factor',
        type=float,
        default=5.0,
        help='the coefficient of arcsinh normalization'
    )
    parser.add_argument(
        '--jac_std',
        type=float,
        default=0.15,
        help='jac std parameter to run parc'
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
