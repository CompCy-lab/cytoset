import os
import argparse
import logging
import flowio
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm


class MpCombiner(object):
    def __init__(self, args):
        self.args = args
        self.marker_idx = [int(x) for x in args.marker_idx.split(',')]

    def combine(self, name_df_pair):
        """ the func to perform the combination """
        ind_name, df = name_df_pair

        df = df.sort_values(by=[self.args.sub_individual_name])
        sample_ids = np.array(df[self.args.id_name])

        npy_data = []
        for fcs_id in sample_ids:
            fname = os.path.join(self.args.fcs_data_dir, fcs_id)
            fcs_data = flowio.FlowData(fname)
            events = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))[:, self.marker_idx]
            npy_data.append(events)
        if self.args.combine_axis == 'column':
            min_cells = min([len(x) for x in npy_data])
            npy_data = [x[:min_cells, :] for x in npy_data]
            combined_events = np.hstack(npy_data)
        else:
            combined_events = np.vstack(npy_data)

        return ind_name, combined_events


def main(args):
    """ Main function to perform fcs file combination """
    fcs_info = pd.read_csv(args.fcs_info_file, sep=',')
    fcs_info_grouped = fcs_info.groupby(by=[args.individual_name])
    markers = args.marker.strip('\n').split(',')
    opt_markers = args.opt_marker.strip('\n').split(',')

    os.makedirs(args.out_dir, exist_ok=True)
    print("Start writing the fcs files")

    combiner = MpCombiner(args)
    pool = multiprocessing.Pool(args.nprocs)
    tasks = pool.imap(combiner.combine, fcs_info_grouped)

    count = 0
    with tqdm(range(len(fcs_info_grouped))) as t:
        for ind_name, events in tasks:
            out_fcs_name = f"{str(ind_name).zfill(4)}.FCS"
            with open(os.path.join(args.out_dir, out_fcs_name), 'wb') as fh:
                flowio.create_fcs(
                    events.flatten().tolist(),
                    channel_names=markers,
                    opt_channel_names=opt_markers,
                    file_handle=fh
                )
            count += 1
            t.update()
    pool.close()
    pool.join()
    # make sure the processing number is correct
    assert len(fcs_info_grouped) == count

    # write meta file
    meta_file_path = os.path.join(args.out_dir, 'sample_with_labels.csv')
    FCS_file, label, ind = [], [], []
    for ind_name, df in fcs_info_grouped:
        FCS_file.append(f"{str(ind_name).zfill(4)}.FCS")
        ind.append(f"{str(ind_name)}"),
        label.append(df[args.label_name].values[0])

    meta_df = pd.DataFrame(
        {
            'FCS_file': FCS_file,
            'Individual': ind,
            'Condition': label
        }
    )
    meta_df.to_csv(meta_file_path, index=False)

    print("Combining fcs files finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Combine fcs from different tubes together")

    parser.add_argument(
        '--fcs_data_dir',
        type=str,
        help='the directory of fcs files'
    )
    parser.add_argument(
        '--fcs_info_file',
        type=str,
        help='path to fcs meta info file'
    )
    parser.add_argument(
        '--id_name',
        type=str,
        help='file id name in the fcs info file'
    )
    parser.add_argument(
        '--label_name',
        type=str,
        help='label name in the fcs info file'
    )
    parser.add_argument(
        '--individual_name',
        type=str,
        help='individual name in the fcs info file'
    )
    parser.add_argument(
        '--sub_individual_name',
        type=str,
        help="sub individual name in the fcs info file"
    )
    parser.add_argument(
        '--marker',
        type=str,
        help='channel labels to use for PnN fields'
    )
    parser.add_argument(
        '--opt_marker',
        type=str,
        default=None,
        help='channel labels to use for PnS fields'
    )
    parser.add_argument(
        '--marker_idx',
        type=str,
        help='the idx of relevant markers'
    )
    parser.add_argument(
        '--combine_axis',
        type=str,
        choices=['row', 'column'],
        help="which axis to combine"
    )
    parser.add_argument(
        '--nprocs',
        type=int,
        default=4,
        help='number of processors to use'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        help='the output directory'
    )
    args = parser.parse_args()

    main(args)
