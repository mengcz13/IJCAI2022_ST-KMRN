'''
Generate subsequences for training/validation/test in the following format:
Output: npz file with keys:
`[train/val/test]_seq`: [#subsequences, #regions+#boroughs], array of [#frames, #feature_dim]
`_time`: [#subsequences, #regions+#boroughs], array of [#frames,]. Number of time steps from the first frame of a fully observed seq.
`_time_abs`: [#subsequences, #regions+#boroughs], array of [#frames,]. Number of time steps from the first frame of the dataset (2017-1-1 00:00)
`input_ts_max`: the time step of the last input frame. Frames with `_time`<=`input_ts_max` are input and the left are output.

fully observed seq: [t0-7d, t0-6d, ..., t0-d, t0-11.5h, t0-11h, ..., t0-0.5h, t0] -> [t0+0.5h, ..., t0+12h, t0+1d, ..., t0+7d]

'''
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import calendar
from argparse import ArgumentParser
import multiprocessing as mp
import json
import traceback

from tqdm import tqdm
import numpy as np
import pandas as pd
import swifter
import sparse


def sample_sequences(valid_volume_arr_with_borough, all_edges, all_edge_types, all_node_types, args):
    if args.max_t != np.inf:
        valid_volume_arr_with_borough = valid_volume_arr_with_borough[:args.max_t, :, :]
    l_season, l_month, l_week, l_day, l_ts = args.l_season, args.l_month, args.l_week, args.l_day, args.l_ts
    # p_season, p_month, p_week, p_day, p_ts = l_season, l_month, l_week, l_day, l_ts
    p_season, p_month, p_week, p_day, p_ts = args.p_season, args.p_month, args.p_week, args.p_day, args.p_ts
    # filename = 'sampled_irr_seqs_{}s{}m{}w{}d{}ts_{}s{}m{}w{}d{}ts.npz'.format(l_season, l_month, l_week, l_day, l_ts, p_season, p_month, p_week, p_day, p_ts)
    filename = 'sampled_irr_seqs{}.npz'.format(args.file_suffix)
    # here we use 1 month = 30 days, 1 season = 3 months = 90 days
    td_ts = timedelta(minutes=30)
    td_day = timedelta(days=1)
    td_week = timedelta(weeks=1)
    td_month = timedelta(days=30)
    td_season = timedelta(days=90)
    sd_ts = 1
    sd_day = sd_ts * int(td_day.total_seconds() / td_ts.total_seconds())
    sd_week = sd_ts * int(td_week.total_seconds() / td_ts.total_seconds())
    sd_month = sd_ts * int(td_month.total_seconds() / td_ts.total_seconds())
    sd_season = sd_ts * int(td_season.total_seconds() / td_ts.total_seconds())
    print(sd_day, sd_week, sd_month, sd_season)

    ts_shift_borough = [0]
    ts_shift_region = [0]
    for k in range(1, l_ts):
        ts_shift_borough.append(0 - k * sd_ts)
        ts_shift_region.append(0 - k * sd_ts)
    for k in range(1, l_day + 1):
        ts_shift_borough.append(0 - k * sd_day)
        ts_shift_region.append(0 - k * sd_day)
    for k in range(1, l_week + 1):
        ts_shift_borough.append(0 - k * sd_week)
        ts_shift_region.append(0 - k * sd_week)
    for k in range(1, l_month + 1):
        ts_shift_borough.append(0 - k * sd_month)
        ts_shift_region.append(0 - k * sd_month)
    for k in range(1, l_season + 1):
        ts_shift_borough.append(0 - k * sd_season)
        ts_shift_region.append(0 - k * sd_season)
    for k in range(1, p_ts + 1):
        ts_shift_borough.append(0 + k * sd_ts)
        ts_shift_region.append(0 + k * sd_ts)
    for k in range(1, p_day + 1):
        ts_shift_borough.append(0 + k * sd_day)
        ts_shift_region.append(0 + k * sd_day)
    for k in range(1, p_week + 1):
        ts_shift_borough.append(0 + k * sd_week)
        ts_shift_region.append(0 + k * sd_week)
    for k in range(1, p_month + 1):
        ts_shift_borough.append(0 + k * sd_month)
        ts_shift_region.append(0 + k * sd_month)
    for k in range(1, p_season + 1):
        ts_shift_borough.append(0 + k * sd_season)
        ts_shift_region.append(0 + k * sd_season)
    ts_shift_borough = list(set(ts_shift_borough))
    ts_shift_region = list(set(ts_shift_region))
    ts_shift_borough = np.array(ts_shift_borough, dtype=np.int64)
    ts_shift_region = np.array(ts_shift_region, dtype=np.int64)
    ts_shift_borough = np.sort(ts_shift_borough)
    ts_shift_region = np.sort(ts_shift_region)
    start_frame = ts_shift_borough[0]
    ts_shift_borough -= start_frame
    ts_shift_region -= start_frame
    input_ts_max = -start_frame
    ts_idx_region_input_flag = np.where(
        ts_shift_region <= (0 - start_frame))[0]

    print(ts_shift_borough)
    print(ts_shift_region)
    print(input_ts_max)

    input_ts_max_idx = np.where(ts_shift_region == input_ts_max)[0].item()
    resolution_ranges = [
        (input_ts_max_idx + 1 - l_ts, input_ts_max_idx), # here we exclude input_ts_max_idx since it belongs to all resolutions, and we don't mask it to make sure other resolutions have fully observed data
        # (input_ts_max_idx + 1 - l_ts - l_day, input_ts_max_idx + 1 - l_ts),
        # (input_ts_max_idx + 1 - l_ts - l_day -
        #  l_week, input_ts_max_idx + 1 - l_ts - l_day),
        # (input_ts_max_idx + 1 - l_ts - l_day - l_week -
        #  l_month, input_ts_max_idx + 1 - l_ts - l_day - l_week),
        # (input_ts_max_idx + 1 - l_ts - l_day - l_week - l_month -
        #  l_season, input_ts_max_idx + 1 - l_ts - l_day - l_week - l_month),
        (input_ts_max_idx + 1, input_ts_max_idx + 1 + p_ts), # only sample half-hour data
        # (input_ts_max_idx + 1 + p_ts, input_ts_max_idx + 1 + p_ts + p_day),
        # (input_ts_max_idx + 1 + p_ts + p_day,
        #  input_ts_max_idx + 1 + p_ts + p_day + p_week),
        # (input_ts_max_idx + 1 + p_ts + p_day + p_week,
        #  input_ts_max_idx + 1 + p_ts + p_day + p_week + p_month),
        # (input_ts_max_idx + 1 + p_ts + p_day + p_week + p_month,
        #  input_ts_max_idx + 1 + p_ts + p_day + p_week + p_month + p_season),
    ]
    print(resolution_ranges)

    seq_ts_num = ts_shift_borough[-1] - ts_shift_borough[0] + 1
    seq_num = (
        valid_volume_arr_with_borough.shape[0] - 3 * (seq_ts_num + 1)) // args.seq_diff
    seq_num = min(seq_num, args.max_samples)
    if args.debug:
        seq_num = 10
    seqs = {
        'train_seq': [], 'train_time': [], 'train_time_abs': [],
        'val_seq': [], 'val_time': [], 'val_time_abs': [],
        'test_seq': [], 'test_time': [], 'test_time_abs': [],
        'input_ts_max': input_ts_max
    }

    train_p, val_p, test_p = args.train_p, args.val_p, args.test_p
    assert (train_p + val_p + test_p == 1)
    train_n = int(train_p * seq_num)
    val_n = int(val_p * seq_num)
    trainval_n = train_n + val_n
    test_n = seq_num - trainval_n
    print('train: {}; val: {}; test: {}'.format(train_n, val_n, test_n))
    init_train_t, max_train_t = 0, max(ts_shift_borough + (train_n - 1) * args.seq_diff)
    init_val_t, max_val_t = max_train_t + 1, max_train_t + 1 + max(ts_shift_borough + (val_n - 1) * args.seq_diff)
    init_test_t, max_test_t = max_val_t + 1, max_val_t + 1 + max(ts_shift_borough + (test_n - 1) * args.seq_diff)

    print('sampling sequences...,')

    if args.sampling_protocol == 'whole':
        # generate global mask
        global_mask = np.random.rand(valid_volume_arr_with_borough.shape[0], valid_volume_arr_with_borough.shape[1]) # T_all x N

    for seq_i in tqdm(range(seq_num)):
        if seq_i < train_n:
            mode = 'train'
            init_t, max_t = init_train_t, max_train_t
        elif seq_i < trainval_n:
            mode = 'val'
            init_t, max_t = init_val_t, max_val_t
            seq_i -= train_n
        else:
            mode = 'test'
            init_t, max_t = init_test_t, max_test_t
            seq_i -= trainval_n
        graph_seq_i = []
        time_i = []
        time_i_abs = []
        ts_idx_borough = ts_shift_borough + seq_i * args.seq_diff + init_t
        ts_idx_region = ts_shift_region + seq_i * args.seq_diff + init_t
        try:
            assert max(ts_idx_borough) <= max_t
        except:
            print(mode, init_t, max_t, ts_idx_borough)
            sys.exit()
        arr_borough = valid_volume_arr_with_borough[ts_idx_borough, :, :]
        arr_region = valid_volume_arr_with_borough[ts_idx_region, :, :]
        for node_i in range(valid_volume_arr_with_borough.shape[1]):
            if all_node_types[node_i] == 0:  # region
                if args.sampling_protocol == 'whole':
                    ts_idx_mask = global_mask[ts_idx_region][:, node_i]
                    ts_idx_mask = (ts_idx_mask <= args.keep_ratio)
                    if mode == 'test': # for test all output is visible for evaluation
                        ts_idx_mask[input_ts_max_idx + 1:] = True
                elif args.sampling_protocol == 'subseq':
                    ts_idx_mask = np.ones(arr_region.shape[0], dtype=np.int64)
                    # sample keep_ratio frames for each resolution
                    for res_range in resolution_ranges:
                        si, ti = res_range
                        if si < ti:
                            if (mode == 'test') and (si > input_ts_max_idx):  # mask input only in test set
                                continue
                            tmp_ones = np.zeros(ti - si, dtype=np.int64)
                            tmp_ones[np.random.choice(
                                ti - si, int(args.keep_ratio * (ti - si)), replace=False)] = 1
                            ts_idx_mask[si:ti] = tmp_ones
                        # print(si, ti, (ts_idx_mask == 0).sum())
                    ts_idx_mask = (ts_idx_mask == 1)

                    # if mode == 'test':  # mask input seq only
                    #     ts_idx_mask[ts_idx_region_input_flag] = (np.random.rand(
                    #         len(ts_idx_region_input_flag)) <= args.keep_ratio).astype(np.int64)
                    # else:  # mask both input and output
                    #     while True:
                    #         ts_idx_mask = (np.random.rand(len(ts_idx_mask))
                    #                        <= args.keep_ratio).astype(np.int64)
                    #         if ts_idx_mask[~ts_idx_region_input_flag].sum() > 0:
                    #             break
                    #         else:
                    #             print(
                    #                 'resampling to ensure at least 1 data point in output...')
                else:
                    raise NotImplementedError()
                graph_seq_i.append(arr_region[:, node_i, :][ts_idx_mask])
                time_i.append(ts_shift_region[ts_idx_mask])
                time_i_abs.append(ts_idx_region[ts_idx_mask])
            elif all_node_types[node_i] == 1:  # borough, don't mask
                graph_seq_i.append(arr_borough[:, node_i, :])
                time_i.append(ts_shift_borough)
                time_i_abs.append(ts_idx_borough)
            else:
                raise NotImplementedError()
        # graph_seq_i = np.array(graph_seq_i, dtype=object)
        # time_i = np.array(time_i, dtype=object)
        # time_i_abs = np.array(time_i_abs, dtype=object)
        seqs['{}_seq'.format(mode)].append(graph_seq_i)
        seqs['{}_time'.format(mode)].append(time_i)
        seqs['{}_time_abs'.format(mode)].append(time_i_abs)
        

    for k in seqs:
        if k.endswith('_seq') or k.endswith('_time') or k.endswith('_time_abs'):
            obj_arr = np.zeros((len(seqs[k]), len(seqs[k][0])), dtype=object)
            for ii in range(obj_arr.shape[0]):
                for jj in range(obj_arr.shape[1]):
                    obj_arr[ii, jj] = seqs[k][ii][jj]
            seqs[k] = obj_arr

    return seqs, filename


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_folder', required=True)
    parser.add_argument('--l_ts', type=int, default=24)
    parser.add_argument('--l_day', type=int, default=7)
    # parser.add_argument('--l_week', type=int, default=4)
    # parser.add_argument('--l_month', type=int, default=3)
    # parser.add_argument('--l_season', type=int, default=4)
    parser.add_argument('--l_week', type=int, default=0)
    parser.add_argument('--l_month', type=int, default=0)
    parser.add_argument('--l_season', type=int, default=0)
    parser.add_argument('--p_ts', type=int, default=24)
    parser.add_argument('--p_day', type=int, default=7)
    parser.add_argument('--p_week', type=int, default=0)
    parser.add_argument('--p_month', type=int, default=0)
    parser.add_argument('--p_season', type=int, default=0)
    parser.add_argument('--train_p', type=float, default=0.6)
    parser.add_argument('--val_p', type=float, default=0.2)
    parser.add_argument('--test_p', type=float, default=0.2)
    parser.add_argument('--keep_ratio', type=float, default=0.4)
    parser.add_argument('--seq_diff', type=int, default=1,
                        help='time diff between neighboring seqs, 1 unit by default')
    parser.add_argument('--max_samples', type=int, default=np.inf)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sampling_protocol', type=str, default='subseq', help='subseq: sampling each subseq independently;\
         whole: randomly masking the whole sequence first, then generate subseqs')
    parser.add_argument('--file_suffix', type=str, default='')
    parser.add_argument('--max_t', type=int, default=np.inf)
    args = parser.parse_args()

    save_folder_path = Path(args.save_folder)
    
    valid_volume_arr_with_borough = sparse.load_npz(
        save_folder_path.joinpath('volume_valid_region_borough.npz'))
    edges_valid_region_borough = np.load(
        save_folder_path.joinpath('edges_valid_region_borough.npz'))
    all_edges, all_edge_types, all_node_types = edges_valid_region_borough[
        'edges'], edges_valid_region_borough['edge_types'], edges_valid_region_borough['node_types']

    # sample region sequences
    # l_season, l_month, l_week, l_day, l_ts, 1 ts is 30-minute
    # for regions, only ts data with high missing rates
    # for boroughs, season...ts data without any missing step
    np.random.seed(42)
    seqs, filename = sample_sequences(valid_volume_arr_with_borough.todense(),
                                      all_edges, all_edge_types, all_node_types, args)
    np.savez_compressed(save_folder_path.joinpath(filename), **seqs)