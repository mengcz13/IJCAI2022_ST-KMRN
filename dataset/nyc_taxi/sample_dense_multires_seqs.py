'''
Generate subsequences for training/validation/test in the following format:
Output: npz file with keys:
`[train/val/test]_seq`: [#subsequences, #regions+#boroughs, #resolutions], array of [#frames, #feature_dim]
`_time`: [#subsequences, #regions+#boroughs, #resolutions], array of [#frames,]. Number of time steps from the first frame of a fully observed seq to the start time of current slot.
`_time_abs`: [#subsequences, #regions+#boroughs, #resolutions], array of [#frames,]. Number of time steps from the first frame of the dataset (2017-1-1 00:00) to the start time of current slot
`resolution_steps`: [#resolutions], array of the number of timesteps each resolution covers
`input_ts_max`: the time step of the last input frame. Frames with `_time`<=`input_ts_max` are input and the left are output.

fully observed seq:
res1: 
  _seq: [(t0-168h, t0-167.5h), ..., (t0-0.5h, t0)] -> [(t0, t0+0.5h), ..., (t0+167.5h, t0+168h)]
  _time: [0, 1, ..., 335] -> [336, 337, ..., 671]
res2:
  _seq: [(t0-7d, t0-6d), ..., (t0-1d, t0)] -> [(t0, t0+1d), ..., (t0+6d, t0+7d)]
  _time: [0, 48, ..., 288] -> [336, 384, ..., 624]
`resolution_steps`: [1, 48]
`input_ts_max`: 335
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


def smooth(seq, window_size):
    v = np.ones(window_size) / window_size
    seq_sm = np.stack([
        np.convolve(seq[:, k], v, mode='same') for k in range(seq.shape[-1])
    ], axis=-1)
    return seq_sm


def sample_sequences(valid_volume_arr_with_borough, all_edges, all_edge_types, all_node_types, args):
    if args.single_node >= 0:
        valid_volume_arr_with_borough = valid_volume_arr_with_borough[:, args.single_node:args.single_node+1, :]

    if args.max_t != np.inf:
        valid_volume_arr_with_borough = valid_volume_arr_with_borough[:args.max_t, :, :]

    if args.smooth_window > 0:
        for ni in range(valid_volume_arr_with_borough.shape[1]):
            valid_volume_arr_with_borough[:, ni, :] = smooth(
                valid_volume_arr_with_borough[:, ni, :], args.smooth_window)
    
    print(valid_volume_arr_with_borough.shape)

    filename = 'sampled_dense_multires_seqs{}.npz'.format(args.file_suffix)
    # here we use 1 month = 30 days, 1 season = 3 months = 90 days
    td_ts = timedelta(minutes=30)
    td_3h = timedelta(hours=3)
    td_day = timedelta(days=1)
    td_week = timedelta(weeks=1)
    td_month = timedelta(days=30)
    td_season = timedelta(days=90)
    sd_ts = 1
    sd_3h = sd_ts * int(td_3h.total_seconds() / td_ts.total_seconds())
    sd_day = sd_ts * int(td_day.total_seconds() / td_ts.total_seconds())
    sd_week = sd_ts * int(td_week.total_seconds() / td_ts.total_seconds())
    sd_month = sd_ts * int(td_month.total_seconds() / td_ts.total_seconds())
    sd_season = sd_ts * int(td_season.total_seconds() / td_ts.total_seconds())
    sd_dict = {
        'ts': sd_ts, '3h': sd_3h, 'day': sd_day, 'week': sd_week, 'month': sd_month, 'season': sd_season
    }
    print(sd_dict)

    res_list = []
    for resname in ['ts', '3h', 'day']:
        l_steps, p_steps = getattr(
            args, 'l_' + resname), getattr(args, 'p_' + resname)
        if (l_steps == 0) or (p_steps == 0):
            pass
        else:
            res_list.append(resname)
    print('res list:', res_list)

    def _generate_chrono_encoding(time_abs):
        td_year = timedelta(days=365)
        sd_year = sd_ts * int(td_year.total_seconds() / td_ts.total_seconds())
        # consider periods over 1 year, season, month, week, day
        periods = [sd_year, sd_season, sd_month, sd_week, sd_day]
        # for N periods considered, each time corresponds to 2N-dim encoding, (sin, cos) for each period
        chrono_encoding = []
        for period in periods:
            deg = 2 * np.pi / period * time_abs
            chrono_encoding.append(np.sin(deg))
            chrono_encoding.append(np.cos(deg))
        chrono_encoding = np.stack(chrono_encoding, axis=-1)
        return chrono_encoding

    def _get_ts_shift_for_res(resname):
        l_steps, p_steps = getattr(
            args, 'l_' + resname), getattr(args, 'p_' + resname)
        sd_res = sd_dict[resname]
        ts_shift_full_input_starts = (-np.arange(1,
                                                 1 + l_steps) * sd_res)[::-1]
        ts_shift_full_output_starts = np.arange(0, p_steps) * sd_res
        ts_shift_full_starts = np.concatenate(
            (ts_shift_full_input_starts, ts_shift_full_output_starts))
        ts_shift_full_end = ts_shift_full_starts[-1] + sd_res
        return [ts_shift_full_starts, ts_shift_full_end, sd_res]

    ts_shift_dict = {k: _get_ts_shift_for_res(k) for k in res_list}
    ts_shift_first_start_list = [ts_shift_dict[k][0][0] for k in res_list]
    ts_shift_last_end_list = [ts_shift_dict[k][1] for k in res_list]
    overall_ts_shift_start = min(ts_shift_first_start_list)
    overall_ts_shift_end = max(ts_shift_last_end_list)
    overall_ts_len = overall_ts_shift_end - overall_ts_shift_start
    input_ts_max = -1 - overall_ts_shift_start
    # make all ts_shift start from 0
    for k in res_list:
        ts_shift_dict[k][0] -= overall_ts_shift_start
        ts_shift_dict[k][1] -= overall_ts_shift_start

    train_p, val_p, test_p = args.train_p, args.val_p, args.test_p
    assert (train_p + val_p + test_p == 1)
    total_frame_num = valid_volume_arr_with_borough.shape[0]
    train_frame_start, train_frame_end = 0, np.round(
        train_p * total_frame_num).astype(int)
    val_frame_start, val_frame_end = train_frame_end, train_frame_end + \
        np.round(val_p * total_frame_num).astype(int)
    test_frame_start, test_frame_end = val_frame_end, total_frame_num

    def _sample_seq_in_interval(reslist, frame_start, frame_end):
        seq_num = ((frame_end - frame_start) -
                   overall_ts_len + 1) // args.seq_diff
        seq_arr = np.zeros(
            (seq_num, valid_volume_arr_with_borough.shape[1], len(reslist)), dtype=object)
        time_arr = np.zeros_like(seq_arr)
        time_abs_arr = np.zeros_like(time_arr)
        chrono_enc_arr = np.zeros_like(time_arr)
        for ti in tqdm(range(seq_num)):
            for res_i, res in enumerate(reslist):
                ts_shift_full_starts, _, sd_res = ts_shift_dict[res]
                ts_sel_start = frame_start + ti * args.seq_diff
                ts_sel_full_starts = ts_shift_full_starts + ts_sel_start
                seq_full = np.array(
                    [valid_volume_arr_with_borough[x:x+sd_res].mean(axis=0) for x in ts_sel_full_starts])
                time_full = ts_shift_full_starts
                time_abs_full = ts_sel_full_starts
                if args.keep_ratio < 1:
                    raise NotImplementedError(
                        'TODO: add sampling for each resolution')
                else:
                    for ni in range(seq_arr.shape[1]):
                        seq_arr[ti, ni, res_i] = seq_full[:, ni, :]
                        time_arr[ti, ni, res_i] = time_full
                        time_abs_arr[ti, ni, res_i] = time_abs_full
                        chrono_enc_arr[ti, ni, res_i] = _generate_chrono_encoding(
                            time_abs_full)
        return seq_arr, time_arr, time_abs_arr, chrono_enc_arr

    train_seq_arr, train_time_arr, train_time_abs_arr, train_chrono_enc_arr = _sample_seq_in_interval(
        res_list, train_frame_start, train_frame_end)
    val_seq_arr, val_time_arr, val_time_abs_arr, val_chrono_enc_arr = _sample_seq_in_interval(
        res_list, val_frame_start, val_frame_end)
    test_seq_arr, test_time_arr, test_time_abs_arr, test_chrono_enc_arr = _sample_seq_in_interval(
        res_list, test_frame_start, test_frame_end)

    seqs = {
        'train_seq': train_seq_arr, 'train_time': train_time_arr, 'train_time_abs': train_time_abs_arr,
        'val_seq': val_seq_arr, 'val_time': val_time_arr, 'val_time_abs': val_time_abs_arr,
        'test_seq': test_seq_arr, 'test_time': test_time_arr, 'test_time_abs': test_time_abs_arr,
        'input_ts_max': input_ts_max,
        'resolution_steps': np.array([ts_shift_dict[k][2] for k in res_list]).astype(int)
    }

    if args.with_chrono_encoding:
        seqs['train_chrono_enc'] = train_chrono_enc_arr
        seqs['val_chrono_enc'] = val_chrono_enc_arr
        seqs['test_chrono_enc'] = test_chrono_enc_arr

    return seqs, filename


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_folder', required=True)
    parser.add_argument('--l_ts', type=int, default=48*7)
    parser.add_argument('--l_3h', type=int, default=8*7)
    parser.add_argument('--l_day', type=int, default=7)
    # parser.add_argument('--l_week', type=int, default=4)
    # parser.add_argument('--l_month', type=int, default=3)
    # parser.add_argument('--l_season', type=int, default=4)
    parser.add_argument('--l_week', type=int, default=0)
    parser.add_argument('--l_month', type=int, default=0)
    parser.add_argument('--l_season', type=int, default=0)
    parser.add_argument('--p_ts', type=int, default=48*7)
    parser.add_argument('--p_3h', type=int, default=8*7)
    parser.add_argument('--p_day', type=int, default=7)
    parser.add_argument('--p_week', type=int, default=0)
    parser.add_argument('--p_month', type=int, default=0)
    parser.add_argument('--p_season', type=int, default=0)
    parser.add_argument('--train_p', type=float, default=0.6)
    parser.add_argument('--val_p', type=float, default=0.2)
    parser.add_argument('--test_p', type=float, default=0.2)
    parser.add_argument('--keep_ratio', type=float, default=0.4)
    parser.add_argument('--seq_diff', type=int, default=48,
                        help='time diff between neighboring seqs, 1 day by default')
    parser.add_argument('--max_samples', type=int, default=np.inf)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sampling_protocol', type=str, default='subseq', help='subseq: sampling each subseq independently;\
         whole: randomly masking the whole sequence first, then generate subseqs')
    parser.add_argument('--file_suffix', type=str, default='')
    parser.add_argument('--max_t', type=int, default=np.inf)
    parser.add_argument('--with_chrono_encoding', action='store_true')
    parser.add_argument('--smooth_window', type=int, default=0,
                        help='using smoothed data on the finest resolution')
    parser.add_argument('--single_node', type=int, default=-1)
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
