'''
A PyTorch Dataset class for loading nyctaxi data in manhattan, ranging from 2017/1/1 to 2019/12/31
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
from copy import deepcopy
from collections import namedtuple
import itertools

from tqdm import tqdm
import numpy as np
import pandas as pd
import sparse

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def calc_t_delta(ts, mask):
    delta = np.zeros_like(ts)
    for ti in range(1, len(delta)):
        delta[ti] = ts[ti] - ts[ti - 1] + (1 - mask[ti - 1]) * delta[ti - 1]
    return delta


class NYCTaxiDataset(Dataset):
    def __init__(self, save_folder, input_len, output_len,
                 # list of res names ['week', 'day', 'X-hour]
                 historical_temp_res,
                 forecast_temp_res,  # list of res names
                 keep_ratio=1.0,  # control the ratio of observed data for the finest spatial & temporal res
                 trainval_ps=(0.6, 0.2), mask_seed=42,
                 seq_diff=1, max_t=None, single_node=None,  # options for limiting data size
                 data_type='train',
                 scale=False,
                 scaler=None,
                 return_delta=True,
                 output_format='multivar',
                 resolution_type='agg', # agg: aggregate data for coarse resolutions; sample: coarse resolutions mean lower sampling rates
                 chrono_arr_cats='mdwh', # default: [month, day, weekday, hour]
                 ):
        super().__init__()

        save_folder_path = Path(save_folder)

        # volume data: [#timesteps, #nodes, #feature]
        self.volume_data = sparse.load_npz(
            save_folder_path.joinpath('volume_valid_region_borough.npz')).todense()
        edges_valid_region_borough = np.load(
            save_folder_path.joinpath('edges_valid_region_borough.npz'))
        # all_edges: [#edges, 2]
        # edge_types: [#edges], 0 - region to region; 1 - region and borough; 2 - borough to borough
        # node_types: [#nodes], 0 - region; 1 - borough
        self.all_edges, self.all_edge_types, self.all_node_types = edges_valid_region_borough[
            'edges'], edges_valid_region_borough['edge_types'], edges_valid_region_borough['node_types']

        self.input_len = input_len
        self.output_len = output_len
        self.historical_temp_res = historical_temp_res
        self.forecast_temp_res = forecast_temp_res
        self.keep_ratio = keep_ratio

        self.train_p, self.val_p = trainval_ps
        self.test_p = 1 - self.train_p - self.val_p
        self.seq_diff = seq_diff
        self.max_t = max_t
        self.single_node = single_node
        self.data_type = data_type
        assert self.data_type in ['train', 'val', 'test', 'all']
        self.return_delta = return_delta

        assert output_format in ['multivar', 'graph', 'sep_multires']
        self.output_format = output_format

        self.resolution_type = resolution_type

        # temporal info
        temporal_info_path = save_folder_path.joinpath('temporal_info.npz')
        if temporal_info_path.exists():
            temporal_info = np.load(temporal_info_path, allow_pickle=True)
            self.start_datetime = temporal_info['start'].item()
            self.step_timedelta = temporal_info['delta'].item()
        else: # for NYC taxi
            self.start_datetime = datetime(year=2017, month=1, day=1)
            self.step_timedelta = timedelta(minutes=30)

        self.volume_data_date = [
            self.start_datetime + self.step_timedelta * k for k in range(self.volume_data.shape[0])
        ]
        # month, day, weekday, hour, (minute)
        self.chrono_arr_cats = chrono_arr_cats
        self.chrono_arr = np.zeros((len(self.volume_data_date), len(self.chrono_arr_cats)))
        for vi, vdd in enumerate(self.volume_data_date):
            if self.chrono_arr_cats == 'mdwh':
                self.chrono_arr[vi] = [
                    vdd.month, vdd.day, vdd.weekday(), vdd.hour
                ]
            elif self.chrono_arr_cats == 'mdwhm':
                self.chrono_arr[vi] = [
                    vdd.month, vdd.day, vdd.weekday(), vdd.hour, vdd.minute
                ]
            else:
                raise NotImplementedError()

        if self.single_node is not None:
            self.volume_data = self.volume_data[:,
                                                self.single_node:self.single_node+1, :]
            self.all_edges = None
            self.all_edge_types = None
            self.all_node_types = self.all_node_types[self.single_node:self.single_node+1]

        if self.max_t is not None:
            self.volume_data = self.volume_data[:self.max_t]
            self.volume_data_date = self.volume_data_date[:self.max_t]
            self.chrono_arr = self.chrono_arr[:self.max_t]

        # standard scale if necessary, with unmasked data
        self.scale = scale
        if self.scale:
            nt, nn, nf = self.volume_data.shape
            if scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(self.volume_data.reshape(-1, nf))
            else:
                self.scaler = scaler
            trans_data = self.scaler.transform(self.volume_data.reshape(-1, nf))
            self.volume_data = trans_data.reshape(nt, nn, nf)
            print('mean:', self.scaler.mean_)
            print('var:', self.scaler.var_)

        # range for selecting samples
        total_ts = self.volume_data.shape[0]
        if self.data_type == 'train':
            t_start, t_end = 0, int(self.train_p * total_ts)
        elif self.data_type == 'val':
            t_start, t_end = int(
                self.train_p * total_ts) - self.input_len, int((self.train_p + self.val_p) * total_ts)
        elif self.data_type == 'test':
            t_start, t_end = int(
                (self.train_p + self.val_p) * total_ts) - self.input_len, total_ts
        elif self.data_type == 'all':
            t_start, t_end = 0, total_ts
        else:
            raise NotImplementedError()
        self.start_ts = np.arange(
            t_start, t_end - self.input_len - self.output_len + 1, self.seq_diff)

        print(self.volume_data_date[t_start])

        if self.resolution_type == 'agg':
            # generate multi-res temporal data
            self._gen_multi_temp_res_data()

            # generate mask for volume data
            self.mask_rng = np.random.RandomState(seed=mask_seed)
            self._gen_data_mask()
        elif self.resolution_type == 'sample':
            self._gen_daemon_data_mask()

    def _get_ts_span(self, res):
        if res == 'week':
            ts_span = np.round(timedelta(weeks=1).total_seconds() /
                          self.step_timedelta.total_seconds()).astype(int).item()
        elif res == 'day':
            ts_span = np.round(timedelta(days=1).total_seconds() /
                          self.step_timedelta.total_seconds()).astype(int).item()
        elif res.endswith('-hour'):
            hours = int(res.split('-')[0])
            ts_span = np.round(timedelta(hours=hours).total_seconds() /
                          self.step_timedelta.total_seconds()).astype(int).item()
        else:
            raise NotImplementedError()
        return ts_span

    def _get_start_ts(self, res):
        # first time step starts from 2017-01-01 00:00:00, so the starting point is t=0 except for week
        if res == 'week':
            start_t_0 = 0
            while self.chrono_arr[start_t_0, 2] != 0:
                start_t_0 += 1  # find first monday
        else:
            start_t_0 = 0

        ts_span = self._get_ts_span(res)
        
        # last frame is the end
        start_ts = np.arange(start_t_0, self.chrono_arr.shape[0] + 1, ts_span)
        return start_ts

    def _get_temp_res_data(self, res, agg='mean'):
        start_ts = self._get_start_ts(res)
        data_res = []
        for si, ei in zip(start_ts[:-1], start_ts[1:]):
            if agg == 'mean':
                data_res_si = np.mean(self.volume_data[si:ei], axis=0)
            elif agg == 'sum':
                data_res_si = np.sum(self.volume_data[si:ei], axis=0)
            else:
                raise NotImplementedError()
            data_res.append(data_res_si)
        data_res = np.stack(data_res, axis=0)
        return data_res, start_ts

    def _gen_multi_temp_res_data(self):
        self.multi_temp_res_data = {}
        for temp_res_list in [self.historical_temp_res, self.forecast_temp_res]:
            for temp_res in temp_res_list:
                if temp_res not in self.multi_temp_res_data:
                    data_res, start_ts = self._get_temp_res_data(
                        temp_res, agg='mean')
                    self.multi_temp_res_data[temp_res] = {
                        'volume_data': data_res,
                        'start_ts': start_ts
                    }

    def _gen_data_mask(self):
        # randomly mask volume_data: each location has prob keep_ratio to be kept
        rng = self.mask_rng
        self.volume_data_mask = (rng.random(
            size=self.volume_data.shape) <= self.keep_ratio).astype(int)
        # don't mask boroughs!
        borough_ids = np.where(self.all_node_types == 1)[0]
        self.volume_data_mask[:, borough_ids, :] = 1
        self.masked_volume_data = deepcopy(self.volume_data)
        self.masked_volume_data *= self.volume_data_mask # fill 0 in parts masked out

    def _gen_daemon_data_mask(self):
        # mask no data, generate masks and masked data just for compatibility
        self.volume_data_mask = np.ones_like(self.volume_data).astype(int)
        self.masked_volume_data = deepcopy(self.volume_data)

    def _select_multi_temp_res_data_in_t(self, res, start_t, end_t):
        multi_temp_res_data = self.multi_temp_res_data[res]
        res_data, res_start_ts = multi_temp_res_data['volume_data'], multi_temp_res_data['start_ts']
        # multi res start_t <= start_t
        res_start_t_idx = np.searchsorted(res_start_ts, start_t, side='right') - 1
        res_start_t_idx = max(0, res_start_t_idx)
        # multi res end_t <= end_t
        res_end_t_idx = np.searchsorted(res_start_ts, end_t, side='right') - 1
        sel_res_data = res_data[res_start_t_idx:res_end_t_idx]
        sel_res_start_ts = res_start_ts[res_start_t_idx:res_end_t_idx+1]
        return sel_res_data, sel_res_start_ts

    def _merge_multires_to_multivar(self, data, data_t_start, data_t_end, multires_data, data_mask):
        # data: [T, N, F], T in [data_t_start, data_t_end]
        merged_data = [data,]
        merged_mask = [data_mask,]
        for res_data, res_start_ts in multires_data:
            res_data_disc = np.zeros_like(data)
            res_data_mask_disc = np.zeros_like(data_mask)
            for idx in range(res_data.shape[0]):
                si, ei = res_start_ts[idx], res_start_ts[idx+1]
                si = max(0, si - data_t_start)
                ei = min(data_t_end, ei - data_t_start)
                res_data_disc[si:ei] = res_data[idx]
                res_data_mask_disc[si:ei] = 1
            merged_data.append(res_data_disc)
            merged_mask.append(res_data_mask_disc)
        merged_data = np.concatenate(merged_data, axis=-1)
        merged_mask = np.concatenate(merged_mask, axis=-1)
        return merged_data, merged_mask

    def _merge_multires_to_list(self, data, multires_data, data_mask, chrono):
        # data: [T, N, F], T in [data_t_start, data_t_end]
        merged_data = [data,]
        merged_mask = [data_mask,]
        merged_chrono = [chrono,]
        for res_data, res_start_ts in multires_data:
            merged_data.append(res_data)
            merged_mask.append(np.ones_like(res_data).astype(int))
            merged_chrono.append(self.chrono_arr[res_start_ts[:-1]])
        return merged_data, merged_mask, merged_chrono

    def _merge_multi_sample_res_to_multivar(self, data, data_t_start, data_t_end, sample_res_list, data_mask, direction='forward'):
        # data: [T, N, F], T in [data_t_start, data_t_end]
        # like _merge_multires_to_multivar, but values are from low sampling rates instead of aggregation
        # forward direction for output seqs; backward direction for input seqs
        merged_data = [data,]
        merged_mask = [data_mask,]
        for sample_res in sample_res_list:
            ts_span = self._get_ts_span(sample_res)
            if direction == 'forward':
                ts_sel = np.arange(data_t_start, data_t_end, ts_span, dtype=np.int64)
                ts_sel = np.repeat(ts_sel, ts_span)[:(data_t_end - data_t_start)]
            elif direction == 'backward':
                # ts_sel = -np.arange(-data_t_end + 1, -data_t_start + 1, ts_span, dtype=np.int64)
                ts_sel = -np.arange(-data_t_end + ts_span, -data_t_start + 1, ts_span, dtype=np.int64)
                ts_sel = np.repeat(ts_sel, ts_span)[:(data_t_end - data_t_start)][::-1]
            res_data = self.masked_volume_data[ts_sel]
            res_data_mask = self.volume_data_mask[ts_sel]
            merged_data.append(res_data)
            merged_mask.append(res_data_mask)
        merged_data = np.concatenate(merged_data, axis=-1)
        merged_mask = np.concatenate(merged_mask, axis=-1)
        return merged_data, merged_mask

    def _merge_multi_sample_res_to_list(self, data, data_t_start, data_t_end, sample_res_list, data_mask, chrono, direction='forward'):
        # data: [T, N, F], T in [data_t_start, data_t_end]
        # like _merge_multires_to_list, but values are from low sampling rates instead of aggregation
        # forward direction for output seqs; backward direction for input seqs
        merged_data = [data,]
        merged_mask = [data_mask,]
        merged_chrono = [chrono,]
        for sample_res in sample_res_list:
            ts_span = self._get_ts_span(sample_res)
            if direction == 'forward':
                ts_sel = np.arange(data_t_start, data_t_end, ts_span, dtype=np.int64)
                # ts_sel = np.repeat(ts_sel, ts_span)[:(data_t_end - data_t_start)]
            elif direction == 'backward':
                # ts_sel = -np.arange(-data_t_end + 1, -data_t_start + 1, ts_span, dtype=np.int64)
                ts_sel = -np.arange(-data_t_end + ts_span, -data_t_start + 1, ts_span, dtype=np.int64)
                ts_sel = ts_sel[::-1]
                # ts_sel = np.repeat(ts_sel, ts_span)[:(data_t_end - data_t_start)][::-1]
            res_data = self.masked_volume_data[ts_sel]
            res_data_mask = self.volume_data_mask[ts_sel]
            merged_data.append(res_data)
            merged_mask.append(res_data_mask)
            merged_chrono.append(self.chrono_arr[ts_sel])
        # merged_data = np.concatenate(merged_data, axis=-1)
        # merged_mask = np.concatenate(merged_mask, axis=-1)
        return merged_data, merged_mask, merged_chrono
    
    def _calc_delta_arr(self, ts, mask):
        delta_arr = np.zeros_like(mask, dtype=np.float32)
        for ni in range(mask.shape[1]):
            for fi in range(mask.shape[2]):
                delta_arr[:, ni, fi] = calc_t_delta(ts, mask[:, ni, fi])
        return delta_arr

    def __getitem__(self, index):
        ti_start = self.start_ts[index]
        # ti_end = ti_start + self.input_len + self.output_len
        ti_input_start, ti_input_end = ti_start, ti_start + self.input_len
        ti_output_start, ti_output_end = ti_start + self.input_len, ti_start + self.input_len + self.output_len
        
        # finest s-t resolution
        # data_input = self.volume_data[ti_input_start:ti_input_end]
        chrono_input = self.chrono_arr[ti_input_start:ti_input_end]
        masked_data_input = self.masked_volume_data[ti_input_start:ti_input_end]
        data_mask_input = self.volume_data_mask[ti_input_start:ti_input_end]

        data_output = self.volume_data[ti_output_start:ti_output_end]
        data_mask_output = np.ones_like(data_output) # dummy mask with all ones
        chrono_output = self.chrono_arr[ti_output_start:ti_output_end]

        ts_input = np.arange(ti_input_end - ti_input_start, dtype=np.float32)
        ts_abs_input = ts_input + ti_input_start
        ts_output = np.arange(ti_output_end - ti_output_start, dtype=np.float32)
        ts_abs_output = ts_output + ti_output_start

        if self.resolution_type == 'agg':
            multires_data_input = []
            for res in self.historical_temp_res:
                sel_res_data, sel_res_start_ts = self._select_multi_temp_res_data_in_t(res, ti_input_start, ti_input_end)
                multires_data_input.append((sel_res_data, sel_res_start_ts))
            
            multires_data_output = []
            for res in self.forecast_temp_res:
                sel_res_data, sel_res_start_ts = self._select_multi_temp_res_data_in_t(res, ti_output_start, ti_output_end)
                multires_data_output.append((sel_res_data, sel_res_start_ts))

            if self.output_format == 'multivar':
                merged_masked_data_input, merged_data_mask_input = self._merge_multires_to_multivar(
                    masked_data_input, ti_input_start, ti_input_end, multires_data_input,
                    data_mask_input
                )
                merged_data_output, merged_data_mask_output = self._merge_multires_to_multivar(
                    data_output, ti_output_start, ti_output_end, multires_data_output,
                    data_mask_output
                )

                ret = {
                    'data_input': merged_masked_data_input,
                    'mask_input': merged_data_mask_input,
                    'chrono_input': chrono_input,
                    'ts_input': ts_input,
                    'ts_abs_input': ts_abs_input,
                    'data_output': merged_data_output,
                    'mask_output': merged_data_mask_output,
                    'chrono_output': chrono_output,
                    'ts_output': ts_output,
                    'ts_abs_output': ts_abs_output,
                    # 'delta_output': self._calc_delta_arr(ts_output, merged_data_mask_output)
                }

                if self.return_delta:
                    ret['delta_input'] = self._calc_delta_arr(ts_input, merged_data_mask_input)
                    delta_output = np.ones_like(merged_data_output, dtype=np.float32)
                    delta_output[0] = 0
                    ret['delta_output'] = delta_output

            elif self.output_format == 'sep_multires':
                merged_masked_data_input, merged_data_mask_input, merged_chrono_input = self._merge_multires_to_list(
                    masked_data_input, multires_data_input,
                    data_mask_input, chrono_input
                )
                merged_data_output, merged_data_mask_output, merged_chrono_output = self._merge_multires_to_list(
                    data_output, multires_data_output,
                    data_mask_output, chrono_output
                )

                ret = {
                    'data_input': merged_masked_data_input,
                    'mask_input': merged_data_mask_input,
                    'chrono_input': merged_chrono_input,
                    'ts_input': ts_input,
                    'ts_abs_input': ts_abs_input,
                    'data_output': merged_data_output,
                    'mask_output': merged_data_mask_output,
                    'chrono_output': merged_chrono_output,
                    'ts_output': ts_output,
                    'ts_abs_output': ts_abs_output,
                    # 'delta_output': self._calc_delta_arr(ts_output, merged_data_mask_output)
                }

            elif self.output_format == 'graph':
                for i in range(len(multires_data_input)):
                    start_t = multires_data_input[i][1][0]
                    multires_data_input[i] = (multires_data_input[i][0], multires_data_input[i][1] - start_t)
                    multires_data_output[i] = (multires_data_output[i][0], multires_data_output[i][1] - start_t)
                ret = {
                    'data_input': masked_data_input,
                    'mask_input': data_mask_input,
                    'multires_data_input': multires_data_input,
                    'chrono_input': chrono_input,
                    'ts_input': ts_input,
                    'ts_abs_input': ts_abs_input,
                    'data_output': data_output,
                    'mask_output': data_mask_output,
                    'multires_data_output': multires_data_output,
                    'chrono_output': chrono_output,
                    'ts_output': ts_output,
                    'ts_abs_output': ts_abs_output,
                }
            else:
                raise NotImplementedError()
        elif self.resolution_type == 'sample':
            if self.output_format == 'multivar':
                # merged_masked_data_input, merged_data_mask_input = masked_data_input, data_mask_input
                merged_masked_data_input, merged_data_mask_input = self._merge_multi_sample_res_to_multivar(
                    masked_data_input, ti_input_start, ti_input_end, self.historical_temp_res,
                    data_mask_input, direction='backward'
                )
                merged_data_output, merged_data_mask_output = self._merge_multi_sample_res_to_multivar(
                    data_output, ti_output_start, ti_output_end, self.forecast_temp_res,
                    data_mask_output, direction='forward'
                )

                ret = {
                    'data_input': merged_masked_data_input,
                    'mask_input': merged_data_mask_input,
                    'chrono_input': chrono_input,
                    'ts_input': ts_input,
                    'ts_abs_input': ts_abs_input,
                    'data_output': merged_data_output,
                    'mask_output': merged_data_mask_output,
                    'chrono_output': chrono_output,
                    'ts_output': ts_output,
                    'ts_abs_output': ts_abs_output,
                    # 'delta_output': self._calc_delta_arr(ts_output, merged_data_mask_output)
                }
            elif self.output_format == 'sep_multires':
                merged_masked_data_input, merged_data_mask_input, merged_chrono_input = self._merge_multi_sample_res_to_list(
                    masked_data_input, ti_input_start, ti_input_end, self.historical_temp_res,
                    data_mask_input, chrono_input, direction='backward'
                )
                merged_data_output, merged_data_mask_output, merged_chrono_output = self._merge_multi_sample_res_to_list(
                    data_output, ti_output_start, ti_output_end, self.forecast_temp_res,
                    data_mask_output, chrono_output, direction='forward'
                )

                ret = {
                    'data_input': merged_masked_data_input,
                    'mask_input': merged_data_mask_input,
                    'chrono_input': merged_chrono_input,
                    'ts_input': ts_input,
                    'ts_abs_input': ts_abs_input,
                    'data_output': merged_data_output,
                    'mask_output': merged_data_mask_output,
                    'chrono_output': merged_chrono_output,
                    'ts_output': ts_output,
                    'ts_abs_output': ts_abs_output,
                    # 'delta_output': self._calc_delta_arr(ts_output, merged_data_mask_output)
                }
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        return ret

    def __len__(self):
        return len(self.start_ts)


if __name__ == '__main__':
    # dataset = NYCTaxiDataset(
    #     save_folder='../../data/nyc_taxi/manhattan',
    #     input_len=1440,
    #     output_len=480,
    #     historical_temp_res=['6-hour', 'day'],
    #     # forecast_temp_res=[],
    #     forecast_temp_res=['6-hour', 'day'],
    #     keep_ratio=0.8,
    #     trainval_ps=(0.6, 0.2),
    #     mask_seed=42,
    #     seq_diff=48,
    #     data_type='train',
    #     scale=True,
    #     scaler=None,
    #     return_delta=False,
    #     output_format='multivar',
    #     resolution_type='sample'
    # )

    # dataset = NYCTaxiDataset(
    #     save_folder='../../data/pems/METR-LA',
    #     input_len=36,
    #     output_len=36,
    #     historical_temp_res=['1-hour'],
    #     # forecast_temp_res=[],
    #     forecast_temp_res=['1-hour'],
    #     keep_ratio=0.8,
    #     trainval_ps=(0.6, 0.2),
    #     mask_seed=42,
    #     seq_diff=1,
    #     data_type='train',
    #     scale=True,
    #     scaler=None,
    #     return_delta=False,
    #     output_format='sep_multires',
    #     resolution_type='sample'
    # )

    # dataset = NYCTaxiDataset(
    #     save_folder='../../data/solar_energy',
    #     input_len=1440,
    #     output_len=480,
    #     historical_temp_res=['6-hour', 'day'],
    #     # forecast_temp_res=[],
    #     forecast_temp_res=['6-hour', 'day'],
    #     keep_ratio=0.8,
    #     trainval_ps=(0.6, 0.2),
    #     mask_seed=42,
    #     seq_diff=48,
    #     data_type='train',
    #     scale=True,
    #     scaler=None,
    #     return_delta=False,
    #     output_format='sep_multires',
    #     # resolution_type='agg'
    #     resolution_type='sample'
    # )

    dataset = NYCTaxiDataset(
        save_folder='../../data/ecl',
        input_len=720,
        output_len=240,
        historical_temp_res=['6-hour', 'day'],
        # forecast_temp_res=[],
        forecast_temp_res=['6-hour', 'day'],
        keep_ratio=0.8,
        trainval_ps=(0.6, 0.2),
        mask_seed=42,
        seq_diff=1,
        data_type='train',
        scale=True,
        scaler=None,
        return_delta=False,
        output_format='sep_multires',
        resolution_type='agg'
    )

    data_sample = dataset[0]
    for k in data_sample:
        try:
            print(k, data_sample[k].shape)
        except:
            print(k, len(data_sample[k]))