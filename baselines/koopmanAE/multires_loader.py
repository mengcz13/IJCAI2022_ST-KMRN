import os
import sys
sys.path.append('../..')

import numpy as np
import torch

from dataset.nyc_taxi.nyctaxi_dataset import NYCTaxiDataset


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MultiresDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, flag, args):
        super().__init__()

        if 'nyc_taxi' in dataset_dir:
            datatype = 'NYCTaxi'
        elif 'solar_energy_10min' in dataset_dir:
            datatype = 'SolarEnergy10min'
        elif 'solar_energy' in dataset_dir:
            datatype = 'SolarEnergy'
        elif 'pems' in dataset_dir:
            datatype = 'PeMS'

        if datatype in ['NYCTaxi', 'SolarEnergy']:
            all_avail_temp_res = ['30min', '6-hour', 'day']
        elif datatype in ['SolarEnergy10min']:
            all_avail_temp_res = ['10min', '1-hour', '6-hour']
        elif datatype in ['PeMS']:
            if (args.seq_in_len < 48) or (args.seq_out_len < 48):
                all_avail_temp_res = ['5min', '1-hour']
            else:
                all_avail_temp_res = ['5min', '1-hour', '4-hour']

        if args.target_res == all_avail_temp_res[0]:
            forecast_temp_res = []
        else:
            forecast_temp_res = all_avail_temp_res[1:]

        if args.single_res_input_output:
            historical_temp_res = []
            forecast_temp_res = []
        else:
            historical_temp_res = all_avail_temp_res[1:]

        self.target_res = args.target_res
        self.all_avail_temp_res = all_avail_temp_res

        if flag == 'test':
            seq_diff = args.test_seq_diff
        else:
            seq_diff = args.seq_diff

        self.dataset = NYCTaxiDataset(
            save_folder=dataset_dir,
            input_len=args.seq_in_len,
            output_len=args.seq_out_len,
            historical_temp_res=historical_temp_res,
            forecast_temp_res=forecast_temp_res,
            keep_ratio=args.keep_ratio,
            trainval_ps=(0.6, 0.2),
            mask_seed=42,
            seq_diff=seq_diff,
            data_type=flag,
            scale=True,
            scaler=None,
            return_delta=False,
            resolution_type=args.resolution_type
        )

        self.scaler = StandardScaler(mean=self.dataset.scaler.mean_, std=np.sqrt(self.dataset.scaler.var_))

        # adj matrix
        num_nodes = len(self.dataset.all_node_types)
        edges = self.dataset.all_edges
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for idx in range(edges.shape[0]):
            adj[edges[idx][0], edges[idx][1]] = 1
        for i in range(num_nodes):
            adj[i, i] = 1
        self.rel_attrs = torch.FloatTensor(adj).unsqueeze(-1) # [N, N, 1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_fn(self, batch):
        xs = []
        ys = []
        for idx in range(len(batch)):
            ret = batch[idx]
            data_input, mask_input, chrono_input = ret['data_input'], ret['mask_input'], ret['chrono_input']
            chrono_input = np.tile(chrono_input[:, np.newaxis, :], (1, data_input.shape[1], 1))
            # x = np.concatenate([data_input, mask_input, chrono_input], axis=-1)
            data_output, mask_output, chrono_output = ret['data_output'], ret['mask_output'], ret['chrono_output']
            chrono_output = np.tile(chrono_output[:, np.newaxis, :], (1, data_output.shape[1], 1))

            data_input, mask_input, chrono_input = torch.FloatTensor(data_input), torch.FloatTensor(mask_input), torch.FloatTensor(chrono_input)
            fake_action_input = torch.zeros_like(data_input)[..., :1]
            data_output, mask_output, chrono_output = torch.FloatTensor(data_output), torch.FloatTensor(mask_output), torch.FloatTensor(chrono_output)
            fake_action_output = torch.zeros_like(data_output)[..., :1]

            rel_attrs_input = self.rel_attrs.unsqueeze(0).expand(data_input.shape[0], -1, -1, -1)
            rel_attrs_output = self.rel_attrs.unsqueeze(0).expand(data_output.shape[0], -1, -1, -1)

            if self.target_res == self.all_avail_temp_res[0]:
                pass
            else:
                fd_start, fd_end = _get_res_feature_indices(data_output.shape[-1], self.target_res, self.all_avail_temp_res)
                data_output = data_output[:, :, fd_start:fd_end]

            xs.append([torch.cat([mask_input, chrono_input], dim=-1), data_input, fake_action_input, rel_attrs_input])
            ys.append([torch.cat([mask_output, chrono_output], dim=-1), data_output, fake_action_output, rel_attrs_output])
            # insert the last frame of xs to ys
            for ei in range(len(ys[-1])):
                ys[-1][ei] = torch.cat([xs[-1][ei][-1:], ys[-1][ei]], dim=0)
        fit_data = [torch.stack(arrs, dim=0) for arrs in zip(*xs)]
        seq_data = [torch.stack(arrs, dim=0) for arrs in zip(*ys)]
        return seq_data, fit_data 


def _get_res_feature_indices(y_true_fdim, target_res, all_avail_temp_res):
    assert y_true_fdim % len(all_avail_temp_res) == 0
    mult = y_true_fdim // len(all_avail_temp_res)
    if target_res == 'all':
        d_start, d_end = 0, len(all_avail_temp_res)
    else:
        assert target_res in all_avail_temp_res
        d_start = all_avail_temp_res.index(target_res)
        d_end = d_start + 1
    return d_start * mult, d_end * mult