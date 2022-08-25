from tqdm import tqdm

from util import StandardScaler
from torch.autograd import Variable
from scipy.sparse import linalg
import torch
import scipy.sparse as sp
import os
import numpy as np
import pickle
import sys
sys.path.append('..')
from dataset.nyc_taxi.nyctaxi_dataset import NYCTaxiDataset


class MultiresDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, flag, args):
        super().__init__()

        if 'nyc_taxi' in dataset_dir:
            datatype = 'NYCTaxi'
        elif 'solar_energy' in dataset_dir:
            datatype = 'SolarEnergy'
        elif 'pems' in dataset_dir:
            datatype = 'PeMS'
        elif dataset_dir.endswith('/ecl') or dataset_dir.endswith('/ecl/'):
            datatype = 'ECL'

        if datatype in ['NYCTaxi', 'SolarEnergy']:
            all_avail_temp_res = ['30min', '6-hour', 'day']
        elif datatype in ['PeMS']:
            if (args.seq_in_len < 48) or (args.seq_out_len < 48):
                all_avail_temp_res = ['5min', '1-hour']
            else:
                all_avail_temp_res = ['5min', '1-hour', '4-hour']
        elif datatype in ['ECL']:
            all_avail_temp_res = ['1-hour', '6-hour', 'day']

        # if args.target_res == all_avail_temp_res[0]:
        #     forecast_temp_res = []
        # else:
        #     forecast_temp_res = all_avail_temp_res[1:]

        # if args.single_res_input_output:
        #     historical_temp_res = []
        #     forecast_temp_res = []
        # else:
        #     historical_temp_res = all_avail_temp_res[1:]

        # self.target_res = args.target_res
        self.all_avail_temp_res = all_avail_temp_res

        if flag == 'test':
            seq_diff = args.test_seq_diff
        else:
            seq_diff = args.seq_diff

        self.dataset = NYCTaxiDataset(
            save_folder=dataset_dir,
            input_len=args.seq_in_len,
            output_len=args.seq_out_len,
            historical_temp_res=all_avail_temp_res[1:],
            forecast_temp_res=all_avail_temp_res[1:],
            keep_ratio=0.8,
            trainval_ps=(0.6, 0.2),
            mask_seed=42,
            seq_diff=seq_diff,
            data_type=flag,
            scale=True,
            scaler=None,
            return_delta=False,
            resolution_type=args.resolution_type,
            output_format='sep_multires'
        )

        self.scaler = StandardScaler(
            mean=self.dataset.scaler.mean_, std=np.sqrt(self.dataset.scaler.var_))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_fn(self, batch):
        xs = []
        ys = []
        for idx in range(len(batch)):
            ret = batch[idx]
            data_input_list, mask_input_list, chrono_input_list = ret['data_input'], ret['mask_input'], ret['chrono_input']
            concat_data_input_list = []
            for data_input, mask_input, chrono_input in zip(data_input_list, mask_input_list, chrono_input_list):
                chrono_input = np.tile(
                    chrono_input[:, np.newaxis, :], (1, data_input.shape[1], 1))
                x = np.concatenate([data_input, mask_input, chrono_input], axis=-1)
                concat_data_input_list.append(x)
            y_list = ret['data_output']

            # if self.target_res == self.all_avail_temp_res[0]:
            #     pass
            # else:
            #     fd_start, fd_end = _get_res_feature_indices(
            #         y.shape[-1], self.target_res, self.all_avail_temp_res)
            #     y = y[:, :, fd_start:fd_end]

            xs.append(concat_data_input_list)
            ys.append(y_list)
        xs = [torch.FloatTensor(np.stack(t, axis=0)) for t in list(zip(*xs))]
        ys = [torch.FloatTensor(np.stack(t, axis=0)) for t in list(zip(*ys))]
        return xs, ys


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


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, args=None):
    datasets = {}
    for cat in ['train', 'val', 'test']:
        datasets[cat] = MultiresDataset(dataset_dir, cat, args)

    data = {}
    scaler = datasets['train'].scaler

    data['train_loader'] = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,
                                                       collate_fn=datasets['train'].collate_fn, drop_last=False)
    data['val_loader'] = torch.utils.data.DataLoader(datasets['val'], batch_size=valid_batch_size, shuffle=False,
                                                     collate_fn=datasets['val'].collate_fn, drop_last=False)
    data['test_loader'] = torch.utils.data.DataLoader(datasets['test'], batch_size=test_batch_size, shuffle=False,
                                                      collate_fn=datasets['test'].collate_fn, drop_last=False)
    data['scaler'] = scaler

    num_nodes = len(datasets['train'].dataset.all_node_types)
    edges = datasets['train'].dataset.all_edges
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for idx in range(edges.shape[0]):
        adj[edges[idx][0], edges[idx][1]] = 1
    for i in range(num_nodes):
        adj[i, i] = 1
    data['predefined_A'] = adj

    sample_x_list, sample_y_list = next(iter(data['train_loader']))
    data['in_dim'] = sample_x_list[0].shape[-1]
    data['out_dim'] = sample_y_list[0].shape[-1]
    data['res_num'] = len(sample_x_list)
    data['x_lens'] = [x.shape[1] for x in sample_x_list]
    data['y_lens'] = [y.shape[1] for y in sample_y_list]

    return data
