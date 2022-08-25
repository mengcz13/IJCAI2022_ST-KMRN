import sys
sys.path.append('..')

from tqdm import tqdm
from dataset.nyc_taxi.nyctaxi_dataset import NYCTaxiDataset
from util import calculate_scaled_laplacian, calculate_normalized_laplacian, sym_adj, asym_adj
from util import StandardScaler
from torch.autograd import Variable
from scipy.sparse import linalg
import torch
import scipy.sparse as sp
import os
import numpy as np
import pickle
from einops import rearrange


def get_blk_ks_nl(input_seq_len):
    blk_ks_nl_dict = {
        1440: [4,4,4],
        120: [4,2,5],
        30: [4,2,3],
        36: [4,2,3],
        3: [4,2,2],
        'default': [4,2,2]
    }
    return blk_ks_nl_dict[input_seq_len]


def chrono_embedding_sincos(chrono_arr, chrono_arr_cats):
    assert chrono_arr.shape[-1] == len(chrono_arr_cats)
    if chrono_arr_cats == 'mdwh':
        periods = np.array([12, 31, 7, 24], dtype=np.float32)
    elif chrono_arr_cats == 'mdwhm':
        periods = np.array([12, 31, 7, 24, 60], dtype=np.float32)
    else:
        raise NotImplementedError()
    t = chrono_arr / periods * 2 * np.pi
    sint, cost = np.sin(t), np.cos(t)
    sincos_emb = np.stack([sint, cost], axis=-1)
    sincos_emb = rearrange(sincos_emb, 't f d -> t (f d)')
    return sincos_emb


class MultiresDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, flag, args):
        super().__init__()

        self.debug = args.debug

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

        if args.target_res == all_avail_temp_res[0]:
            forecast_temp_res = []
        else:
            forecast_temp_res = all_avail_temp_res[1:]

        if args.single_res_input_output:
            historical_temp_res = []
            forecast_temp_res = []
        else:
            historical_temp_res = all_avail_temp_res[1:]

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
            resolution_type=args.resolution_type,
            output_format='sep_multires',
            chrono_arr_cats=args.chrono_arr_cats
        )

        self.scaler = StandardScaler(
            mean=self.dataset.scaler.mean_, std=np.sqrt(self.dataset.scaler.var_))
        
        self.chrono_sincos_emb = args.chrono_sincos_emb

    def __len__(self):
        if self.debug:
            return 17
        else:
            return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_fn(self, batch):
        xs = []
        ys = []
        y_attrs = []
        for idx in range(len(batch)):
            ret = batch[idx]
            data_input_list, mask_input_list, chrono_input_list = ret[
                'data_input'], ret['mask_input'], ret['chrono_input']
            concat_data_input_list = []
            for data_input, mask_input, chrono_input in zip(data_input_list, mask_input_list, chrono_input_list):
                if self.chrono_sincos_emb:
                    chrono_input = chrono_embedding_sincos(chrono_input, self.dataset.chrono_arr_cats)
                chrono_input = np.tile(
                    chrono_input[:, np.newaxis, :], (1, data_input.shape[1], 1))
                x = np.concatenate(
                    [data_input, mask_input, chrono_input], axis=-1)
                concat_data_input_list.append(x)
            y_list = ret['data_output']
            mask_output_list, chrono_output_list = ret['mask_output'], ret['chrono_output']
            concat_data_output_attrs_list = []
            for data_output, mask_output, chrono_output in zip(y_list, mask_output_list, chrono_output_list):
                if self.chrono_sincos_emb:
                    chrono_output = chrono_embedding_sincos(chrono_output, self.dataset.chrono_arr_cats)
                chrono_output = np.tile(chrono_output[:, np.newaxis, :], (1, data_output.shape[1], 1))
                y_attr = np.concatenate([mask_output, chrono_output], axis=-1)
                concat_data_output_attrs_list.append(y_attr)

            # if self.target_res == self.all_avail_temp_res[0]:
            #     pass
            # else:
            #     fd_start, fd_end = _get_res_feature_indices(
            #         y.shape[-1], self.target_res, self.all_avail_temp_res)
            #     y = y[:, :, fd_start:fd_end]

            xs.append(concat_data_input_list)
            ys.append(y_list)
            y_attrs.append(concat_data_output_attrs_list)
        xs = [torch.FloatTensor(np.stack(t, axis=0)) for t in list(zip(*xs))]
        ys = [torch.FloatTensor(np.stack(t, axis=0)) for t in list(zip(*ys))]
        y_attrs = [torch.FloatTensor(np.stack(t, axis=0))for t in list(zip(*y_attrs))]
        return xs, ys, y_attrs


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

    sample_x_list, sample_y_list, sample_y_attrs = next(iter(data['train_loader']))
    data['in_dim'] = sample_x_list[0].shape[-1]
    data['out_dim'] = sample_y_list[0].shape[-1]
    data['attr_dim'] = sample_y_attrs[0].shape[-1]
    data['res_num'] = len(sample_x_list)
    data['x_lens'] = [x.shape[1] for x in sample_x_list]
    data['y_lens'] = [y.shape[1] for y in sample_y_list]
    # blk, ks, nl
    if args.larger_rf:
        data['blk_ks_nl'] = [get_blk_ks_nl(t) for t in data['x_lens']]
    else:
        data['blk_ks_nl'] = [get_blk_ks_nl('default') for _ in data['x_lens']]

    return data


def load_adj(adj_mx, adjtype):
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj
