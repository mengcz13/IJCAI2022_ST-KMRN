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
            chrono_arr_cats=args.chrono_arr_cats,
        )

        self.scaler = StandardScaler(
            mean=self.dataset.scaler.mean_, std=np.sqrt(self.dataset.scaler.var_))

        # adj matrix
        num_nodes = len(self.dataset.all_node_types)
        edges = self.dataset.all_edges
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for idx in range(edges.shape[0]):
            adj[edges[idx][0], edges[idx][1]] = 1
        for i in range(num_nodes):
            adj[i, i] = 1
        self.rel_attrs = torch.FloatTensor(adj).unsqueeze(-1) # [N, N, 1]

        self.chrono_sincos_emb = args.chrono_sincos_emb

    def __len__(self):
        if self.debug:
            return 17
        else:
            return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_fn(self, batch):
        seq_data_list = []
        fit_data_list = []
        for ri in range(len(batch[0]['data_input'])):
            xs = []
            ys = []
            y_attrs = []
            for idx in range(len(batch)):
                ret = batch[idx]
                data_input, mask_input, chrono_input = ret['data_input'][ri], ret['mask_input'][ri], ret['chrono_input'][ri]
                if self.chrono_sincos_emb:
                    chrono_input = chrono_embedding_sincos(chrono_input, self.dataset.chrono_arr_cats)
                chrono_input = np.tile(chrono_input[:, np.newaxis, :], (1, data_input.shape[1], 1))
                # normalize
                # chrono_input /= np.array([12, 31, 7, 24], dtype=np.float32)
                # x = np.concatenate([data_input, mask_input, chrono_input], axis=-1)
                data_output, mask_output, chrono_output = ret['data_output'][ri], ret['mask_output'][ri], ret['chrono_output'][ri]
                if self.chrono_sincos_emb:
                    chrono_output = chrono_embedding_sincos(chrono_output, self.dataset.chrono_arr_cats)
                chrono_output = np.tile(chrono_output[:, np.newaxis, :], (1, data_output.shape[1], 1))
                # chrono_output /= np.array([12, 31, 7, 24], dtype=np.float32)

                data_input, mask_input, chrono_input = torch.FloatTensor(data_input), torch.FloatTensor(mask_input), torch.FloatTensor(chrono_input)
                fake_action_input = torch.zeros_like(data_input)[..., :1]
                data_output, mask_output, chrono_output = torch.FloatTensor(data_output), torch.FloatTensor(mask_output), torch.FloatTensor(chrono_output)
                fake_action_output = torch.zeros_like(data_output)[..., :1]

                rel_attrs_input = self.rel_attrs.unsqueeze(0).expand(data_input.shape[0], -1, -1, -1)
                rel_attrs_output = self.rel_attrs.unsqueeze(0).expand(data_output.shape[0], -1, -1, -1)

                # if self.target_res == self.all_avail_temp_res[0]:
                #     pass
                # else:
                #     fd_start, fd_end = _get_res_feature_indices(y.shape[-1], self.target_res, self.all_avail_temp_res)
                #     y = y[:, :, fd_start:fd_end]

                xs.append([torch.cat([mask_input, chrono_input], dim=-1), data_input, fake_action_input, rel_attrs_input])
                ys.append([torch.cat([mask_output, chrono_output], dim=-1), data_output, fake_action_output, rel_attrs_output])
                # insert the last frame of xs to ys
                for ei in range(len(ys[-1])):
                    ys[-1][ei] = torch.cat([xs[-1][ei][-1:], ys[-1][ei]], dim=0)
            fit_data = [torch.stack(arrs, dim=0) for arrs in zip(*xs)]
            seq_data = [torch.stack(arrs, dim=0) for arrs in zip(*ys)]
            seq_data_list.append(seq_data)
            fit_data_list.append(fit_data)
        x_data_list = fit_data_list
        y_data_list = seq_data_list
        return x_data_list, y_data_list 


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

    data['predefined_A'] = datasets['train'].rel_attrs[..., 0]

    sample_x_data_list, sample_y_data_list = next(iter(data['train_loader']))
    attr_res0, state_res0, action_res0, rel_attr_res0 = sample_x_data_list[0]
    data['in_dim'] = state_res0.shape[-1] + attr_res0.shape[-1]
    data['out_dim'] = state_res0.shape[-1]
    data['res_num'] = len(sample_x_data_list)
    data['x_lens'] = [x[0].shape[1] for x in sample_x_data_list]
    data['y_lens'] = [y[0].shape[1] - 1 for y in sample_y_data_list]
    data['attr_dim'] = attr_res0.shape[-1]
    data['state_dim'] = state_res0.shape[-1]
    data['action_dim'] = action_res0.shape[-1]
    data['relation_dim'] = rel_attr_res0.shape[-1]

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
