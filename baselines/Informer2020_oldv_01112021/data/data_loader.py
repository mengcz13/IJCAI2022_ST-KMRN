import sys
sys.path.append('../..') # [proj] root dir

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from dataset.nyc_taxi.nyctaxi_dataset import NYCTaxiDataset

import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True):
        # size [seq_len, label_len pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            data = scaler.fit_transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
        df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
        data_stamp = df_stamp.drop(['date'],1).values
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True):
        # size [seq_len, label_len pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            data = scaler.fit_transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
        df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row:row.minute,1)
        df_stamp['minute'] = df_stamp.minute.map(lambda x:x//15)
        data_stamp = df_stamp.drop(['date'],1).values
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1


class Dataset_NYCTaxi(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features=None, data_path='sampled_dense_multires_seqs_30d-7d_diff1d.npz', 
                 target=None, scale=True):
        datanpz = np.load(os.path.join(root_path, data_path), allow_pickle=True)
        if datanpz['{}_seq'.format(flag)].shape[1] > 1:
            self.node_i = 2
        else:
            self.node_i = 0
        _seq = np.stack(datanpz['{}_seq'.format(flag)][:, self.node_i, 0], axis=0) # N x T x F
        _time_abs = np.stack(datanpz['{}_time_abs'.format(flag)][:, self.node_i, 0], axis=0) # N x T
        _time = datanpz['{}_time'.format(flag)][0, 0, 0]
        input_ts_max = datanpz['input_ts_max'].item()
        self.seq_len = input_ts_max + 1
        self.pred_len = len(_time) - self.seq_len
        self.label_len = self.pred_len
        print('seq_len, pred_len, label_len:', self.seq_len, self.pred_len, self.label_len)

        self._seq = _seq
        self._time_abs = _time_abs

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.__process_data__()

    def __process_data__(self):
        scaler = StandardScaler()
        if self.scale:
            data = scaler.fit_transform(self._seq.reshape(-1, self._seq.shape[-1]))
            data = data.reshape(self._seq.shape[0], self._seq.shape[1], -1)
            print('mean:', scaler.mean_)
            print('var:', scaler.var_)
        else:
            data = self._seq
        self.scaler = scaler

        df_stamp = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.float32)
        start_date = datetime(year=2017, month=1, day=1)
        for si in range(df_stamp.shape[0]):
            for ti in range(df_stamp.shape[1]):
                date_st = start_date + timedelta(hours=self._time_abs[si, ti]*0.5)
                df_stamp[si, ti, 0] = date_st.month
                df_stamp[si, ti, 1] = date_st.day
                df_stamp[si, ti, 2] = date_st.weekday()
                df_stamp[si, ti, 3] = date_st.hour
            
        self.data_x = data[:, :self.seq_len, :]
        self.data_y = data[:, self.seq_len-self.label_len:, :]
        self.data_stamp_x = df_stamp[:, :self.seq_len, :]
        self.data_stamp_y = df_stamp[:, self.seq_len-self.label_len:, :]
    
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], self.data_stamp_x[index], self.data_stamp_y[index]
    
    def __len__(self):
        return self.data_x.shape[0]


class Dataset_Base_Multires(Dataset):
    def __init__(self, root_path, flag, size, 
                 features, data_path, 
                 target, scale, seq_diff, target_res, keep_ratio,
                 with_ts_delta, with_input_mask, single_input_res,
                 all_avail_temp_res,
                 resolution_type,
                 chrono_arr_cats):
        # size [seq_len, label_len pred_len]
        # info
        if size == None:
            raise NotImplementedError()
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        
        self.all_avail_temp_res = list(all_avail_temp_res)

        # if target_res == self.all_avail_temp_res[0]:
        #     forecast_temp_res = []
        # else:
        #     # forecast_temp_res = ['6-hour', 'day']
        forecast_temp_res = self.all_avail_temp_res[1:]

        self.dataset = NYCTaxiDataset(
            save_folder=root_path, input_len=self.seq_len, output_len=self.pred_len,
            historical_temp_res=self.all_avail_temp_res[1:],
            forecast_temp_res=forecast_temp_res,
            keep_ratio=keep_ratio,
            trainval_ps=(0.6, 0.2),
            mask_seed=42,
            seq_diff=seq_diff,
            data_type=flag,
            scale=scale,
            scaler=None,
            return_delta=with_ts_delta,
            resolution_type=resolution_type,
            chrono_arr_cats=chrono_arr_cats
        )

        self.target_res = target_res
        self.with_ts_delta = with_ts_delta
        self.with_input_mask = with_input_mask
        self.single_input_res = single_input_res

    def _get_res_feature_indices(self, y_true_fdim, target_res, all_avail_temp_res):
        assert y_true_fdim % len(all_avail_temp_res) == 0
        mult = y_true_fdim // len(all_avail_temp_res)
        if self.target_res == 'all':
            d_start, d_end = 0, len(all_avail_temp_res)
        else:
            assert target_res in all_avail_temp_res
            d_start = all_avail_temp_res.index(target_res)
            d_end = d_start + 1
        return d_start * mult, d_end * mult
    
    def __getitem__(self, index):
        sample = self.dataset[index]

        seq_x_concat_list = [sample['data_input']]
        if self.with_input_mask:
            seq_x_concat_list.append(sample['mask_input'])
        if self.with_ts_delta:
            seq_x_concat_list.append(sample['delta_input'])

        # if self.with_ts_delta:
        #     seq_x = np.concatenate([
        #         sample['data_input'], sample['mask_input'], sample['delta_input']
        #     ], axis=-1)
        # else:
        #     seq_x = np.concatenate([
        #         sample['data_input'], sample['mask_input']
        #     ], axis=-1)

        seq_y = sample['data_output']
        seq_x_mark = sample['chrono_input']
        seq_y_mark = sample['chrono_output']

        if self.label_len > 0:
            seq_y = np.concatenate([
                sample['data_input'][-self.label_len:, :, :seq_y.shape[-1]], seq_y
            ], axis=0)
            seq_y_mark = np.concatenate([
                seq_x_mark[-self.label_len:], seq_y_mark
            ], axis=0)

        # if self.target_res != '30min':
        #     if seq_y.shape[-1] == 3:
        #         mult = 1
        #     elif seq_y.shape[-1] == 6:
        #         mult = 2
        #     else:
        #         raise NotImplementedError()
        #     if self.target_res == 'day':
        #         d_start, d_end = 2, 3
        #     elif self.target_res == '30min':
        #         d_start, d_end = 0, 1
        #     elif self.target_res == '6-hour':
        #         d_start, d_end = 1, 2
        #     elif self.target_res == 'all':
        #         d_start, d_end = 0, 3
        #     else:
        #         raise NotImplementedError()

        #     seq_y = seq_y[:, :, d_start*mult:d_end*mult]

        # if self.target_res == self.all_avail_temp_res[0]:
        #     pass
        # else:

        fd_start, fd_end = self._get_res_feature_indices(seq_y.shape[-1], self.target_res, self.all_avail_temp_res)
        seq_y = seq_y[:, :, fd_start:fd_end]
        
        if self.single_input_res:
            seq_x_concat_list = [x[:, :, fd_start:fd_end] for x in seq_x_concat_list]
        seq_x = np.concatenate(seq_x_concat_list, axis=-1)

        # concate features of all nodes
        seq_x = seq_x.reshape(seq_x.shape[0], -1)
        seq_y = seq_y.reshape(seq_y.shape[0], -1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.dataset)


class Dataset_NYCTaxi_Multires(Dataset_Base_Multires):
    def __init__(self, root_path, flag='train', size=None, 
                 features=None, data_path=None, 
                 target=None, scale=True, seq_diff=1, target_res='30min', keep_ratio=0.8,
                 with_ts_delta=False, with_input_mask=False, single_input_res=False, resolution_type='agg', chrono_arr_cats='mdwh'):
        super().__init__(root_path, flag=flag, size=size, features=features, data_path=data_path, 
            target=target, scale=scale, seq_diff=seq_diff, target_res=target_res, 
            keep_ratio=keep_ratio, with_ts_delta=with_ts_delta, with_input_mask=with_input_mask, single_input_res=single_input_res,
            all_avail_temp_res=('30min', '6-hour', 'day'), resolution_type=resolution_type, chrono_arr_cats=chrono_arr_cats)
            
            
class Dataset_SolarEnergy_Multires(Dataset_Base_Multires):
    def __init__(self, root_path, flag='train', size=None, 
                 features=None, data_path=None, 
                 target=None, scale=True, seq_diff=1, target_res='30min', keep_ratio=0.8,
                 with_ts_delta=False, with_input_mask=False, single_input_res=False, resolution_type='agg', chrono_arr_cats='mdwh'):
        super().__init__(root_path, flag=flag, size=size, features=features, data_path=data_path, 
            target=target, scale=scale, seq_diff=seq_diff, target_res=target_res, 
            keep_ratio=keep_ratio, with_ts_delta=with_ts_delta, with_input_mask=with_input_mask, single_input_res=single_input_res, 
            all_avail_temp_res=('30min', '6-hour', 'day'), resolution_type=resolution_type, chrono_arr_cats=chrono_arr_cats)
            
            
class Dataset_SolarEnergy10min_Multires(Dataset_Base_Multires):
    def __init__(self, root_path, flag='train', size=None, 
                 features=None, data_path=None, 
                 target=None, scale=True, seq_diff=1, target_res='10min', keep_ratio=0.8,
                 with_ts_delta=False, with_input_mask=False, single_input_res=False, resolution_type='agg', chrono_arr_cats='mdwh'):
        super().__init__(root_path, flag=flag, size=size, features=features, data_path=data_path, 
            target=target, scale=scale, seq_diff=seq_diff, target_res=target_res, 
            keep_ratio=keep_ratio, with_ts_delta=with_ts_delta, with_input_mask=with_input_mask, single_input_res=single_input_res, 
            all_avail_temp_res=('10min', '1-hour', '6-hour'), resolution_type=resolution_type, chrono_arr_cats=chrono_arr_cats)




class Dataset_PeMS_Multires(Dataset_Base_Multires):
    def __init__(self, root_path, flag='train', size=None, 
                 features=None, data_path=None, 
                 target=None, scale=True, seq_diff=1, target_res='5min', keep_ratio=0.8,
                 with_ts_delta=False, with_input_mask=False, single_input_res=False, resolution_type='agg', chrono_arr_cats='mdwh'):
        seq_in_len = size[0]
        seq_out_len = size[2]
        if (seq_in_len < 48) or (seq_out_len < 48):
            all_avail_temp_res = ['5min', '1-hour']
        else:
            all_avail_temp_res = ['5min', '1-hour', '4-hour']
        super().__init__(root_path, flag=flag, size=size, features=features, data_path=data_path, 
            target=target, scale=scale, seq_diff=seq_diff, target_res=target_res, 
            keep_ratio=keep_ratio, with_ts_delta=with_ts_delta, with_input_mask=with_input_mask, single_input_res=single_input_res, 
            all_avail_temp_res=tuple(all_avail_temp_res), resolution_type=resolution_type, chrono_arr_cats=chrono_arr_cats)


class Dataset_ECL_Multires(Dataset_Base_Multires):
    def __init__(self, root_path, flag='train', size=None, 
                 features=None, data_path=None, 
                 target=None, scale=True, seq_diff=1, target_res='1-hour', keep_ratio=0.8,
                 with_ts_delta=False, with_input_mask=False, single_input_res=False, resolution_type='agg', chrono_arr_cats='mdwh'):
        super().__init__(root_path, flag=flag, size=size, features=features, data_path=data_path, 
            target=target, scale=scale, seq_diff=seq_diff, target_res=target_res, 
            keep_ratio=keep_ratio, with_ts_delta=with_ts_delta, with_input_mask=with_input_mask, single_input_res=single_input_res, 
            all_avail_temp_res=('1-hour', '6-hour', 'day'), resolution_type=resolution_type, chrono_arr_cats=chrono_arr_cats)