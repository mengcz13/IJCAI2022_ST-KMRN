from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_NYCTaxi
from data.data_loader import Dataset_NYCTaxi_Multires, Dataset_SolarEnergy_Multires, Dataset_SolarEnergy10min_Multires, Dataset_PeMS_Multires, Dataset_ECL_Multires
from exp.exp_basic import Exp_Basic
from models.model import Informer, GRU

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
from copy import deepcopy

from tqdm import tqdm
import wandb

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        if args.model == 'informer':
            expname = '_'.join(['Informer', args.data])
        elif args.model == 'gru':
            expname = '_'.join(['GRU', args.data])
        if args.expname_suffix != '':
            expname += '_{}'.format(args.expname_suffix)
        wandb.init(project='multires_st', config=args,
               name=expname)
        wandb.watch(self.model)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'gru':GRU
        }
        if self.args.model=='informer':
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.data[:-1],
                self.args.activation,
                self.args.output_attention,
                self.device
            )
        elif self.args.model == 'gru':
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.d_model,
                self.args.e_layers,
                self.args.d_layers
            )
        
        return model.double()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'NYCTaxi_30d-7d-1stride':Dataset_NYCTaxi,
            'NYCTaxi_48h-24h-1stride':Dataset_NYCTaxi,
            'NYCTaxi_30d-7d':Dataset_NYCTaxi,
            'NYCTaxi_48h-24h':Dataset_NYCTaxi,
            'NYCTaxi_7d-7d-1stride':Dataset_NYCTaxi,
            'NYCTaxi_Multires_6h-1d':Dataset_NYCTaxi_Multires,
            'NYCTaxiGreen_Multires_6h-1d':Dataset_NYCTaxi_Multires,
            'NYCTaxiFHV_Multires_6h-1d':Dataset_NYCTaxi_Multires,
            'SolarEnergy_Multires_6h-1d':Dataset_SolarEnergy_Multires,
            'SolarEnergy10min_Multires_1h-6h':Dataset_SolarEnergy10min_Multires,
            'SolarEnergyAr_Multires_6h-1d':Dataset_SolarEnergy_Multires,
            'METR-LA_Multires_1h-4h':Dataset_PeMS_Multires,
            'PEMS-BAY_Multires_1h-4h':Dataset_PeMS_Multires,
            'ECL_Multires_6h-1d':Dataset_ECL_Multires
        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size
        
        if hasattr(self, '{}_data_set'.format(flag)):
            data_set = getattr(self, '{}_data_set'.format(flag))
        else:
            # if Data is Dataset_NYCTaxi_Multires:
            if 'Multires' in self.args.data:
                if flag == 'test':
                    seq_diff = args.test_seq_diff
                else:
                    seq_diff = args.seq_diff
                data_set = Data(
                    root_path=args.root_path,
                    data_path=args.data_path,
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    seq_diff=seq_diff,
                    target_res=args.target_res,
                    keep_ratio=args.keep_ratio,
                    with_ts_delta=args.with_ts_delta,
                    with_input_mask=args.with_input_mask,
                    single_input_res=args.single_input_res,
                    resolution_type=args.resolution_type,
                    chrono_arr_cats=args.chrono_arr_cats
                )
            else:
                data_set = Data(
                    root_path=args.root_path,
                    data_path=args.data_path,
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features
                )
            setattr(self, '{}_data_set'.format(flag), data_set)
        print(flag, len(data_set))

        args.seq_len = data_set.seq_len
        args.label_len = data_set.label_len
        args.pred_len = data_set.pred_len

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in tqdm(enumerate(vali_loader), total=len(vali_loader)):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double()
            
            batch_x_mark = batch_x_mark.double().to(self.device)
            batch_y_mark = batch_y_mark.double().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
            # encoder - decoder
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true) 

            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
        
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join('checkpoints', wandb.run.id)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in tqdm(enumerate(train_loader), total=len(train_loader)):
                iter_count += 1
                
                model_optim.zero_grad()
                
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double()
                
                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
                dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    wandb.log({'Epoch': epoch + 1, 'Step Train Loss': loss.item()})
                
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            wandb.log({'Epoch': epoch + 1, 'Train Loss': train_loss, 'Vali Loss': vali_loss, 'Test Loss': test_loss})
            early_stopping(vali_loss, self.model, path)

            curr_state_dict = deepcopy(self.model.state_dict())
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            self.test(setting)
            self.model.load_state_dict(curr_state_dict)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        if self.args.output_attention:
            attns = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double()
            batch_x_mark = batch_x_mark.double().to(self.device)
            batch_y_mark = batch_y_mark.double().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
            # encoder - decoder
            if self.args.output_attention:
                outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)
            
            pred = outputs.detach().cpu().numpy()#.squeeze()
            true = batch_y.detach().cpu().numpy()#.squeeze()
            
            preds.append(pred)
            trues.append(true)

            if self.args.output_attention:
                attn = [a.detach().cpu().numpy() for a in attn]
                attns.append(attn)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if self.args.output_attention:
            layer_attns = []
            for battn in zip(*attns):
                layer_attns.append(np.concatenate(battn, axis=0))
            for l_i, la in enumerate(layer_attns):
                print('attn at layer{} shape:'.format(l_i), la.shape)
        
        # result save
        folder_path = os.path.join('results', wandb.run.id)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        wandb.log({
            'Test MSE': mse, 'Test MAE': mae
        })

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        if hasattr(test_data, 'scaler'):
            if hasattr(test_data.scaler, 'mean_'):
                np.savez(os.path.join(folder_path, 'scaler.npz'), mean=test_data.scaler.mean_, var=test_data.scaler.var_)

        if self.args.output_attention:
            attn_dict = {}
            for li, la in enumerate(layer_attns):
                attn_dict['layer_{}'.format(li)] = la
            np.savez_compressed(os.path.join(folder_path, 'attn.npz'), **attn_dict)

        return