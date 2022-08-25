import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import util
from torch.optim.lr_scheduler import ReduceLROnPlateau
from base_models.gw_kae.kae_model import kae_consist_lossfunc


def move_data_list(data_list, device):
    return [[ts.to(device) for ts in res_data] for res_data in data_list]


def multires_loss(lossfunc, predict, real):
    if predict is None:
        return torch.zeros(1, device=real[0].device)
    loss = 0
    items = 0
    for pr, rr in zip(predict, real):
        if pr is None:
            continue
        else:
            try:
                loss += lossfunc(pr, rr, 0.0)
            except:
                loss += lossfunc(pr, rr)
            items += 1
    loss /= items
    return loss

class trainer():
    def __init__(self, model, scaler, lrate, wdecay, device, loss_type='masked_mae', kae_lamb=1, kae_nu=1e-1, kae_eta=1e-2, kae_lr=1e-3, kae_wd=0.0, kae_gradclip=0.05, kae_lr_decay=0.2):
        # self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length*out_dim, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16, args=args)
        self.model = model
        self.model.to(device)

        # separate optimizer for DL and Koopman
        kae_params = [p for p in model.kae_models.parameters() if p.requires_grad]
        self.kae_params = kae_params
        self.kae_optimizer = optim.AdamW(kae_params, lr=kae_lr, weight_decay=kae_wd)
        self.kae_scheduler = ReduceLROnPlateau(self.kae_optimizer, 'min', factor=kae_lr_decay, patience=2, verbose=True)
        self.kae_gradclip = kae_gradclip
        
        dl_params = [p for name, p in model.named_parameters() if (not name.startswith('kae_models.')) and p.requires_grad]
        self.dl_params = dl_params
        self.dl_optimizer = optim.Adam(
            dl_params, lr=lrate, weight_decay=wdecay)
       
        if loss_type == 'masked_mae':
            self.loss = util.masked_mae
        elif loss_type == 'masked_mse':
            self.loss = util.masked_mse
        else:
            raise NotImplementedError('Loss {} not supported!'.format(loss_type))
        self.scaler = scaler
        self.clip = 5

        self.kae_lamb = kae_lamb
        self.kae_nu = kae_nu
        self.kae_eta = kae_eta

    def train(self, x_data_list, y_data_list):
        with torch.enable_grad():
            self.model.train()
            self.model.zero_grad()
            # input = [nn.functional.pad(x.transpose(1, 3), (1, 0, 0, 0)).transpose(1, 3) for x in input]
            output_dict = self.model(x_data_list, y_data_list)
            predict = output_dict['pred']
            out, out_ds, out_ups, out_up_down_s = output_dict['out'], output_dict['out_ds'], output_dict['out_ups'], output_dict['out_up_down_s']
            real = [y[1][:, 1:] for y in y_data_list] 

            # pred loss
            pred_loss = multires_loss(self.loss, predict, real)
            # DL + KAE model loss
            out_loss = multires_loss(self.loss, out, real)

            kae_criterion = nn.MSELoss()
            # KAE forward loss
            kae_loss_fwd = multires_loss(kae_criterion, output_dict['kae_preds'], real)
            # KAE identity loss
            init_states = [y[1][:, :1] for y in y_data_list]
            kae_loss_identity = multires_loss(kae_criterion, output_dict['kae_identity_init_state'], init_states)
            # KAE backward loss
            init_and_real = [y[1] for y in y_data_list]
            kae_loss_bwd = multires_loss(kae_criterion, output_dict['kae_bwds'], init_and_real)
            # KAE consist loss
            kae_loss_consist = 0
            for kae_model in self.model.kae_models:
                kae_loss_consist += kae_consist_lossfunc(
                    A=kae_model.dynamics.dynamics.weight,
                    B=kae_model.backdynamics.dynamics.weight
                )
            kae_loss = kae_loss_fwd + self.kae_lamb * kae_loss_identity + self.kae_nu * kae_loss_bwd + self.kae_eta * kae_loss_consist

            # upsampling/downsampling loss
            out_ds_loss = multires_loss(self.loss, out_ds, real)
            out_ups_loss = multires_loss(self.loss, out_ups, real)
            out_up_down_s_loss = multires_loss(self.loss, out_up_down_s, real)

            loss = pred_loss + out_loss + kae_loss + out_ds_loss + out_ups_loss + out_up_down_s_loss
            loss.backward()

            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.dl_params, self.clip)
            torch.nn.utils.clip_grad_norm_(self.kae_params, self.kae_gradclip)
            self.kae_optimizer.step()
            self.dl_optimizer.step()

            mape = sum([util.masked_mape(pr, rr, 0.0).item() for pr, rr in zip(predict, real)])
            rmse = sum([util.masked_rmse(pr, rr, 0.0).item() for pr, rr in zip(predict, real)])
        loss_item_dict = {
            'total_loss': loss.item(), 'loss': pred_loss.item(), 'out_loss': out_loss.item(),
            'kae_loss': kae_loss.item(), 'kae_loss_fwd': kae_loss_fwd.item(), 'kae_loss_identity': kae_loss_identity.item(),
            'kae_loss_bwd': kae_loss_bwd.item(), 'kae_loss_consist': kae_loss_consist.item(),
            'out_ds_loss': out_ds_loss.item(), 'out_ups_loss': out_ups_loss.item(),
            'out_up_down_s_loss': out_up_down_s_loss.item()
        }
        return loss_item_dict, mape, rmse

    def eval(self, x_data_list, y_data_list):
        with torch.no_grad():
            self.model.eval()
            # input = [nn.functional.pad(x.transpose(1, 3), (1, 0, 0, 0)).transpose(1, 3) for x in input]
            output_dict = self.model(x_data_list, y_data_list)
            predict = output_dict['pred']
            out, out_ds, out_ups, out_up_down_s = output_dict['out'], output_dict['out_ds'], output_dict['out_ups'], output_dict['out_up_down_s']
            real = [y[1][:, 1:] for y in y_data_list]
            
            # pred loss
            pred_loss = multires_loss(self.loss, predict, real)
            # DL + KAE model loss
            out_loss = multires_loss(self.loss, out, real)

            kae_criterion = nn.MSELoss()
            # KAE forward loss
            kae_loss_fwd = multires_loss(kae_criterion, output_dict['kae_preds'], real)
            # KAE identity loss
            init_states = [y[1][:, :1] for y in y_data_list]
            kae_loss_identity = multires_loss(kae_criterion, output_dict['kae_identity_init_state'], init_states)
            # KAE backward loss
            init_and_real = [y[1] for y in y_data_list]
            kae_loss_bwd = multires_loss(kae_criterion, output_dict['kae_bwds'], init_and_real)
            # KAE consist loss
            kae_loss_consist = 0
            for kae_model in self.model.kae_models:
                kae_loss_consist += kae_consist_lossfunc(
                    A=kae_model.dynamics.dynamics.weight,
                    B=kae_model.backdynamics.dynamics.weight
                )
            kae_loss = kae_loss_fwd + self.kae_lamb * kae_loss_identity + self.kae_nu * kae_loss_bwd + self.kae_eta * kae_loss_consist

            # upsampling/downsampling loss
            out_ds_loss = multires_loss(self.loss, out_ds, real)
            out_ups_loss = multires_loss(self.loss, out_ups, real)
            out_up_down_s_loss = multires_loss(self.loss, out_up_down_s, real)

            loss = pred_loss + out_loss + kae_loss + out_ds_loss + out_ups_loss + out_up_down_s_loss
            mape = sum([util.masked_mape(pr, rr, 0.0).item() for pr, rr in zip(predict, real)])
            rmse = sum([util.masked_rmse(pr, rr, 0.0).item() for pr, rr in zip(predict, real)])
        loss_item_dict = {
            'total_loss': loss.item(), 'loss': pred_loss.item(), 'out_loss': out_loss.item(),
            'kae_loss': kae_loss.item(), 'kae_loss_fwd': kae_loss_fwd.item(), 'kae_loss_identity': kae_loss_identity.item(),
            'kae_loss_bwd': kae_loss_bwd.item(), 'kae_loss_consist': kae_loss_consist.item(),
            'out_ds_loss': out_ds_loss.item(), 'out_ups_loss': out_ups_loss.item(),
            'out_up_down_s_loss': out_up_down_s_loss.item()
        }
        return loss_item_dict, mape, rmse
