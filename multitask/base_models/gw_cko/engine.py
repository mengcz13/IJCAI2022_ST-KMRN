from base_models.gw_cko.CompositionalKoopmanOperators import cko_sysid_and_pred
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import util
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
            loss += lossfunc(pr, rr, 0.0)
            items += 1
    loss /= items
    return loss

class trainer():
    def __init__(self, model, scaler, lrate, wdecay, cko_lrate, cko_wdecay, device, loss_type='masked_mae', cko_lambda_loss_metric=0.3):
        # self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length*out_dim, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16, args=args)
        self.model = model
        self.model.to(device)

        # separate optimizer for DL and Koopman
        cko_params = [p for p in model.cko_models.parameters() if p.requires_grad]
        cko_beta1 = 0.9
        self.cko_optimizer = optim.Adam(cko_params, lr=cko_lrate, betas=(cko_beta1, 0.999), weight_decay=cko_wdecay)
        self.cko_scheduler = ReduceLROnPlateau(self.cko_optimizer, 'min', factor=0.6, patience=2, verbose=True)
        
        dl_params = [p for name, p in model.named_parameters() if (not name.startswith('cko_models.')) and p.requires_grad]
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
        self.cko_lambda_loss_metric = cko_lambda_loss_metric

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
            # DL + CKO model loss
            out_loss = multires_loss(self.loss, out, real)
            # CKO auto encode loss
            cko_auto_encode_loss = sum(output_dict['cko_loss_auto_encode'])
            # CKO pred loss
            cko_prediction_loss = sum(output_dict['cko_loss_prediction'])
            # CKO metric loss
            cko_metric_loss = sum(output_dict['cko_loss_metric'])
            # CKO loss
            cko_loss = cko_auto_encode_loss + cko_prediction_loss + cko_metric_loss * self.cko_lambda_loss_metric
            # upsampling/downsampling loss
            out_ds_loss = multires_loss(self.loss, out_ds, real)
            out_ups_loss = multires_loss(self.loss, out_ups, real)
            out_up_down_s_loss = multires_loss(self.loss, out_up_down_s, real)

            loss = pred_loss + out_loss + cko_loss + out_ds_loss + out_ups_loss + out_up_down_s_loss
            loss.backward()

            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip)
            self.cko_optimizer.step()
            self.dl_optimizer.step()

            mape = sum([util.masked_mape(pr, rr, 0.0).item() for pr, rr in zip(predict, real)])
            rmse = sum([util.masked_rmse(pr, rr, 0.0).item() for pr, rr in zip(predict, real)])
        loss_item_dict = {
            'total_loss': loss.item(), 'loss': pred_loss.item(), 'out_loss': out_loss.item(),
            'cko_loss': cko_loss.item(), 'cko_auto_encode_loss': cko_auto_encode_loss.item(),
            'cko_prediction_loss': cko_prediction_loss.item(), 'cko_metric_loss': cko_metric_loss.item(),
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
            # DL + CKO model loss
            out_loss = multires_loss(self.loss, out, real)
            # CKO auto encode loss
            cko_auto_encode_loss = sum(output_dict['cko_loss_auto_encode'])
            # CKO pred loss
            cko_prediction_loss = sum(output_dict['cko_loss_prediction'])
            # CKO metric loss
            cko_metric_loss = sum(output_dict['cko_loss_metric'])
            # CKO loss
            cko_loss = cko_auto_encode_loss + cko_prediction_loss + cko_metric_loss * self.cko_lambda_loss_metric
            # upsampling/downsampling loss
            out_ds_loss = multires_loss(self.loss, out_ds, real)
            out_ups_loss = multires_loss(self.loss, out_ups, real)
            out_up_down_s_loss = multires_loss(self.loss, out_up_down_s, real)

            loss = pred_loss + out_loss + cko_loss + out_ds_loss + out_ups_loss + out_up_down_s_loss
            mape = sum([util.masked_mape(pr, rr, 0.0).item() for pr, rr in zip(predict, real)])
            rmse = sum([util.masked_rmse(pr, rr, 0.0).item() for pr, rr in zip(predict, real)])
        loss_item_dict = {
            'total_loss': loss.item(), 'loss': pred_loss.item(), 'out_loss': out_loss.item(),
            'cko_loss': cko_loss.item(), 'cko_auto_encode_loss': cko_auto_encode_loss.item(),
            'cko_prediction_loss': cko_prediction_loss.item(), 'cko_metric_loss': cko_metric_loss.item(),
            'out_ds_loss': out_ds_loss.item(), 'out_ups_loss': out_ups_loss.item(),
            'out_up_down_s_loss': out_up_down_s_loss.item()
        }
        return loss_item_dict, mape, rmse
