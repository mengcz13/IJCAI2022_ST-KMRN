import torch
import torch.nn as nn
import torch.optim as optim
import util


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
    def __init__(self, model, scaler, lrate, wdecay, device, loss_type='masked_mae'):
        # self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length*out_dim, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16, args=args)
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=10, verbose=True)
        if loss_type == 'masked_mae':
            self.loss = util.masked_mae
        elif loss_type == 'masked_mse':
            self.loss = util.masked_mse
        else:
            raise NotImplementedError('Loss {} not supported!'.format(loss_type))
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, yattr):
        with torch.enable_grad():
            self.model.train()
            self.optimizer.zero_grad()
            input = [nn.functional.pad(x.transpose(1, 3), (1, 0, 0, 0)).transpose(1, 3) for x in input]
            output_dict = self.model(input, yattr)
            predict = output_dict['pred']
            out, out_ds, out_ups, out_up_down_s = output_dict['out'], output_dict['out_ds'], output_dict['out_ups'], output_dict['out_up_down_s']
            real = real_val

            # pred loss
            pred_loss = multires_loss(self.loss, predict, real)
            out_loss = multires_loss(self.loss, out, real)
            out_ds_loss = multires_loss(self.loss, out_ds, real)
            out_ups_loss = multires_loss(self.loss, out_ups, real)
            out_up_down_s_loss = multires_loss(self.loss, out_up_down_s, real)

            loss = pred_loss + out_loss + out_ds_loss + out_ups_loss + out_up_down_s_loss
            loss.backward()

            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip)
            self.optimizer.step()
            mape = sum([util.masked_mape(pr, rr, 0.0).item() for pr, rr in zip(predict, real)])
            rmse = sum([util.masked_rmse(pr, rr, 0.0).item() for pr, rr in zip(predict, real)])
        loss_item_dict = {
            'total_loss': loss.item(), 'loss': pred_loss.item(), 'out_loss': out_loss.item(),
            'out_ds_loss': out_ds_loss.item(), 'out_ups_loss': out_ups_loss.item(),
            'out_up_down_s_loss': out_up_down_s_loss.item()
        }
        return loss_item_dict, mape, rmse

    def eval(self, input, real_val, yattr):
        with torch.no_grad():
            self.model.eval()
            input = [nn.functional.pad(x.transpose(1, 3), (1, 0, 0, 0)).transpose(1, 3) for x in input]
            output_dict = self.model(input, yattr)
            predict = output_dict['pred']
            out, out_ds, out_ups, out_up_down_s = output_dict['out'], output_dict['out_ds'], output_dict['out_ups'], output_dict['out_up_down_s']
            real = real_val

            pred_loss = multires_loss(self.loss, predict, real)
            out_loss = multires_loss(self.loss, out, real)
            out_ds_loss = multires_loss(self.loss, out_ds, real)
            out_ups_loss = multires_loss(self.loss, out_ups, real)
            out_up_down_s_loss = multires_loss(self.loss, out_up_down_s, real)

            loss = pred_loss + out_loss + out_ds_loss + out_ups_loss + out_up_down_s_loss
            mape = sum([util.masked_mape(pr, rr, 0.0).item() for pr, rr in zip(predict, real)])
            rmse = sum([util.masked_rmse(pr, rr, 0.0).item() for pr, rr in zip(predict, real)])
        loss_item_dict = {
            'total_loss': loss.item(), 'loss': pred_loss.item(), 'out_loss': out_loss.item(),
            'out_ds_loss': out_ds_loss.item(), 'out_ups_loss': out_ups_loss.item(),
            'out_up_down_s_loss': out_up_down_s_loss.item()
        }
        return loss_item_dict, mape, rmse
