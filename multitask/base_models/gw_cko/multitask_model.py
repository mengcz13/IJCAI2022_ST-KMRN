import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.nn.modules.activation import Sigmoid

from modules.upsampler import SimpleUpsampler, SimpleDownsampler, UnetUpsampler
from base_models.gw_cko.CompositionalKoopmanOperators import CompositionalKoopmanOperators
from base_models.gw_cko.CompositionalKoopmanOperators import cko_sysid_and_pred


class MultitaskModel(nn.Module):
    def __init__(self,
        input_dim, output_dim, res_seq_lens,
        enc_base_model_list, dec_base_model_list,
        self_attn_embed_dim, self_attn_num_heads,
        attr_dim, state_dim, action_dim, relation_dim,
        fit_type, g_dim, nf_particle, nf_relation, nf_effect, residual, pstep, I_factor, enc_dec_type, nodenum,
        cko_dl_gate,
        cko_dl_gate_hdim,
        dropout=0.1,
        use_downsampling_pred=False,
        use_upsampling_pred=False,
        upsampler='simple',
        updown_fusion='param',
        agg_res_hdim=32,
        downsampler_agg='mean',
        with_multitask_attn=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.res_seq_lens = res_seq_lens
        self.use_downsampling_pred = use_downsampling_pred
        self.use_upsampling_pred = use_upsampling_pred
        self.upsampler = upsampler
        self.updown_fusion = updown_fusion
        self.agg_res_hdim = agg_res_hdim

        self.enc_base_model_list = nn.ModuleList(enc_base_model_list)
        self.dec_base_model_list = nn.ModuleList(dec_base_model_list)
        assert len(self.enc_base_model_list) == len(self.dec_base_model_list)

        # one cko model for each resolution
        self.cko_models = nn.ModuleList([
            CompositionalKoopmanOperators(
                attr_dim, state_dim, action_dim, relation_dim,
                fit_type, g_dim, nf_particle, nf_relation, nf_effect, nodenum,
                residual, enc_dec_type
            ) for _ in range(len(self.res_seq_lens))
        ])
        self.pstep = pstep
        self.I_factor = I_factor

        # gate for combining CKO results and DL results
        self.cko_dl_gate_flag = cko_dl_gate
        if self.cko_dl_gate_flag:
            self.cko_dl_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels=(2 * state_dim + attr_dim), out_channels=cko_dl_gate_hdim,
                        kernel_size=(1, 3), padding=(0, 1)),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=cko_dl_gate_hdim, out_channels=state_dim,
                        kernel_size=(1, 3), padding=(0, 1)),
                    nn.Sigmoid()
                ) for _ in range(len(self.res_seq_lens))
            ])

        self.with_multitask_attn = with_multitask_attn
        if self.with_multitask_attn:
            self.self_attn = nn.MultiheadAttention(self_attn_embed_dim, self_attn_num_heads, dropout=dropout)
            self.norm = nn.LayerNorm(self_attn_embed_dim)
            self.dropout = nn.Dropout(dropout)

        if self.use_upsampling_pred:
            assert self.use_downsampling_pred

        if self.use_downsampling_pred or self.use_upsampling_pred:
            for ri, rl in enumerate(self.res_seq_lens):
                if ri == 0:
                    pass
                else:
                    assert self.res_seq_lens[ri - 1] % rl == 0
            if self.use_downsampling_pred:
                self.downsamplers = nn.ModuleList()
                for ri, rl in enumerate(self.res_seq_lens):
                    if ri < len(self.res_seq_lens) - 1:
                        self.downsamplers.append(SimpleDownsampler(
                            self.res_seq_lens[ri] // self.res_seq_lens[ri + 1],
                            agg=downsampler_agg
                        ))
                    else:
                        self.downsamplers.append(None)
            if self.use_upsampling_pred:
                self.upsamplers = nn.ModuleList()
                for ri, rl in enumerate(self.res_seq_lens):
                    if ri > 0:
                        if self.upsampler == 'simple':
                            self.upsamplers.append(SimpleUpsampler(
                                self.res_seq_lens[ri - 1] // self.res_seq_lens[ri], self.input_dim, self.output_dim
                            ))
                        elif self.upsampler == 'unet':
                            self.upsamplers.append(UnetUpsampler(
                                seq_in_len=self.res_seq_lens[ri], seq_out_len=self.res_seq_lens[ri - 1], input_dim=self.input_dim, output_dim=self.output_dim, layer_num=4
                            ))
                        else:
                            raise NotImplementedError()
                    else:
                        self.upsamplers.append(None)

            weighted_agg_res_counts = np.ones(len(self.res_seq_lens), dtype=int)
            if self.use_downsampling_pred:
                weighted_agg_res_counts[1:] += 1
            if self.use_upsampling_pred:
                weighted_agg_res_counts[:-1] += 1
            if updown_fusion == 'param':
                self.agg_res_weights = nn.ParameterList()
                for ci, c in enumerate(weighted_agg_res_counts):
                    self.agg_res_weights.append(
                        nn.parameter.Parameter(torch.ones(c, dtype=torch.float32))
                    )
            elif updown_fusion == 'conv':
                self.agg_res_weights_conv = nn.ModuleList()
                for ci, c in enumerate(weighted_agg_res_counts):
                    self.agg_res_weights_conv.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels=c * self.output_dim + attr_dim,
                            out_channels=self.agg_res_hdim,
                            kernel_size=(1, 3), padding=(0, 1)
                        ),
                        nn.ReLU(),
                        nn.Conv2d(
                            in_channels=self.agg_res_hdim, out_channels=c * self.output_dim,
                            kernel_size=(1, 3), padding=(0, 1)
                        ),
                    ))

        # record some middle values such as K matrix and attention scores
        self.kmat = None
        self.cko_gate_value = None
        self.agg_ups_ds_weights = None

    def forward(self, x_data_list, y_data_list):
        assert len(x_data_list) == len(self.enc_base_model_list)

        # multires_x_list: [data, mask, chrono]
        multires_x_list = []
        for data_list in x_data_list:
            multires_x_list.append(torch.cat([data_list[1], data_list[0]], dim=-1))
        # multires_y_attr_list: [mask, chrono]
        multires_yattr_list = []
        for data_list in y_data_list:
            multires_yattr_list.append(data_list[0][:, 1:])

        # prediction from DL models
        enc_emb = []
        for ri in range(len(multires_x_list)):
            enc_emb_ri = self.enc_base_model_list[ri](multires_x_list[ri])
            enc_emb.append(enc_emb_ri)
        enc_emb = torch.stack(enc_emb, dim=2) # [bs, node, R, channel]

        bs, node = enc_emb.shape[0], enc_emb.shape[1]
        self_attn_in = enc_emb.flatten(0, 1).permute(1, 0, 2) # [R, bs x node, channel]
        
        if self.with_multitask_attn:
            attn, attn_weights = self.self_attn(self_attn_in, self_attn_in, self_attn_in)
            dec_emb = self.norm(self_attn_in + self.dropout(attn)) # [R, bs x node, channel]
        else:
            dec_emb = self_attn_in

        dec_emb = dec_emb.reshape(dec_emb.shape[0], bs, node, dec_emb.shape[-1]) # [R, bs, node, channel]
        dl_preds = []
        for ri in range(len(multires_x_list)):
            dl_preds.append(self.dec_base_model_list[ri](dec_emb[ri])) # list of [b, t, n, f]

        # prediction from CKO (compositional koopman operator)
        cko_out_dicts = []
        for ri in range(len(multires_x_list)):
            cko_out_dicts.append(cko_sysid_and_pred(self.cko_models[ri], fit_data=x_data_list[ri], seq_data=y_data_list[ri], 
                pstep=self.pstep, I_factor=self.I_factor))
        cko_preds = [d['decode_s_for_pred'] for d in cko_out_dicts]

        # collect Koopman matrices
        self.kmat = []
        for cm in self.cko_models:
            self.kmat.append(cm.A)
        self.kmat = torch.stack(self.kmat, dim=1)

        # out = gate * cko_pred + (1 - gate) * dl_pred
        # out = [cko_pred + dl_pred for cko_pred, dl_pred in zip(cko_preds, dl_preds)]
        out = []
        if self.cko_dl_gate_flag:
            # collect gate values
            self.cko_gate_value = []
            for ri in range(len(multires_x_list)):
                gate_ri_input = torch.cat([cko_preds[ri], dl_preds[ri], y_data_list[ri][0][:, 1:]], dim=-1)
                gate_ri_input = rearrange(gate_ri_input, 'b t n f -> b f n t')
                gate_ri = self.cko_dl_gates[ri](gate_ri_input)
                gate_ri = rearrange(gate_ri, 'b f n t -> b t n f')
                out.append(gate_ri * cko_preds[ri] + (1 - gate_ri) * dl_preds[ri])
                self.cko_gate_value.append(gate_ri)
        else:
            out = [cko_pred + dl_pred for cko_pred, dl_pred in zip(cko_preds, dl_preds)]


        ret_dict = {
            'cko_preds': cko_preds,
            'dl_preds': dl_preds,
            'out': out,
            'cko_loss_auto_encode': [d['loss_auto_encode'] for d in cko_out_dicts],
            'cko_loss_prediction': [d['loss_prediction'] for d in cko_out_dicts],
            'cko_loss_metric': [d['loss_metric'] for d in cko_out_dicts]
        }

        if self.use_downsampling_pred:
            out_ds = [None]
            for ri in range(len(self.res_seq_lens) - 1):
                out_ds.append(self.downsamplers[ri](out[ri]))
        else:
            out_ds = None

        if self.use_upsampling_pred:
            out_ups = []
            out_up_down_s = [None]
            for ri in range(1, len(self.res_seq_lens)):
                upsampler_input = torch.cat([out[ri], multires_yattr_list[ri]], dim=-1)
                out_ups.append(self.upsamplers[ri](upsampler_input))
                out_up_down_s.append(self.downsamplers[ri - 1](out_ups[-1]))
            out_ups.append(None)
        else:
            out_ups, out_up_down_s = None, None

        if self.updown_fusion == 'param':
            if hasattr(self, 'agg_res_weights'):
                pred = []
                for ri in range(len(self.res_seq_lens)):
                    valid_res = []
                    for reslist in [out, out_ds, out_ups]:
                        if (reslist is not None) and (reslist[ri] is not None):
                            valid_res.append(reslist[ri])
                    assert len(valid_res) == self.agg_res_weights[ri].size(0)
                    agg_weights = F.softmax(self.agg_res_weights[ri], dim=0)
                    weighted_agg_pred_ri = (torch.stack(valid_res, dim=-1) * agg_weights).sum(dim=-1)
                    pred.append(weighted_agg_pred_ri)
            else:
                pred = out
        elif self.updown_fusion == 'conv':
            pred = []
            # collect agg weights
            self.agg_ups_ds_weights = []
            for ri in range(len(self.res_seq_lens)):
                valid_res = []
                for reslist in [out, out_ds, out_ups]:
                    if (reslist is not None) and (reslist[ri] is not None):
                        valid_res.append(reslist[ri])
                cat_valid_res = torch.stack(valid_res, dim=-1)
                agg_res_weights_conv_input = torch.cat([rearrange(cat_valid_res, 'b t n f s -> b t n (f s)'), multires_yattr_list[ri]], dim=-1)
                agg_res_weights_conv_input = rearrange(agg_res_weights_conv_input, 'b t n f -> b f n t')
                agg_weights = self.agg_res_weights_conv[ri](agg_res_weights_conv_input)
                agg_weights = rearrange(agg_weights, 'b (f s) n t -> b t n f s', s=len(valid_res))
                agg_weights = F.softmax(agg_weights, dim=-1)
                weighted_agg_pred_ri = reduce(cat_valid_res * agg_weights, 'b t n f s -> b t n f', 'sum')
                pred.append(weighted_agg_pred_ri)
                self.agg_ups_ds_weights.append(agg_weights)

        ret_dict.update({
            'pred': pred, # weighted summation of multiple methods
            'out_ds': out_ds,
            'out_ups': out_ups,
            'out_up_down_s': out_up_down_s
        })

        return ret_dict
