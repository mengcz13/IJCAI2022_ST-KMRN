import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from modules.upsampler import SimpleUpsampler, SimpleDownsampler, UnetUpsampler
from base_models.gw_kae.kae_model import koopmanAEwithattr, kae_sysid_and_pred


class MultitaskModel(nn.Module):
    def __init__(self,
        input_dim, output_dim, res_input_seq_lens, res_seq_lens,
        enc_base_model_list, dec_base_model_list,
        self_attn_embed_dim, self_attn_num_heads,
        attr_dim, state_dim, action_dim, relation_dim,
        nodenum, bottleneck, alpha, init_scale, kae_backward,
        dropout=0.1,
        use_downsampling_pred=False,
        use_upsampling_pred=False,
        upsampler='simple'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.res_input_seq_lens = res_input_seq_lens
        self.res_seq_lens = res_seq_lens
        self.use_downsampling_pred = use_downsampling_pred
        self.use_upsampling_pred = use_upsampling_pred
        self.upsampler = upsampler

        self.enc_base_model_list = nn.ModuleList(enc_base_model_list)
        self.dec_base_model_list = nn.ModuleList(dec_base_model_list)
        assert len(self.enc_base_model_list) == len(self.dec_base_model_list)

        # one kae model for each resolution
        self.kae_models = nn.ModuleList([
            koopmanAEwithattr(
                fdim=state_dim, adim=attr_dim, nodenum=nodenum,
                b=bottleneck,# dim of Koopman state space
                steps=self.res_seq_lens[res_i],# length of forward
                steps_back=self.res_seq_lens[res_i],# length of backward
                alpha=alpha,
                init_scale=init_scale
            ) for res_i in range(len(self.res_seq_lens))
        ])
        self.kae_backward = kae_backward

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
                            agg='mean'
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
            self.agg_res_weights = nn.ParameterList()
            for ci, c in enumerate(weighted_agg_res_counts):
                self.agg_res_weights.append(
                    nn.parameter.Parameter(torch.ones(c, dtype=torch.float32))
                )

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
        
        attn, attn_weights = self.self_attn(self_attn_in, self_attn_in, self_attn_in)
        dec_emb = self.norm(self_attn_in + self.dropout(attn)) # [R, bs x node, channel]

        dec_emb = dec_emb.reshape(dec_emb.shape[0], bs, node, dec_emb.shape[-1]) # [R, bs, node, channel]
        dl_preds = []
        for ri in range(len(multires_x_list)):
            dl_preds.append(self.dec_base_model_list[ri](dec_emb[ri])) # list of [b, t, n, f]

        # prediction from KAE (Koopman Autoencoder)
        kae_out_dicts = []
        for ri in range(len(multires_x_list)):
            kae_out_dicts.append(kae_sysid_and_pred(self.kae_models[ri], fit_data=x_data_list[ri], seq_data=y_data_list[ri], backward=self.kae_backward))
        kae_preds = [d['out_fw'][:, :-1] for d in kae_out_dicts]
        kae_identity_init_state = [d['out_fw'][:, -1:] for d in kae_out_dicts]
        if 'out_bw' in kae_out_dicts[0]:
            kae_bwds = [d['out_bw'] for d in kae_out_dicts]

        # out = kae_pred + dl_pred
        out = [kae_pred + dl_pred for kae_pred, dl_pred in zip(kae_preds, dl_preds)]

        ret_dict = {
            'kae_preds': kae_preds,
            'kae_identity_init_state': kae_identity_init_state,
            'kae_bwds': kae_bwds,
            'dl_preds': dl_preds,
            'out': out
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

        ret_dict.update({
            'pred': pred, # weighted summation of multiple methods
            'out_ds': out_ds,
            'out_ups': out_ups,
            'out_up_down_s': out_up_down_s
        })

        return ret_dict
