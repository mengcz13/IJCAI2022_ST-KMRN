import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from modules.upsampler import SimpleUpsampler, SimpleDownsampler, UnetUpsampler


class MultitaskModel(nn.Module):
    def __init__(self,
        input_dim, output_dim, attr_dim, res_seq_lens,
        enc_base_model_list, dec_base_model_list,
        self_attn_embed_dim, self_attn_num_heads,
        dropout=0.1,
        use_downsampling_pred=False,
        use_upsampling_pred=False,
        upsampler='simple',
        updown_fusion='param', agg_res_hdim=32, downsampler_agg='mean'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attr_dim = attr_dim
        self.res_seq_lens = res_seq_lens
        self.use_downsampling_pred = use_downsampling_pred
        self.use_upsampling_pred = use_upsampling_pred
        self.upsampler = upsampler
        self.updown_fusion = updown_fusion
        self.agg_res_hdim = agg_res_hdim

        self.enc_base_model_list = nn.ModuleList(enc_base_model_list)
        self.dec_base_model_list = nn.ModuleList(dec_base_model_list)
        assert len(self.enc_base_model_list) == len(self.dec_base_model_list)

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
                            in_channels=c * self.output_dim + self.attr_dim,
                            out_channels=self.agg_res_hdim,
                            kernel_size=(1, 3), padding=(0, 1)
                        ),
                        nn.ReLU(),
                        nn.Conv2d(
                            in_channels=self.agg_res_hdim, out_channels=c * self.output_dim,
                            kernel_size=(1, 3), padding=(0, 1)
                        ),
                    ))



    def forward(self, multires_x_list, multires_yattr_list, *args, **kwargs):
        assert len(multires_x_list) == len(self.enc_base_model_list)

        enc_emb = []
        for ri in range(len(multires_x_list)):
            enc_emb_ri = self.enc_base_model_list[ri](multires_x_list[ri], *args, **kwargs)
            enc_emb.append(enc_emb_ri)
        enc_emb = torch.stack(enc_emb, dim=2) # [bs, node, R, channel]

        bs, node = enc_emb.shape[0], enc_emb.shape[1]
        self_attn_in = enc_emb.flatten(0, 1).permute(1, 0, 2) # [R, bs x node, channel]
        
        attn, attn_weights = self.self_attn(self_attn_in, self_attn_in, self_attn_in)
        dec_emb = self.norm(self_attn_in + self.dropout(attn)) # [R, bs x node, channel]

        dec_emb = dec_emb.reshape(dec_emb.shape[0], bs, node, dec_emb.shape[-1]) # [R, bs, node, channel]
        out = []
        for ri in range(len(multires_x_list)):
            out.append(self.dec_base_model_list[ri](dec_emb[ri])) # list of [b, t, n, f]

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

        return {
            'pred': pred, # weighted summation of multiple methods
            'out': out,
            'out_ds': out_ds,
            'out_ups': out_ups,
            'out_up_down_s': out_up_down_s
        }
