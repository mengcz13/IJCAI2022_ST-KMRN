import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class SelfAttentionEncodingLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.outproj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.outnorm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        attn_output = self.norm(query + self.dropout(attn_output))
        output = self.outnorm(attn_output + self.dropout(self.outproj(attn_output)))
        return output


class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0):
        super().__init__()
        self.attn_enc = SelfAttentionEncodingLayer(
            embed_dim, num_heads, dropout
        )

    def forward(self, input):
        b = input.shape[0]
        input = rearrange(input, 'b t n f -> t (b n) f')
        query = input[-1:]
        key, value = input, input
        output = self.attn_enc(query, key, value)
        output = rearrange(output, 't (b n) f -> (t b) n f', b=b)
        return output


class TemporalSelfAttentionv2(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.transfer_h = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.transfer_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, input):
        tr_h = self.transfer_h(input) # b x t x n x f
        tr_h_avg = reduce(tr_h, 'b t n f -> b () n f', 'mean') # b x 1 x n x f
        tr_a = self.transfer_a(tr_h_avg)
        attention = F.sigmoid(torch.matmul(rearrange(tr_h, 'b t n f -> b t n 1 f'), rearrange(tr_a, 'b 1 n f -> b 1 n f 1')))
        attention = rearrange(attention, 'b t n 1 1 -> b t n 1')
        res = reduce(attention * tr_h, 'b t n f -> b n f', 'mean')
        return res


class SpatialSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0):
        super().__init__()
        self.attn_enc = SelfAttentionEncodingLayer(
            embed_dim, num_heads, dropout
        )

    def forward(self, input):
        input = rearrange(input, 'b n f -> n b f')
        output = self.attn_enc(input, input, input)
        output = rearrange(output, 'n b f -> b n f')
        return output


class GraphWaveNetBlock(nn.Module):
    def __init__(self, num_nodes, dropout=0.3, supports=None, gcn_bool=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_dim=12, nembdim=10,
                 residual_channels=32, dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(GraphWaveNetBlock, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, nembdim), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(nembdim, num_nodes), requires_grad=True)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :nembdim], torch.diag(p[:nembdim] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:nembdim] ** 0.5), n[:, :nembdim].t())
                self.nodevec1 = nn.Parameter(
                    initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(
                    initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                # new_dilation *= 2
                # receptive_field += additional_scope
                # additional_scope *= 2
                new_dilation *= kernel_size
                receptive_field += additional_scope
                additional_scope *= kernel_size
                if self.gcn_bool:
                    self.gconv.append(
                        gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        print('Receptive field: {}'.format(self.receptive_field))

    def forward(self, input):
        # input: [bs, T, N, F]
        input = input.transpose(1, 3)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(
                input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(
                F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        # x = skip
        x = F.relu(skip)
        # x = F.relu(reduce(skip, 'b c n t -> b c n ()', 'mean'))
        x = F.relu(self.end_conv_1(x))
        x = F.relu(self.end_conv_2(x))
        x = rearrange(x, 'b c n t -> b t n c')
        return x


class MultiresAttnEncoder(nn.Module):
    def __init__(self, num_levels, num_nodes, dropout=0.3, supports=None, gcn_bool=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_dim=12, nembdim=10,
                 residual_channels=32, dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2,
                 attn_dropout=0.1, attn_num_heads=8,
                 ds_kernel_size=3, ds_stride=2):
        super().__init__()
        self.num_levels = num_levels
        self.gw_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.temp_attn_layers = nn.ModuleList()
        # self.spat_attn_layers = nn.ModuleList()
        for li in range(num_levels):
            if li == 0:
                gw_in_dim = in_dim
            else:
                gw_in_dim = end_channels
            self.gw_blocks.append(
                GraphWaveNetBlock(
                    num_nodes=num_nodes, dropout=dropout, supports=supports, gcn_bool=gcn_bool,
                    addaptadj=addaptadj, aptinit=aptinit, in_dim=gw_in_dim, out_dim=None, nembdim=nembdim,
                    residual_channels=residual_channels, dilation_channels=dilation_channels, skip_channels=skip_channels,
                    end_channels=end_channels, kernel_size=kernel_size, blocks=blocks, layers=layers
                )
            )
            self.downsample_layers.append(
                nn.Sequential(
                    Rearrange('b t n f -> b f n t'),
                    nn.Conv2d(in_channels=end_channels, out_channels=end_channels, kernel_size=(1, ds_kernel_size), padding=(kernel_size - 1)//2),
                    nn.BatchNorm2d(end_channels),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=(1, ds_stride)),
                    Rearrange('b f n t -> b t n f')
                )
            )
            # self.temp_attn_layers.append(
            #     TemporalSelfAttention(embed_dim=end_channels, num_heads=attn_num_heads, dropout=attn_dropout)
            # )
            # self.spat_attn_layers.append(
            #     SpatialSelfAttention(embed_dim=end_channels, num_heads=attn_num_heads, dropout=attn_dropout)
            # )

            self.temp_attn_layers.append(
                TemporalSelfAttentionv2(hidden_dim=end_channels)
            )

    def forward(self, input):
        blk_input = input
        res_emb = 0
        for li in range(self.num_levels):
            blk_out = self.gw_blocks[li](blk_input)
            blk_input = self.downsample_layers[li](blk_out)
            blk_out = self.temp_attn_layers[li](blk_out)
            # blk_out = self.spat_attn_layers[li](blk_out)
            # blk_out = rearrange(blk_out, 'b 1 n f -> b n f')
            res_emb += blk_out
        return res_emb
        

class MultiresAttnDecoder(nn.Module):
    def __init__(self, in_channels, end_channels, seq_out_len, out_dim):
        super().__init__()
        self.end_conv_1 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=seq_out_len*out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.seq_out_len = seq_out_len
        self.out_dim = out_dim

    def forward(self, x):
        # x: [bs, node, c]
        x = x.permute(0, 2, 1).unsqueeze(-1)  # [bs, c, node, 1]
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # [bs, output_len * fnum, node, 1] -> [bs, fnum, node, output_len]
        x = x[:, :, :, 0]
        bs, noden = x.shape[0], x.shape[2]
        x = x.reshape(bs, self.seq_out_len,
                      self.out_dim, noden).permute(0, 2, 3, 1)
        x = x.permute(0, 3, 2, 1)  # [bs, T, N, F]
        return x
