import os
import sys
import numpy as np
from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


class SimpleDownsampler(nn.Module):
    def __init__(self, down_ratio, agg='mean'):
        super().__init__()
        self.down_ratio = down_ratio
        self.agg = agg

    def forward(self, input):
        if self.agg == 'first':
            x = rearrange(input, 'b (t dr) n f -> b t dr n f', dr=self.down_ratio)
            x = x[:, :, 0, :, :]
        else:
            x = reduce(input, 'b (t dr) n f -> b t n f', self.agg, dr=self.down_ratio)
        return x


class SimpleUpsampler(nn.Module):
    def __init__(self, up_ratio, input_dim, output_dim):
        super().__init__()

        self.up_ratio = up_ratio
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim, out_channels=64,
                kernel_size=(1, 5),
                padding=(0, 2)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=32,
                kernel_size=(1, 3),
                padding=(0, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=up_ratio*output_dim,
                kernel_size=(1, 3),
                padding=(0, 1)
            ),
            Rearrange('b (ts c) n t -> b c n (t ts)', ts=up_ratio)
        )

    def forward(self, input):
        input = rearrange(input, 'b t n f -> b f n t')
        out = self.layers(input)
        out = rearrange(out, 'b f n t -> b t n f')
        return out


class UnetUpsampler(nn.Module):
    def __init__(self, seq_in_len, seq_out_len, 
                input_dim, output_dim, layer_num, 
                min_out_channels=128, max_out_channels=512, min_kernel_size=3):
        super().__init__()

        self.seq_in_len = seq_in_len
        self.seq_out_len = seq_out_len

        out_channels = [min_out_channels,]
        start_kernel_size = 2 ** (np.ceil(np.log2(seq_in_len)).astype(int).item() + 1) + 1
        kernel_sizes = [start_kernel_size,]
        for _ in range(layer_num - 1):
            out_channels.append(min(out_channels[-1] * 2, max_out_channels))
            kernel_sizes.append(max((kernel_sizes[-1] - 1) // 2 + 1, min_kernel_size))
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        print('out channels:', self.out_channels)
        print('kernel sizes:', self.kernel_sizes)

        # downsampling layers
        seq_len = seq_in_len
        downsampling_layers = []
        for li, (out_channel, kernel_size) in enumerate(zip(self.out_channels, self.kernel_sizes)):
            if li == 0:
                in_channel = input_dim
            else:
                in_channel = self.out_channels[li - 1]
            downsampling_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel, out_channels=out_channel,
                        kernel_size=(1, kernel_size), stride=(1, 2),
                        padding=(0, (kernel_size - 1) // 2)
                    ),
                    nn.LeakyReLU(0.2)
                )
            )
            seq_len = (seq_len - 1) // 2 + 1
        self.downsampling_layers = nn.ModuleList(downsampling_layers)

        # bottleneck
        self.bottleneck_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_channels[-1], out_channels=self.out_channels[-1],
                kernel_size=(1, self.kernel_sizes[-1]), stride=(1, 2),
                padding=(0, (self.kernel_sizes[-1] - 1) // 2)
            ),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2)
        )
        seq_len = (seq_len - 1) // 2 + 1

        # upsampling layers
        upsampling_layers = []
        for li, out_channel, kernel_size in reversed(list(zip(range(layer_num), self.out_channels, self.kernel_sizes))):
            if li == layer_num - 1:
                in_channel = self.out_channels[-1]
            else:
                in_channel = 2 * self.out_channels[li + 1]
            out_channel = self.out_channels[li]
            upsampling_layers.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channel, out_channels=2 * out_channel,
                    kernel_size=(1, kernel_size),
                    padding=(0, (kernel_size - 1) // 2)
                ),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                Rearrange('b (ts c) n t -> b c n (t ts)', ts=2)
            ))
            seq_len *= 2
        self.upsampling_layers = nn.ModuleList(upsampling_layers)

        # final conv layer
        up_r = np.ceil(seq_out_len / seq_len).astype(int).item()
        self.final_conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * self.out_channels[0],
                out_channels=up_r * output_dim,
                kernel_size=(1, self.kernel_sizes[-1]),
                padding=(0, (self.kernel_sizes[-1] - 1) // 2)
            ),
            Rearrange('b (ts c) n t -> b c n (t ts)', ts=up_r)
        )

    def forward(self, input):
        # input: [bs, t, n, f]
        input = rearrange(input, 'b t n f -> b f n t')
        shortcuts = []
        x = input
        for li, l in enumerate(self.downsampling_layers):
            x = l(x)
            shortcuts.append(x)
        x = self.bottleneck_layer(x)
        for li, l in enumerate(self.upsampling_layers):
            x = l(x)
            sc_to_cat = shortcuts[-li-1]
            sc_to_cat = F.pad(sc_to_cat, (0, x.shape[-1] - sc_to_cat.shape[-1]), 'constant', 0)
            x = torch.cat((x, sc_to_cat), dim=1)
        out = self.final_conv_layer(x)[..., :self.seq_out_len]
        out = rearrange(out, 'b f n t -> b t n f')
        return out