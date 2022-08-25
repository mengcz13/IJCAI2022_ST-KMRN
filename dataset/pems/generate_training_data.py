'''
Modified from https://github.com/liyaguang/DCRNN/blob/318b5f1f112cae3d395e4db5c0be0dda2cf49141/scripts/generate_training_data.py.

- Added option for window size
- Added option for dense multires format output
  - Output: npz file with keys:
    `[train/val/test]_seq`: [#subsequences, #regions+#boroughs, #resolutions], array of [#frames, #feature_dim]
    `_time`: [#subsequences, #regions+#boroughs, #resolutions], array of [#frames,]. Number of time steps from the first frame of a fully observed seq to the start time of current slot.
    `_time_abs`: [#subsequences, #regions+#boroughs, #resolutions], array of [#frames,]. Number of time steps from the first frame of the dataset (2017-1-1 00:00) to the start time of current slot
    `resolution_steps`: [#resolutions], array of the number of timesteps each resolution covers
    `input_ts_max`: the time step of the last input frame. Frames with `_time`<=`input_ts_max` are input and the left are output.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None,
        return_abs_time=False, stride=1
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    # if return_abs_time:
    # x_abs_time: (epoch_size, input_length)
    # y_abs_time: (epoch_size, output_length)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    x_abs_time, y_abs_time = [], []
    for t in range(min_t, max_t, stride):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
        x_abs_time.append(t + x_offsets)
        y_abs_time.append(t + y_offsets)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    x_abs_time = np.stack(x_abs_time)
    y_abs_time = np.stack(y_abs_time)
    if return_abs_time:
        return x, y, x_abs_time, y_abs_time
    else:
        return x, y


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        # np.concatenate((np.arange(-11, 1, 1),))
        np.concatenate((np.arange(-(args.window_size - 1), 1, 1),))
    )
    # Predict the next one hour
    # y_offsets = np.sort(np.arange(1, 13, 1))
    y_offsets = np.sort(np.arange(1, args.window_size + 1, 1))
    assert len(x_offsets) == len(y_offsets)
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y, x_abs_time, y_abs_time = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=args.add_time_in_day,
        add_day_in_week=args.add_day_in_week,
        return_abs_time=True,
        stride=args.stride
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]
    # abs_time
    x_abs_time_train, y_abs_time_train = x_abs_time[:num_train], y_abs_time[:num_train]
    x_abs_time_val, y_abs_time_val = x_abs_time[num_train: num_train + num_val], y_abs_time[num_train: num_train + num_val]
    x_abs_time_test, y_abs_time_test = x_abs_time[-num_test:], y_abs_time[-num_test:]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.dense_multires_output:
        to_save_dict = {}
        for cat in ['train', 'val', 'test']:
            _x, _y = locals()['x_' + cat], locals()['y_' + cat] # [#samples, T, #nodes, #dim]
            _xy = np.concatenate((_x, _y), axis=1).transpose(0, 2, 1, 3)
            _seq = np.zeros((_xy.shape[0], _xy.shape[1], 1), dtype=object)
            x_abs_time_, y_abs_time_ = locals()['x_abs_time_' + cat], locals()['y_abs_time_' + cat]
            xy_abs_time_ = np.concatenate((x_abs_time_, y_abs_time_), axis=1)
            _time = np.zeros_like(_seq)
            _time_abs = np.zeros_like(_seq)
            for ii in range(_seq.shape[0]):
                for jj in range(_seq.shape[1]):
                    _seq[ii, jj, 0] = _xy[ii, jj]
                    _time[ii, jj, 0] = np.arange(args.window_size * 2)
                    _time_abs[ii, jj, 0] = xy_abs_time_[ii]
            to_save_dict['{}_seq'.format(cat)] = _seq
            to_save_dict['{}_time'.format(cat)] = _time
            to_save_dict['{}_time_abs'.format(cat)] = _time_abs
        to_save_dict['resolution_steps'] = np.array([1], dtype=np.int64)
        to_save_dict['input_ts_max'] = args.window_size - 1
        np.savez_compressed(
            os.path.join(args.output_dir, 'sampled_dense_multires_seqs.npz'),
            **to_save_dict
        )

        # generate an empty graph
        node_num = x_train.shape[2]
        np.savez_compressed(
            os.path.join(args.output_dir, 'edges_valid_region_borough.npz'),
            edges=np.zeros((1, 2), dtype=np.int64),
            node_types=np.ones(node_num, dtype=np.int64)
        )
    else:
        for cat in ["train", "val", "test"]:
            _x, _y = locals()["x_" + cat], locals()["y_" + cat]
            print(cat, "x: ", _x.shape, "y:", _y.shape)
            np.savez_compressed(
                os.path.join(args.output_dir, "%s.npz" % cat),
                x=_x,
                y=_y,
                x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
            )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw traffic readings.",
    )
    parser.add_argument(
        '--window_size', type=int, default=12, help='num of steps for input/output, 12 by default (1 hour)'
    )
    parser.add_argument(
        '--stride', type=int, default=1
    )
    parser.add_argument(
        '--dense_multires_output', action='store_true', help='enable output format for dense_multires format'
    )
    parser.add_argument('--no_time_in_day', dest='add_time_in_day', action='store_false')
    parser.add_argument('--add_day_in_week', action='store_true')
    args = parser.parse_args()
    main(args)