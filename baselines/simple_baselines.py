import os
import sys

from numpy.lib.npyio import save
sys.path.append('..')
from argparse import ArgumentParser

import numpy as np
import sparse
from tqdm import tqdm

from dataset.nyc_taxi.nyctaxi_dataset import NYCTaxiDataset


def _convert_interval_to_steps(interval, baseres):
    if interval == '1week':
        if baseres == '30min':
            return 24*7*2
        elif baseres == '5min':
            return 24*7*12
        elif baseres == '10min':
            return 24*7*6
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def historical_averaging(dataset, args):
    '''
    Return the average of values in trainval set
    '''
    rets = []
    for idx in tqdm(range(len(dataset))):
        rets.append(dataset[idx])
    preds, trues = [], []
    for ret in tqdm(rets):
        true_i = ret['data_output']
        ts_abs_output_i = ret['ts_abs_output']
        trainval_end = dataset.start_ts[0]
        pred_i = []
        for t_abs in ts_abs_output_i:
            hist_idx = -np.arange(-t_abs, 1, _convert_interval_to_steps('1week', args.baseres)).astype(int)
            hist_idx = hist_idx[hist_idx < trainval_end]
            masked_volume_data = dataset.masked_volume_data[hist_idx]
            volume_data_mask = dataset.volume_data_mask[hist_idx]
            hist_avg_i = masked_volume_data.sum(axis=0) / volume_data_mask.sum(axis=0)
            pred_i.append(hist_avg_i)
        pred_i = np.stack(pred_i, axis=0)
        preds.append(pred_i)
        trues.append(true_i)
    preds = np.stack(preds, axis=0)
    trues = np.stack(trues, axis=0)
    return preds, trues


def static(dataset, args):
    '''
    Return the value of one week prior to output
    '''
    rets = []
    for idx in tqdm(range(len(dataset))):
        rets.append(dataset[idx])
    preds, trues = [], []
    # for ret in tqdm(rets):
    #     true_i = ret['data_output']
    #     pred_i = ret['data_input']
    #     last_idx = np.arange(true_i.shape[0]) + pred_i.shape[0]
    #     while (last_idx.max() >= pred_i.shape[0]):
    #         last_idx[last_idx >= pred_i.shape[0]] -= _convert_interval_to_steps('1week', args.baseres)
    #     pred_i = pred_i[last_idx][:, :, :true_i.shape[2]]
    #     preds.append(pred_i)
    #     trues.append(true_i)
    # preds = np.stack(preds, axis=0)
    # trues = np.stack(trues, axis=0)
    # return preds, trues

    for ret in tqdm(rets):
        true_i = ret['data_output']
        input_i = ret['data_input']
        input_mask_i = ret['mask_input']
        last_idx = np.arange(true_i.shape[0]) + input_i.shape[0]

        pred_i = []
        for li in last_idx:
            pred_ii = np.zeros(shape=(true_i.shape[1], true_i.shape[2]))
            for ni in range(true_i.shape[1]):
                for fi in range(true_i.shape[2]):
                    s = li
                    while ((s >= input_i.shape[0]) or (input_mask_i[s, ni, fi] == 0)) and (s >= 0):
                        s -= _convert_interval_to_steps('1week', args.baseres)
                    if s >= 0:
                        pred_ii[ni, fi] = input_i[s, ni, fi]
            pred_i.append(pred_ii)
        pred_i = np.stack(pred_i, axis=0)
        preds.append(pred_i)
        trues.append(true_i)
    preds = np.stack(preds, axis=0)
    trues = np.stack(trues, axis=0)
    return preds, trues


BASELINES = {
    'hist_avg': historical_averaging,
    'static': static
}


def main(args):
    dataset = NYCTaxiDataset(
        save_folder=args.data_folder,
        input_len=args.input_len,
        output_len=args.output_len,
        # historical_temp_res=['6-hour', 'day'],
        historical_temp_res=[],
        forecast_temp_res=[],
        keep_ratio=args.keep_ratio,
        trainval_ps=(0.6, 0.2),
        mask_seed=42,
        seq_diff=args.seq_diff,
        data_type='test',
        scale=args.scale,
        scaler=None,
        return_delta=False,
        resolution_type=args.resolution_type
    )
    preds, trues = BASELINES[args.method](dataset, args)

    print('MSE: {:.4f}'.format(np.square(preds - trues).mean()))
    print('RMSE: {:.4f}'.format(np.sqrt(np.square(preds - trues).mean())))
    print('MAE: {:.4f}'.format(np.abs(preds - trues).mean()))

    folder_path = os.path.join(args.res_folder, args.method)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(os.path.join(folder_path, 'pred.npy'), preds)
    np.save(os.path.join(folder_path, 'true.npy'), trues)
    np.savez(os.path.join(folder_path, 'scaler.npz'), mean=dataset.scaler.mean_, var=dataset.scaler.var_)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--method', required=True,
        help='hist_avg or static')
    parser.add_argument('--data_folder', required=True,
        help='path to dataset folder')
    parser.add_argument('--res_folder', required=True,
        help='path to the folder saving results')
    parser.add_argument('--input_len', type=int, required=True)
    parser.add_argument('--output_len', type=int, required=True)
    parser.add_argument('--seq_diff', type=int, default=1)
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--baseres', type=str, default='30min')
    parser.add_argument('--resolution_type', type=str, default='agg')
    parser.add_argument('--keep_ratio', type=float, default=0.8)
    args = parser.parse_args()
    main(args)