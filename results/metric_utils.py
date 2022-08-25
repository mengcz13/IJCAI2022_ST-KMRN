import os
import sys
sys.path.append('..')
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset.nyc_taxi.nyctaxi_dataset import NYCTaxiDataset


def masked_err(errfunc, pred_y, data, mask):
    elem_err_mask = errfunc(pred_y, data) * mask
    elem_err_mask[np.isnan(elem_err_mask)] = 0
    elem_err_mask[np.isinf(elem_err_mask)] = 0
    return elem_err_mask.sum() / mask.sum()


def masked_mse(pred_y, data, mask):
    def mse(pred_y, data):
        return (pred_y - data) ** 2
    return masked_err(mse, pred_y, data, mask)


def masked_rmse(pred_y, data, mask):
    return np.sqrt(masked_mse(pred_y, data, mask))


def masked_mae(pred_y, data, mask):
    def mae(pred_y, data):
        return np.abs(pred_y - data)
    return masked_err(mae, pred_y, data, mask)


def masked_mape(pred_y, data, mask):
    def mape(pred_y, data):
        return np.abs(pred_y - data) / np.abs(data)
    return masked_err(mape, pred_y, data, mask)


def masked_smape(pred_y, data, mask):
    def smape(pred_y, data):
        return np.abs(pred_y - data) / (np.abs(data) + np.abs(pred_y))
    return masked_err(smape, pred_y, data, mask)


def eval_metrics(pred_y, data, threshold):
    pred_y = pred_y.flatten()
    data = data.flatten()
    mask = (np.abs(data) >= threshold)
    return {
        'rmse': masked_rmse(pred_y, data, mask),
        'mae': masked_mae(pred_y, data, mask),
        'mape': masked_mape(pred_y, data, mask),
        'smape': masked_smape(pred_y, data, mask)
    }


def reshape_arr(arr, nodenum):
    '''
    Reshape arr to [#samples, T, #nodes, #features]
    '''
    if len(arr.shape) == 3:
        s, t = arr.shape[:2]
        return arr.reshape(s, t, nodenum, -1)
    elif len(arr.shape) == 4:
        return arr
    else:
        raise NotImplementedError()


def inverse_transform(arr, scaler):
    return arr * np.sqrt(scaler.var_) + scaler.mean_


def eval_model(modelname, resfolder, scaler, nodenum, node_type, target_node_res, threshold=10, start_t=None, end_t=None, output_dims=None, agg=None, agg_res=None, agg_res_to_base=None):
    if modelname.startswith('Multitask_'):
        res_id = output_dims[0]
        pred_y = np.load(os.path.join(resfolder, 'res-{}-pred.npy'.format(res_id)))
        data = np.load(os.path.join(resfolder, 'res-{}-true.npy'.format(res_id)))
        pred_y = reshape_arr(pred_y, nodenum)
        data = reshape_arr(data, nodenum)
        
#         # focus on the finest regions only
#         node_sel = (node_type == 0)
        # focus on the boroughs only
#         node_sel = (node_type == 1)
        node_sel = (node_type == target_node_res)
        pred_y = inverse_transform(pred_y, scaler)[:, :, node_sel, :]
        data = inverse_transform(data, scaler)[:, :, node_sel, :]
        if (start_t is not None) and (end_t is not None):
            start_ti = start_t // agg_res_to_base
            end_ti = end_t // agg_res_to_base
            pred_y = pred_y[:, start_ti:end_ti, :, :]
            data = data[:, start_ti:end_ti, :, :]
        if agg is not None:
            if agg == 'mean':
                s, t, n, f = pred_y.shape
                pred_y = pred_y.reshape(s, -1, agg_res, n, f).mean(axis=2)
                data = data.reshape(s, -1, agg_res, n, f).mean(axis=2)
    else:
        if modelname in ['Latent ODE']:
            res = np.load(os.path.join(resfolder, 'test_res.npz'), allow_pickle=True)['pred'].item()
            pred_y = res['pred_y'].mean(axis=0)
            data = res['data']
        elif modelname in ['HA', 'Static', 'GRU', 'Informer', 'Graph WaveNet', 'MTGNN', 'GMAN', 'Informer_Recursive', 'Graph Transformer', 'KoopmanAE'] \
            or modelname.startswith('GRU_') or modelname.startswith('Informer_') or modelname.startswith('Graph WaveNet_') or modelname.startswith('MTGNN_') or modelname.startswith('HA_') or modelname.startswith('Static_') or modelname.startswith('KoopmanAE_'):
            pred_y = np.load(os.path.join(resfolder, 'pred.npy'))
            data = np.load(os.path.join(resfolder, 'true.npy'))
        elif modelname.startswith('GraphTransformer_'):
            pred_y = np.load(os.path.join(resfolder, 'pred.npy'))
            data = np.load(os.path.join(resfolder, 'true.npy'))
            def expand_resolutions(pred):
                bs, ns, fs = pred.shape[0], pred.shape[2], pred.shape[3]
                res0 = pred[:, :480, :, :]
                res1 = np.repeat(pred[:, 480:520, :, :], 12, axis=1)
                res2 = np.repeat(pred[:, 520:530, :, :], 48, axis=1)
                return np.concatenate([res0, res1, res2], axis=-1)
            pred_y = expand_resolutions(pred_y)
            data = expand_resolutions(data)
        
        pred_y = reshape_arr(pred_y, nodenum)
        data = reshape_arr(data, nodenum)
        if (output_dims is not None):
            fdim = pred_y.shape[-1]
            sd, ed = fdim // 3 * output_dims[0], fdim // 3 * output_dims[1]
            pred_y, data = pred_y[..., sd:ed], data[..., sd:ed]
        
#         # focus on the finest regions only
#         node_sel = (node_type == 0)
        # focus on the boroughs only
#         node_sel = (node_type == 1)
        node_sel = (node_type == target_node_res)
        pred_y = inverse_transform(pred_y, scaler)[:, :, node_sel, :]
        data = inverse_transform(data, scaler)[:, :, node_sel, :]
        if (start_t is not None) and (end_t is not None):
            pred_y = pred_y[:, start_t:end_t, :, :]
            data = data[:, start_t:end_t, :, :]
        if agg is not None:
            if agg == 'mean':
                s, t, n, f = pred_y.shape
                pred_y = pred_y.reshape(s, -1, agg_res, n, f).mean(axis=2)
                data = data.reshape(s, -1, agg_res, n, f).mean(axis=2)
    return eval_metrics(pred_y, data, threshold)


res_configs_dict = {
    'NYCTaxiFull': [
        ('HA_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_full/hist_avg',
        ]),
        ('Static_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_full/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/rn97vtsz',
            '../baselines/Informer2020_oldv_01112021/results/3jvgaovi',
            '../baselines/Informer2020_oldv_01112021/results/2lf4k542',
        ]),
        ('GRU_res-day', [
            '../baselines/Informer2020_oldv_01112021/results/1knv08o6',
            '../baselines/Informer2020_oldv_01112021/results/3kxmkbjx',
            '../baselines/Informer2020_oldv_01112021/results/3p8wutve',
        ]),
        ('GRU_res-6hour', [
            '../baselines/Informer2020_oldv_01112021/results/3q4tnld4',
            '../baselines/Informer2020_oldv_01112021/results/3av7okm8',
            '../baselines/Informer2020_oldv_01112021/results/2qj45g5a',
        ]),
        ('GRU_res-30min', [
            '../baselines/Informer2020_oldv_01112021/results/1ei6ecmk',
            '../baselines/Informer2020_oldv_01112021/results/12yk4xfj',
            '../baselines/Informer2020_oldv_01112021/results/2zbrnk0o',
        ]),
        ('Informer_nodt_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/3m97aonn',
            '../baselines/Informer2020_oldv_01112021/results/2ju1a7dr',
            '../baselines/Informer2020_oldv_01112021/results/3jitfs6g',
        ]),
        ('Informer_nodt_res-day', [
            '../baselines/Informer2020_oldv_01112021/results/33oycbdy',
            '../baselines/Informer2020_oldv_01112021/results/b8zcags0',
            '../baselines/Informer2020_oldv_01112021/results/z9hzeuem',
        ]),
        ('Informer_nodt_res-6hour', [
            '../baselines/Informer2020_oldv_01112021/results/37nbm9u4',
            '../baselines/Informer2020_oldv_01112021/results/1w7rax69',
            '../baselines/Informer2020_oldv_01112021/results/1c6b0iez',
        ]),
        ('Informer_nodt_res-30min', [
            '../baselines/Informer2020_oldv_01112021/results/2qzckp56',
            '../baselines/Informer2020_oldv_01112021/results/1jx11y9r',
            '../baselines/Informer2020_oldv_01112021/results/2qjjjfht',
        ]),
        ('Informer_withme_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/17sphair',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/3ihy871p',
            '../baselines/Graph-WaveNet/garage/25bjpl51',
            '../baselines/Graph-WaveNet/garage/3fxsn361',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/28yaogvq',
            '../baselines/MTGNN/save/296mpv5l',
            '../baselines/MTGNN/save/2gv4qeqz',
        ]),
        ('Multitask_GWplusCKO_gate_res-all', [
            '../multitask/save/6wx0ka3b',
            '../multitask/save/r9g0td69',
            '../multitask/save/3l8csyhq',
        ]),
#         ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
#             '../multitask/save/lsv16f1d',
#             '../multitask/save/s7t1ddfx',
#             '../multitask/save/yx610e7s',
#         ]),
        # mae loss
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/1x9bl99n',
            '../multitask/save/36nznynd',
            '../multitask/save/3gg34il3',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/277qvz70',
            '../baselines/koopmanAE/save/1qvbdr41',
            '../baselines/koopmanAE/save/13yc1e6u',
        ]),
    ],
    'NYCTaxi':[
        ('HA_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi/hist_avg',
        ]),
        ('Static_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi/static',
        ]),
#         ('GRU_res-all', [
#             '../baselines/Informer2020/results/tq3g52g6',
#             '../baselines/Informer2020/results/1ub5vlm0',
#             '../baselines/Informer2020/results/2ge5ok0l',
#         ]),
#         ('GRU_res-day', [
#             '../baselines/Informer2020/results/2opfwh9t',
#             '../baselines/Informer2020/results/37ksagh2',
#             '../baselines/Informer2020/results/s55zndbk',
#         ]),
#         ('GRU_res-6hour', [
#             '../baselines/Informer2020/results/2zx5ikev',
#             '../baselines/Informer2020/results/243slknn',
#             '../baselines/Informer2020/results/2a463kfm',
#         ]),
#         ('GRU_res-30min', [
#             '../baselines/Informer2020/results/2buqpwno',
#             '../baselines/Informer2020/results/3grqbfya',
#             '../baselines/Informer2020/results/1g7686gh',
#         ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/3mfpc0ya',
            '../baselines/Graph-WaveNet/garage/2l7dmn78',
            '../baselines/Graph-WaveNet/garage/3k19prwa',
        ]),
        ('Graph WaveNet_res-day', [
            '../baselines/Graph-WaveNet/garage/2fma0g93',
            '../baselines/Graph-WaveNet/garage/2pzs8igx',
            '../baselines/Graph-WaveNet/garage/1vj3hexo',
        ]),
        ('Graph WaveNet_res-6hour', [
            '../baselines/Graph-WaveNet/garage/22e4tp4s',
            '../baselines/Graph-WaveNet/garage/3dziq07r',
            '../baselines/Graph-WaveNet/garage/3ahvhq5r',
        ]),
        ('Graph WaveNet_res-30min', [
            '../baselines/Graph-WaveNet/garage/36qjos1l',
            '../baselines/Graph-WaveNet/garage/1fn03dc7',
            '../baselines/Graph-WaveNet/garage/2s8hla05',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/2wdro2ot',
            '../baselines/MTGNN/save/tp4pnkxv',
            '../baselines/MTGNN/save/2m37pp1t',
        ]),
        ('Informer_nodt_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/2j03ndl5',
            '../baselines/Informer2020_oldv_01112021/results/1tnmjbeg',
            '../baselines/Informer2020_oldv_01112021/results/3ql4y11q',
        ]),
        ('Informer_nodt_res-day', [
            '../baselines/Informer2020_oldv_01112021/results/16ii2x4f',
            '../baselines/Informer2020_oldv_01112021/results/36y8tiah',
            '../baselines/Informer2020_oldv_01112021/results/397v24zo',
        ]),
        ('Informer_nodt_res-6hour', [
            '../baselines/Informer2020_oldv_01112021/results/2ik4rgio',
            '../baselines/Informer2020_oldv_01112021/results/24o74ga1',
            '../baselines/Informer2020_oldv_01112021/results/1rdr7h8h',
        ]),
        ('Informer_nodt_res-30min', [
            '../baselines/Informer2020_oldv_01112021/results/1oyuzvcz',
            '../baselines/Informer2020_oldv_01112021/results/329rpk53',
            '../baselines/Informer2020_oldv_01112021/results/1sbvrnyn',
        ]),
        ('Multitask_res-all', [
            # with GW
            '../multitask/save/1o48n9o8',
            '../multitask/save/3ea4uafp',
            '../multitask/save/zas3fuhm',
        ]),
        ('Multitask_ds_res-all', [
            '../multitask/save/eyjs3vl5',
        ]),
        ('Multitask_ds_ups_res-all', [
            '../multitask/save/3i78665g',
            '../multitask/save/2sz2n4ox',
            '../multitask/save/302nhb9m',
        ]),
        ('Multitask_ds_ups_convfusion_res-all', [
            '../multitask/save/e6m3gbma',
            '../multitask/save/5tb8beau',
            '../multitask/save/nbuqlxnx',
        ]),
        ('Multitask_ds_ups_newlr_res-all', [
            '../multitask/save/22ivynd4',
            '../multitask/save/1t40ludj',
            '../multitask/save/2muzx9bz',
        ]),
        ('Multitask_ds_ups_bl213_res-all', [
            '../multitask/save/3c1ztbcm',
            '../multitask/save/20qnkig4',
            '../multitask/save/226tsulr',
        ]),
        ('Multitask_ds_ups_bl113_res-all', [
            '../multitask/save/19meg4ys',
            '../multitask/save/7dx1pja8',
            '../multitask/save/2kqiteb7',
        ]),
        ('Multitask_ds_ups_bl414_res-all', [
            '../multitask/save/rig2fg8y',
            '../multitask/save/x79fhp2o',
            '../multitask/save/1r9o208z',
        ]),
        ('Multitask_ds_ups_mseloss_res-all', [
            '../multitask/save/9kufzs4q',
            '../multitask/save/28s39sq5',
            '../multitask/save/1d02hak7',
        ]),
        ('Multitask_ds_ups_newlr_lrs_res-all', [
            '../multitask/save/2ov6w6fu',
        ]),
        ('Multitask_GWplusCKO_res-all', [
            # default wdecay 1e-4
            '../multitask/save/o57c48ba',
            '../multitask/save/crzqd42n',
            '../multitask/save/21bt4f4p',
        ]),
        ('Multitask_GWplusCKO_gate_res-all', [
            '../multitask/save/2y9y9z7q',
            '../multitask/save/1u15k55r',
            '../multitask/save/3tt2325o',
        ]),
        ('Multitask_GWplusCKO_gate_ups_ds_res-all', [
            '../multitask/save/2kgmhbzh',
            '../multitask/save/36ir271d',
            '../multitask/save/8v8899nq',
        ]),
#         ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
#             '../multitask/save/30n88syu',
#             '../multitask/save/a1i7c5ir',
#             '../multitask/save/248ly5go',
#         ]),
        # MAE loss
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/19z4709w',
            '../multitask/save/1fczh1rr',
            '../multitask/save/1v3p10j3',
        ]),
        ('Multitask_MultiresAttn_res-all', [
            '../multitask/save/2nncisr3',
            '../multitask/save/28p0g8s5'
        ]),
        ('Multitask_nosa_res-all', [
            '../multitask/save/10cvp0wz',
            '../multitask/save/3frzgeap',
            '../multitask/save/j6eiq0xd',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/8yqgtl83',
            '../baselines/koopmanAE/save/32drg2gu',
            '../baselines/koopmanAE/save/1ryfhzqm',
        ]),
    ],
    'NYCTaxi06': [
        ('HA_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_06/hist_avg',
        ]),
        ('Static_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_06/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/p9hjkjto',
            '../baselines/Informer2020_oldv_01112021/results/2s2qg3f9',
            '../baselines/Informer2020_oldv_01112021/results/31748u29',
        ]),
        ('Informer_nodt_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/1vr7rqkh',
            '../baselines/Informer2020_oldv_01112021/results/1onrgktx',
            '../baselines/Informer2020_oldv_01112021/results/7iw51451',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/37vjrtl5',
            '../baselines/Graph-WaveNet/garage/2v3ckcc9',
            '../baselines/Graph-WaveNet/garage/1zzqn2zp',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/14p9iy1b',
            '../baselines/MTGNN/save/1yuexj60',
            '../baselines/MTGNN/save/2i3pw4wv',
        ]),
#         ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
#             '../multitask/save/34u9e0bb',
#             '../multitask/save/jm3bhiod',
#             '../multitask/save/31j703ad',
#         ]),
        # MAE
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/2gq4nhdr',
            '../multitask/save/9obbuxfg',
            '../multitask/save/36athjiv',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/1fisuddt',
            '../baselines/koopmanAE/save/1ahldzj6',
            '../baselines/koopmanAE/save/rx3we3bd',
        ]),
    ],
    'NYCTaxi04': [
        ('HA_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_04/hist_avg',
        ]),
        ('Static_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_04/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/bdcmkc83',
            '../baselines/Informer2020_oldv_01112021/results/2q59zhfy',
            '../baselines/Informer2020_oldv_01112021/results/udb4rvex',
        ]),
        ('Informer_nodt_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/b740p0yq',
            '../baselines/Informer2020_oldv_01112021/results/2kkqhl1v',
            '../baselines/Informer2020_oldv_01112021/results/lxonb717',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/saebuc6u',
            '../baselines/Graph-WaveNet/garage/2kqu2hfx',
            '../baselines/Graph-WaveNet/garage/16gcggiu',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/1ebkypwn',
            '../baselines/MTGNN/save/3aldo4wn',
            '../baselines/MTGNN/save/376577tx',
        ]),
#         ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
#             '../multitask/save/18bzjtvb',
#         ]),
        # MAE loss
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/39x1o4lh',
            '../multitask/save/2v4z5uzx',
            '../multitask/save/2m1y8nx7',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/178m0ire',
            '../baselines/koopmanAE/save/53keakc7',
            '../baselines/koopmanAE/save/3isb78ve',
        ]),
    ],
    'NYCTaxi02': [
        ('HA_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_02/hist_avg',
        ]),
        ('Static_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_02/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/hzkbwoo1',
            '../baselines/Informer2020_oldv_01112021/results/1z0q6h24',
            '../baselines/Informer2020_oldv_01112021/results/30mot1lw',
        ]),
        ('Informer_nodt_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/ggf03i06',
            '../baselines/Informer2020_oldv_01112021/results/nbivlhus',
            '../baselines/Informer2020_oldv_01112021/results/382cwp8g',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/32fc7rat',
            '../baselines/Graph-WaveNet/garage/31s8yjvh',
            '../baselines/Graph-WaveNet/garage/2ccq997c',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/18fcp19u',
            '../baselines/MTGNN/save/5f9mlnkm',
            '../baselines/MTGNN/save/27aor076',
        ]),
#         ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
#             '../multitask/save/a4gflrfa',
#         ]),
        # MAE
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/w8wk4gw0',
            '../multitask/save/2e188rot',
            '../multitask/save/3s2sh33q',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/7ebnorlg',
            '../baselines/koopmanAE/save/1h7i44u6',
            '../baselines/koopmanAE/save/24f7qnpz',
        ]),
    ],
    'NYCTaxiGreenFull': [
        ('HA_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_green_full/hist_avg',
        ]),
        ('Static_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_green_full/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/38hodgp5',
            '../baselines/Informer2020_oldv_01112021/results/19z5kjio',
            '../baselines/Informer2020_oldv_01112021/results/273o487e',
        ]),
        ('GRU_res-day', [
            '../baselines/Informer2020_oldv_01112021/results/2tpk9xyl',
            '../baselines/Informer2020_oldv_01112021/results/13ofk4wi',
            '../baselines/Informer2020_oldv_01112021/results/2gva2vse',
        ]),
        ('GRU_res-6hour', [
            '../baselines/Informer2020_oldv_01112021/results/1k91e1yr',
            '../baselines/Informer2020_oldv_01112021/results/3i393fel',
            '../baselines/Informer2020_oldv_01112021/results/35a8ffzj',
        ]),
        ('GRU_res-30min', [
            '../baselines/Informer2020_oldv_01112021/results/232i17gm',
            '../baselines/Informer2020_oldv_01112021/results/pmbchhgp',
            '../baselines/Informer2020_oldv_01112021/results/2bocqeqw',
        ]),
        ('Informer_nodt_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/2tbd59b0',
            '../baselines/Informer2020_oldv_01112021/results/13i3s8uj',
            '../baselines/Informer2020_oldv_01112021/results/koqxxgnh',
        ]),
        ('Informer_nodt_res-day', [
            '../baselines/Informer2020_oldv_01112021/results/3kqzozt4',
            '../baselines/Informer2020_oldv_01112021/results/2hy63ihp',
            '../baselines/Informer2020_oldv_01112021/results/qvewh9dz',
        ]),
        ('Informer_nodt_res-6hour', [
            '../baselines/Informer2020_oldv_01112021/results/1vwqhwg5',
            '../baselines/Informer2020_oldv_01112021/results/chviu6zg',
            '../baselines/Informer2020_oldv_01112021/results/2z4614e2',
        ]),
        ('Informer_nodt_res-30min', [
            '../baselines/Informer2020_oldv_01112021/results/38i3f4y6',
            '../baselines/Informer2020_oldv_01112021/results/38qmuc0z',
            '../baselines/Informer2020_oldv_01112021/results/kg8bau6j',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/2x5b2iix',
            '../baselines/Graph-WaveNet/garage/o0m4322h',
            '../baselines/Graph-WaveNet/garage/36lsfgj4',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/1zx157xi',
            '../baselines/MTGNN/save/1a2isj2g',
            '../baselines/MTGNN/save/1o2wn2rf',
        ]),
        ('Multitask_GWplusCKO_gate_res-all', [
            '../multitask/save/1vc081r4',
            '../multitask/save/1bz4hgj1',
            '../multitask/save/1e6a2miv',
        ]),
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/i432ae50',
            '../multitask/save/1o8b074n',
            '../multitask/save/5hu6hsea',
        ]),
        # MAE
#         ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
#             '../multitask/save/35p7befn',
#             '../multitask/save/dnevtfb0',
#             '../multitask/save/ndssx9of',
#         ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/z7jco8ou',
            '../baselines/koopmanAE/save/8npv4ta8',
#             '../baselines/koopmanAE/save/3mbkhjwt',
            '../baselines/koopmanAE/save/l4k6p36j',
        ]),
    ],
    'NYCTaxiGreen': [
        ('HA_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_green/hist_avg',
        ]),
        ('Static_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_green/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/2uycctql',
            '../baselines/Informer2020_oldv_01112021/results/2v61smp4',
            '../baselines/Informer2020_oldv_01112021/results/p7eexgrp',
        ]),
        ('Informer_nodt_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/1whv4ly3',
            '../baselines/Informer2020_oldv_01112021/results/3614g6zj',
            '../baselines/Informer2020_oldv_01112021/results/2otp4z6r',
        ]),
        ('Informer_nodt_res-day', [
            '../baselines/Informer2020_oldv_01112021/results/2acglt35',
            '../baselines/Informer2020_oldv_01112021/results/rtc00qw3',
            '../baselines/Informer2020_oldv_01112021/results/k434it0z',
        ]),
        ('Informer_nodt_res-6hour', [
            '../baselines/Informer2020_oldv_01112021/results/3if7soa1',
            '../baselines/Informer2020_oldv_01112021/results/k1brz8p3',
            '../baselines/Informer2020_oldv_01112021/results/1m755zx4',
        ]),
        ('Informer_nodt_res-30min', [
            '../baselines/Informer2020_oldv_01112021/results/36eflgpp',
            '../baselines/Informer2020_oldv_01112021/results/24r1xvw4',
            '../baselines/Informer2020_oldv_01112021/results/1jrp8m92',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/2riki5rr',
            '../baselines/Graph-WaveNet/garage/3pw5oyb6',
            '../baselines/Graph-WaveNet/garage/1j966p7c',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/2btusplr',
            '../baselines/MTGNN/save/1p805q4u',
            '../baselines/MTGNN/save/2p8pg364',
        ]),
        ('Multitask_GWplusCKO_gate_res-all', [
            '../multitask/save/mdt2adgz',
            '../multitask/save/mex4e1az',
            '../multitask/save/11xvi079',
        ]),
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/29633nlj',
            '../multitask/save/280alqdv',
            '../multitask/save/79nouqlh',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/20cjx0yd',
            '../baselines/koopmanAE/save/3p6rfw0i',
            '../baselines/koopmanAE/save/1nvcdl10',
        ]),
    ],
    'NYCTaxiGreen06': [
        ('HA_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_green_06/hist_avg',
        ]),
        ('Static_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_green_06/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/3ojcra9e',
            '../baselines/Informer2020_oldv_01112021/results/36n7hfiq',
            '../baselines/Informer2020_oldv_01112021/results/3kjlnqjh',
        ]),
        ('Informer_nodt_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/4kjicvox',
            '../baselines/Informer2020_oldv_01112021/results/3lsma8mt',
            '../baselines/Informer2020_oldv_01112021/results/1grsazm4',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/1iyranda',
            '../baselines/Graph-WaveNet/garage/8eahmpu2',
            '../baselines/Graph-WaveNet/garage/u3w8dmbn',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/1iqgbq83',
            '../baselines/MTGNN/save/1zlxc2xd',
            '../baselines/MTGNN/save/fuo1hc4i',
        ]),
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/3rv6jngz',
            '../multitask/save/xk21f8dp',
            '../multitask/save/19gpvdky',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/11tqkky3',
            '../baselines/koopmanAE/save/3cmxloo5',
            '../baselines/koopmanAE/save/uef0q3jn',
        ]),
    ],
    'NYCTaxiGreen04': [
        ('HA_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_green_04/hist_avg',
        ]),
        ('Static_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_green_04/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/tw4t6daq',
            '../baselines/Informer2020_oldv_01112021/results/1p65fggk',
            '../baselines/Informer2020_oldv_01112021/results/t2gefh8y',
        ]),
        ('Informer_nodt_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/c4nmz7cf',
            '../baselines/Informer2020_oldv_01112021/results/20aomvnf',
            '../baselines/Informer2020_oldv_01112021/results/1tn5hmax',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/1loy7oi9',
            '../baselines/Graph-WaveNet/garage/1opywe0v',
            '../baselines/Graph-WaveNet/garage/1ezmzvl3',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/22ji7gv0',
            '../baselines/MTGNN/save/2ij92bm2',
            '../baselines/MTGNN/save/2xxfxj4x',
        ]),
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/3c98yuk9',
            '../multitask/save/3eiex6i5',
            '../multitask/save/167wehbr',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/3p4439s5',
            '../baselines/koopmanAE/save/3jlwdfg0',
            '../baselines/koopmanAE/save/3rfsszsg',
        ]),
    ],
    'NYCTaxiGreen02': [
        ('HA_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_green_02/hist_avg',
        ]),
        ('Static_res-30min', [
            '../baselines/simple_baseline_res/nyctaxi_green_02/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/1ral9nvy',
            '../baselines/Informer2020_oldv_01112021/results/o4mi0a3e',
            '../baselines/Informer2020_oldv_01112021/results/aj0kmpqw',
        ]),
        ('Informer_nodt_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/17s8c9ak',
            '../baselines/Informer2020_oldv_01112021/results/2j6k7n80',
            '../baselines/Informer2020_oldv_01112021/results/3580yknt',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/3r1hrwjk',
            '../baselines/Graph-WaveNet/garage/h4lgnd4n',
            '../baselines/Graph-WaveNet/garage/1qffn915',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/21v0dsd5',
            '../baselines/MTGNN/save/3qdpjftb',
            '../baselines/MTGNN/save/2kvkqcgf',
        ]),
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/1agqk76u',
            '../multitask/save/2yk10yim',
            '../multitask/save/1j57n8t4',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/2o5v107s',
            '../baselines/koopmanAE/save/2z2m35n3',
            '../baselines/koopmanAE/save/22uznaqn',
        ]),
    ],
    'Solar Energy 10min Full': [
        ('HA_res-10min', [
            '../baselines/simple_baseline_res/solar_energy_10min_full/hist_avg',
        ]),
        ('Static_res-10min', [
            '../baselines/simple_baseline_res/solar_energy_10min_full/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/2j2tnr3q',
            '../baselines/Informer2020_oldv_01112021/results/1cbi31tn',
            '../baselines/Informer2020_oldv_01112021/results/1czhjtcq',
        ]),
        ('GRU_res-6hour', [
            '../baselines/Informer2020_oldv_01112021/results/2c5ru4ez',
            '../baselines/Informer2020_oldv_01112021/results/1tl5jvfr',
            '../baselines/Informer2020_oldv_01112021/results/c8hkdrj4',
        ]),
        ('GRU_res-1hour', [
            '../baselines/Informer2020_oldv_01112021/results/2ixxhwll',
            '../baselines/Informer2020_oldv_01112021/results/173qp5zp',
            '../baselines/Informer2020_oldv_01112021/results/2p0o60dc',
        ]),
        ('GRU_res-10min', [
            '../baselines/Informer2020_oldv_01112021/results/kktx376r',
            '../baselines/Informer2020_oldv_01112021/results/2zlrhwps',
            '../baselines/Informer2020_oldv_01112021/results/1539wdn3',
        ]),
        ('Informer_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/25x8f9p9',
            '../baselines/Informer2020_oldv_01112021/results/30uye92q',
        ]),
        ('Informer_res-6hour', [
            '../baselines/Informer2020_oldv_01112021/results/9ml388hu',
            '../baselines/Informer2020_oldv_01112021/results/2sy3j5ek',
            '../baselines/Informer2020_oldv_01112021/results/uck7opv9',
        ]),
        ('Informer_res-1hour', [
            '../baselines/Informer2020_oldv_01112021/results/3mqn27jq',
            '../baselines/Informer2020_oldv_01112021/results/3mtj2joy',
            '../baselines/Informer2020_oldv_01112021/results/14zaxps9',
        ]),
        ('Informer_res-10min', [
            '../baselines/Informer2020_oldv_01112021/results/1kksvl7a',
            '../baselines/Informer2020_oldv_01112021/results/22uy746i',
            '../baselines/Informer2020_oldv_01112021/results/3brsrtly',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/1alr9fy4',
            '../baselines/Graph-WaveNet/garage/tow81hg6',
            '../baselines/Graph-WaveNet/garage/2pwdubdh',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/1xyf3zlk',
            '../baselines/MTGNN/save/15ag0gwz',
            '../baselines/MTGNN/save/2tjsxon7',
        ]),
        ('Multitask_GWplusCKO_gate_res-all', [
            '../multitask/save/fxvkxbjf',
            '../multitask/save/270z524w',
            '../multitask/save/yqpzw7ks',
        ]),
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/xp1byhgr',
            '../multitask/save/2dn1mrs2',
            '../multitask/save/18bfjxow',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/cdfnxazc',
            '../baselines/koopmanAE/save/z6aam7w1',
            '../baselines/koopmanAE/save/2svf989n',
        ]),
    ],
    'Solar Energy 10min ': [
        ('HA_res-10min', [
            '../baselines/simple_baseline_res/solar_energy_10min/hist_avg',
        ]),
        ('Static_res-10min', [
            '../baselines/simple_baseline_res/solar_energy_10min/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/2dpfslt1',
            '../baselines/Informer2020_oldv_01112021/results/2b83tc1f',
            '../baselines/Informer2020_oldv_01112021/results/27sue0q4',
        ]),
        ('Informer_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/1qzvl4rk',
            '../baselines/Informer2020_oldv_01112021/results/2jr0r77w',
            '../baselines/Informer2020_oldv_01112021/results/vk4qtm3z',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/3ghyglw2',
            '../baselines/Graph-WaveNet/garage/1fzb2300',
            '../baselines/Graph-WaveNet/garage/35y538pj',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/16c68l5x',
            '../baselines/MTGNN/save/3varc7wl',
            '../baselines/MTGNN/save/1x94d9xs',
        ]),
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/iaks3xvj',
            '../multitask/save/20x6zx98',
            '../multitask/save/t1fx8qew',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/30pksyua',
            '../baselines/koopmanAE/save/6rdkavhs',
            '../baselines/koopmanAE/save/1aljcqrm',
        ]),
    ],
    'Solar Energy 10min 06': [
        ('HA_res-10min', [
            '../baselines/simple_baseline_res/solar_energy_10min_06/hist_avg',
        ]),
        ('Static_res-10min', [
            '../baselines/simple_baseline_res/solar_energy_10min_06/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/8766btoc',
            '../baselines/Informer2020_oldv_01112021/results/1z368rrq',
            '../baselines/Informer2020_oldv_01112021/results/18auwn0a',
        ]),
        ('Informer_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/2c3btfb1',
            '../baselines/Informer2020_oldv_01112021/results/17wacimz',
            '../baselines/Informer2020_oldv_01112021/results/1s8s5x3q',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/398wi31f',
            '../baselines/Graph-WaveNet/garage/298soqyu',
            '../baselines/Graph-WaveNet/garage/1lrzkgnj',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/63gt4eod',
            '../baselines/MTGNN/save/2sj5lgzt',
            '../baselines/MTGNN/save/25vka9q2',
        ]),
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/2hw8a9lh',
            '../multitask/save/12jcmx1y',
            '../multitask/save/3re23062',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/4rzzmyv9',
            '../baselines/koopmanAE/save/32yco4bs',
            '../baselines/koopmanAE/save/285ehytj',
        ]),
    ],
    'Solar Energy 10min 04': [
        ('HA_res-10min', [
            '../baselines/simple_baseline_res/solar_energy_10min_04/hist_avg',
        ]),
        ('Static_res-10min', [
            '../baselines/simple_baseline_res/solar_energy_10min_04/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/1ygiufak',
            '../baselines/Informer2020_oldv_01112021/results/ht3aoulp',
            '../baselines/Informer2020_oldv_01112021/results/1ghryx9e',
        ]),
        ('Informer_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/sqxv52hu',
            '../baselines/Informer2020_oldv_01112021/results/3kilfyql',
            '../baselines/Informer2020_oldv_01112021/results/2nenk94v',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/3lnrdiw6',
            '../baselines/Graph-WaveNet/garage/2uxyzt2j',
            '../baselines/Graph-WaveNet/garage/1uspfxza',
        ]),
        ('MTGNN_res-all', [
            '../baselines/MTGNN/save/rh7zn1sm',
            '../baselines/MTGNN/save/3fs0omdm',
            '../baselines/MTGNN/save/3lykz07p',
        ]),
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/29u0q3ze',
            '../multitask/save/1s5ywsj3',
            '../multitask/save/7pi0z9t1',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/jyxrn7sj',
            '../baselines/koopmanAE/save/2lvu2097',
            '../baselines/koopmanAE/save/1e3htsi4',
        ]),
    ],
    'Solar Energy 10min 02': [
        ('HA_res-10min', [
            '../baselines/simple_baseline_res/solar_energy_10min_02/hist_avg',
        ]),
        ('Static_res-10min', [
            '../baselines/simple_baseline_res/solar_energy_10min_02/static',
        ]),
        ('GRU_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/2ytdojri',
            '../baselines/Informer2020_oldv_01112021/results/26zo3r3f',
            '../baselines/Informer2020_oldv_01112021/results/1x9jcyet',
        ]),
        ('Informer_res-all', [
            '../baselines/Informer2020_oldv_01112021/results/ue00x2lm',
            '../baselines/Informer2020_oldv_01112021/results/3bd4dlbn',
            '../baselines/Informer2020_oldv_01112021/results/2tpl6oo1',
        ]),
        ('Graph WaveNet_res-all', [
            '../baselines/Graph-WaveNet/garage/1g1jp1wg',
            '../baselines/Graph-WaveNet/garage/1izg79gp',
            '../baselines/Graph-WaveNet/garage/134pywu1',
        ]),
        ('MTGNN_res-all', [
#             '../baselines/MTGNN/save/22nql2px',
            '../baselines/MTGNN/save/2uyjzfxk',
            '../baselines/MTGNN/save/32n9dy88',
            '../baselines/MTGNN/save/bnxs8jdl',
        ]),
        ('Multitask_GWplusCKO_gate_ups_ds_convfusion_res-all', [
            '../multitask/save/2uo6q9ka',
            '../multitask/save/3uu3igjl',
            '../multitask/save/2fshz5bx',
        ]),
        ('KoopmanAE_res-all', [
            '../baselines/koopmanAE/save/3w29g0pd',
            '../baselines/koopmanAE/save/2z9yd4pm',
            '../baselines/koopmanAE/save/1kq69pgl',
        ]),
    ],
}

def get_data_meta(dataname='NYCTaxi'):
    if dataname == 'NYCTaxi':
        # NYCTaxi
        test_dataset = NYCTaxiDataset(
            save_folder='../data/nyc_taxi/manhattan',
            input_len=1440,
            output_len=480,
            historical_temp_res=['6-hour', 'day'],
            forecast_temp_res=[],
            keep_ratio=0.8,
            trainval_ps=(0.6, 0.2),
            mask_seed=42,
            seq_diff=48,
            data_type='test',
            scale=True,
            scaler=None,
            return_delta=False
        )
    elif dataname in ['NYCTaxiFull', 'NYCTaxi06', 'NYCTaxi04', 'NYCTaxi02']:
        # NYCTaxi
        test_dataset = NYCTaxiDataset(
            save_folder='../data/nyc_taxi/manhattan',
            input_len=1440,
            output_len=480,
            historical_temp_res=['6-hour', 'day'],
            forecast_temp_res=[],
            keep_ratio=1,
            trainval_ps=(0.6, 0.2),
            mask_seed=42,
            seq_diff=48,
            data_type='test',
            scale=True,
            scaler=None,
            return_delta=False
        )
    elif dataname == 'NYCTaxiGreen':
        # NYCTaxi
        test_dataset = NYCTaxiDataset(
            save_folder='../data/nyc_taxi/sel_green',
            input_len=1440,
            output_len=480,
            historical_temp_res=['6-hour', 'day'],
            forecast_temp_res=[],
            keep_ratio=0.8,
            trainval_ps=(0.6, 0.2),
            mask_seed=42,
            seq_diff=48,
            data_type='test',
            scale=True,
            scaler=None,
            return_delta=False
        )
    elif dataname in ['NYCTaxiGreenFull', 'NYCTaxiGreen06', 'NYCTaxiGreen04', 'NYCTaxiGreen02']:
        # NYCTaxi
        test_dataset = NYCTaxiDataset(
            save_folder='../data/nyc_taxi/sel_green',
            input_len=1440,
            output_len=480,
            historical_temp_res=['6-hour', 'day'],
            forecast_temp_res=[],
            keep_ratio=1.0,
            trainval_ps=(0.6, 0.2),
            mask_seed=42,
            seq_diff=48,
            data_type='test',
            scale=True,
            scaler=None,
            return_delta=False
        )
    elif dataname.startswith('Solar Energy 10min '):
        test_dataset = NYCTaxiDataset(
            save_folder='../data/solar_energy_10min',
            input_len=1440,
            output_len=432,
            historical_temp_res=['1-hour', '6-hour'],
            forecast_temp_res=[],
            keep_ratio=1,
            trainval_ps=(0.6, 0.2),
            mask_seed=42,
            seq_diff=36,
            data_type='test',
            scale=True,
            scaler=None,
            return_delta=False
        )
    else:
        raise NotImplementedError()
    node_type = test_dataset.all_node_types
    nodenum = len(test_dataset.all_node_types)
    scaler = test_dataset.scaler
    return node_type, nodenum, scaler, res_configs_dict[dataname]


def res_to_fdims(resname, reslist=('day', '6hour', '30min')):
    if resname == 'all':
        return None
    else:
        rlist = list(reslist)[::-1]
        assert resname in rlist
        s = rlist.index(resname)
        return s, s+1


def res_to_aggres(resname, outputresname, baseresname='30min'):
    if baseresname == '30min':
        d = {
            '30min': 1,
            '6hour': 6*2,
            'day': 24*2
        }
    elif baseresname == '5min':
        d = {
            '5min': 1,
            '1hour': 12,
            '4hour': 4*12
        }
    elif baseresname == '10min':
        d = {
            '10min': 1,
            '1hour': 6,
            '6hour': 36
        }
    elif baseresname == '1hour':
        d = {
            '1hour': 1,
            '6hour': 6,
            'day': 24
        }
    assert d[resname] % d[outputresname] == 0
    return d[resname] // d[outputresname]


def parse_model_results(dataname, modelnames, res_list=('day', '6hour', '30min'), horizons=(1, 3, 6, 10), threshold=10, corresponding=False, target_node_res=0, error_range='agg'):
    node_type, nodenum, scaler, resconfs = get_data_meta(dataname)
    resconfs_dict = {k: v for (k, v) in resconfs}
#     print(list(resconfs_dict.keys()))
    target_res_list = ['all'] + list(res_list)
#     target_res_list = ['all']
    for modelbasename in modelnames:
        for target_res in target_res_list:
            modelname = modelbasename + '_res-{}'.format(target_res)
            try:
                assert modelname in resconfs_dict
            except:
                pass
#                 print(modelname)
          
    all_results = []
    for modelbasename in modelnames:
        for horizon_d in horizons:
            res_row = {}
            for target_res in target_res_list:
                modelname = modelbasename + '_res-{}'.format(target_res)
                if modelname not in resconfs_dict:
                    continue
                resfolders = resconfs_dict[modelname]
                if target_res == 'all':
                    output_ress = res_list
                else:
                    output_ress = [target_res]
                for output_res in output_ress:
                    if target_res != 'all':
                        output_dims = None
                    else:
                        output_dims = res_to_fdims(output_res, res_list)
#                     if modelbasename.startswith('Multitask'):
#                         aval_pred_res_list = [output_res]
#                     else:
                    if corresponding and (modelbasename not in ['HA', 'Static']):
                        aval_pred_res_list = [output_res]
                    else:
                        max_aval_pred_res_idx = res_list.index(output_res)
                        aval_pred_res_list = res_list[:max_aval_pred_res_idx+1]
                    if modelbasename.startswith('Multitask'):
                        agg_base_res = output_res
                    else:
                        agg_base_res = res_list[-1]
                    for idx, aval_pred_res in enumerate(aval_pred_res_list):
                        evals = []
                        for resfolder in resfolders:
                            if error_range == 'agg':
                                start_t, end_t = 0, horizon_d
                            elif error_range == 'point':
                                start_t, end_t = horizon_d - 1, horizon_d
                            evals.append(eval_model(modelname, resfolder, scaler, nodenum, node_type, target_node_res, threshold=threshold, start_t=start_t, 
                                                    end_t=end_t, output_dims=output_dims, agg='mean', agg_res=res_to_aggres(aval_pred_res, agg_base_res, res_list[-1]),
                                                   agg_res_to_base=res_to_aggres(output_res, res_list[-1], res_list[-1])))
                        agg_evals = {}
                        for k in evals[0]:
                            k_list = [ei[k] for ei in evals]
                            agg_evals[k] = (np.mean(k_list), np.std(k_list))
                        if aval_pred_res not in res_row:
                            res_row[aval_pred_res] = []
                        res_row[aval_pred_res].append(agg_evals)
#             print(modelbasename, horizon_d)

            win_results = {}
            for res in res_row: # select the model winning most times
                win_times = np.zeros(len(res_row[res]))
                win_idx = {}
                for metric in res_row[res][0]:
                    best_idx = np.argmin([res_row[res][idx][metric][0] for idx in range(len(res_row[res]))])
                    win_times[best_idx] += 1
                    win_idx[metric] = best_idx
                if (win_times == np.max(win_times)).sum() > 1:
                    final_idx = win_idx['rmse']
                else:
                    final_idx = np.argmax(win_times)
                win_results[res] = res_row[res][final_idx]
            all_results.append((modelbasename, horizon_d, win_results))
        
    return all_results


from functools import partial


def latex_results_multihorizon(all_results, modelnames, reslist=('day', '6hour', '30min'), metriclist=('rmse', 'mae', 'mape'), num_decimals=3, horizon_name_dict=None, model_name_dict=None,
                              with_relative=False):
    reslist = list(reslist)
    all_results_rows = []
    for modelname, horizon, resdict in all_results:
        for resolution in reslist:
            for metric in metriclist:
                val = resdict[resolution][metric]
                val_str = '{:.0{num_decimals}f}({:.0{num_decimals}f})'.format(val[0], val[1], num_decimals=num_decimals)
                all_results_rows.append((modelname, horizon, resolution, metric.upper(), val_str))

    df_all_results = pd.DataFrame(all_results_rows, columns=['Model', 'Horizon', 'Resolution', 'Metric', 'mean(std)']).set_index('Model').sort_values(by=['Resolution','Horizon'])
    df_all_results.drop(columns=['Resolution'], inplace=True)
    # df_all_results
    df_all_results = df_all_results.pivot(columns=['Horizon',  'Metric'], values='mean(std)').reindex(modelnames)
    if horizon_name_dict is not None:
        df_all_results.rename(columns=horizon_name_dict, inplace=True)
    if model_name_dict is not None:
        df_all_results.rename(index=model_name_dict, inplace=True)
    # df_all_results
    column_format = 'c|'
    for gi in range(len(df_all_results.columns) // len(metriclist)):
        column_format += ('c'*len(metriclist) + '|')
        
    # mark values with minimum avg
    def bold_formatter(x, value, pval, num_decimals=num_decimals):
        xv = float(re.search(r'\d+\.\d+', x)[0])
        if (not with_relative) or (xv == pval):
            if round(xv, num_decimals) == round(value, num_decimals):
                return f'\\textbf{{{x}}}'
            else:
                return x
        else:
            pstr = '{:+.2f}'.format(((xv - pval) / pval) * 100)
            if round(xv, num_decimals) == round(value, num_decimals):
                s = f'\\begin{{tabular}}[c]{{@{{}}c@{{}}}}\\textbf{{{x}}}\\\\{pstr}\%\end{{tabular}}'
#                 print(s)
                return s
            else:
                s = f'\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{x}\\\\{pstr}\%\end{{tabular}}'
#                 print(s)
                return s
        
    formatters = {}
    for column in df_all_results.columns:
        all_vals = df_all_results[column].apply(lambda x: float(re.search(r'\d+\.\d+', x)[0]))
        best_val = all_vals.min()
        pval = all_vals['\modelshortname']
        formatters[column] = partial(bold_formatter, value=best_val, pval=pval)
#     formatters = {column: partial(bold_formatter, value=df_all_results[column].apply(lambda x: float(x.split('(')[0])).min()) for column in df_all_results.columns}
        
    latex_str = df_all_results.to_latex(multicolumn_format='c|', multirow=True, index_names=False, column_format=column_format, formatters=formatters, escape=False)
    latex_lines = latex_str.split('\n')
    latex_lines.insert(-4, '\midrule')
    return '\n'.join(latex_lines)
#     return df_all_results


def get_dataname_with_keep_ratio(basename, keep_ratio):
    if keep_ratio == 1:
        suffix = 'Full'
    elif keep_ratio == 0.8:
        suffix = ''
    else:
        suffix = '{:02d}'.format(int(keep_ratio * 10))
    return basename + suffix


def collect_results(basenames, keep_ratios):
    all_results = []
    for basename in basenames:
        if basename in ['NYCTaxi', 'NYCTaxiGreen']:
            modelnames=[
                'HA', 'Static', 
                'GRU', 'Informer_nodt', 'Graph WaveNet', 'MTGNN',
                'Multitask_GWplusCKO_gate_ups_ds_convfusion', 'KoopmanAE'
            ]
            for keep_ratio in keep_ratios:
                all_results_sn = parse_model_results(
                    dataname=get_dataname_with_keep_ratio(basename, keep_ratio),
                    modelnames=modelnames,
    #                 horizons=[1, 12, 48, 240, 480],
                    horizons=[1, 12, 48, 480],
                    res_list=('30min',),
                    threshold=1e-6,
                    corresponding=True,
                    target_node_res=0
                )
                all_results.append((basename, keep_ratio, all_results_sn))
        elif basename in ['Solar Energy 10min ']:
            modelnames=[
                'HA', 'Static', 
                'GRU', 'Informer', 'Graph WaveNet', 'MTGNN',
                'Multitask_GWplusCKO_gate_ups_ds_convfusion', 'KoopmanAE'
            ]
            for keep_ratio in keep_ratios:
#                 assert keep_ratio == 1
                all_results_sn = parse_model_results(
                    dataname=get_dataname_with_keep_ratio(basename, keep_ratio),
                    modelnames=modelnames,
    #                 horizons=[1, 12, 48, 240, 480],
                    horizons=[1, 6, 36, 432],
                    res_list=('10min',),
                    threshold=1e-6,
                    corresponding=True,
                    target_node_res=1
                )
                all_results.append((basename, keep_ratio, all_results_sn))
        else:
            raise NotImplementedError
    return all_results


def latex_results_multihorizon_multikr(all_results, modelnames, datanames, reslist=('day', '6hour', '30min'), metriclist=('rmse', 'mae', 'mape'), 
                                       num_decimals=3, horizon_name_dict=None, model_name_dict=None, data_name_dict=None, drop_obs_ratio=False, horizon_list=None,
                                       dropdata=False, drophorizon=False):
    reslist = list(reslist)
    all_results_rows = []
    for dataset, kr, all_results_list in all_results:
        for modelname, horizon, resdict in all_results_list:
            if horizon_list is not None:
                if horizon not in horizon_list:
                    continue
            if len(resdict) == 0:
                continue
            for resolution in reslist:
                for metric in metriclist:
                    val = resdict[resolution][metric]
                    if val[1] == 0:
                        val_str = '{:.0{num_decimals}f}'.format(val[0], num_decimals=num_decimals)
                    else:
                        val_str = '{:.0{num_decimals}f}({:.0{num_decimals}f})'.format(val[0], val[1], num_decimals=num_decimals)
                    all_results_rows.append((dataset, modelname, kr, horizon, resolution, metric.upper(), val_str))

    df_all_results = pd.DataFrame(all_results_rows, columns=['Data', 'Model', 'Obs Ratio', 'Horizon', 'Resolution', 'Metric', 'mean(std)']).sort_values(by=['Resolution','Horizon'])
    df_all_results.drop(columns=['Resolution'], inplace=True)
    # df_all_results
    if drop_obs_ratio:
        df_all_results.drop(columns=['Obs Ratio'], inplace=True)
        df_all_results = df_all_results.pivot(columns=['Model',], values='mean(std)', index=['Data', 'Horizon', 'Metric']).sort_index(level=['Data', 'Horizon'], ascending=[True, True]).reindex(modelnames, axis=1).reindex(datanames, level='Data')
        column_format = 'cccc'
    else:
        df_all_results = df_all_results.pivot(columns=['Model',], values='mean(std)', index=['Data', 'Obs Ratio', 'Horizon', 'Metric']).sort_index(level=['Data', 'Obs Ratio', 'Horizon'], ascending=[True, False, True]).reindex(modelnames, axis=1).reindex(datanames, level='Data')
        column_format = 'ccccc'
    if horizon_name_dict is not None:
        df_all_results.rename(index=horizon_name_dict, level='Horizon', inplace=True)
    if data_name_dict is not None:
        df_all_results.rename(index=data_name_dict, level='Data', inplace=True)
    if model_name_dict is not None:
        df_all_results.rename(columns=model_name_dict, inplace=True)
    # df_all_results
    for gi in range(len(df_all_results.columns) // len(metriclist)):
        column_format += ('c'*len(metriclist))
        
    # mark values with minimum avg
    def bold_formatter(x, value, subvalue, num_decimals=num_decimals):
        if '(' in x:
            xv = float(x.split('(')[0])
        else:
            xv = float(x)
        if round(xv, num_decimals) == round(value, num_decimals):
            return f'\\textbf{{{x}}}'
        elif round(xv, num_decimals) == round(subvalue, num_decimals):
            return f'\\underline{{\\textit{{{x}}}}}'
        else:
            return x
        
#     print(df_all_results)
        
    df_all_results = df_all_results.T
    formatters = {}
    imp_p_c, imp_p_gw_c = [], []
    for column in df_all_results.columns:
        values = df_all_results[column].apply(lambda x: float(re.search(r'\d+\.\d+', x)[0]))
        best_value = values.min()
        pval = values['\modelshortname']
        gwval = values['Graph WaveNet']
        best_bs_value = values.drop(labels=['\modelshortname']).min()
        imp_p = (pval - best_bs_value) / best_bs_value
        imp_p_c.append('{:.2f}\%'.format(imp_p * 100))
        imp_p_gw = (pval - gwval) / gwval
        imp_p_gw_c.append('{:.2f}\%'.format(imp_p_gw * 100))
        formatters[column] = partial(bold_formatter, value=best_value, subvalue=best_bs_value)
    for key in df_all_results.columns.values:
        if key in formatters:
            df_all_results[key] = df_all_results[key].apply(formatters[key])
#     print(df_all_results.shape)
#     print(pd.DataFrame([imp_p_c], columns=df_all_results.columns))
    df_all_results = df_all_results.append(pd.DataFrame([imp_p_c], columns=df_all_results.columns).rename(index={0: 'RelErr'}))
    column_format += 'c'
    df_all_results = df_all_results.append(pd.DataFrame([imp_p_gw_c], columns=df_all_results.columns).rename(index={0: 'RelErrGW'}))
    column_format += 'c'
            
    outdf = df_all_results.T
    if dropdata:
        outdf = outdf.droplevel(level='Data', axis=0)
        column_format = column_format[1:]
    if drophorizon:
        outdf = outdf.droplevel(level='Horizon', axis=0)
        column_format = column_format[1:]
    latex_str = outdf.to_latex(multicolumn_format='c|', multirow=True, index_names=True, column_format=column_format, escape=False)
    return latex_str
        
#     latex_str = df_all_results.to_latex(multicolumn_format='c|', multirow=True, index_names=True, column_format=column_format, formatters=formatters, escape=False)
#     latex_lines = latex_str.split('\n')
#     latex_lines.insert(-4, '\midrule')
#     latex_lines ='\n'.join(latex_lines)
#     return latex_l
#     return df_all_results.applymap(lambda x: float(re.search(r'\d+\.\d+', x)[0])), latex_lines
#     return df_all_results