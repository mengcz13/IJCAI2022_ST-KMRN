import argparse
import os
import torch
import numpy as np

from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer',help='model of the experiment')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='location of the data file')    
parser.add_argument('--features', type=str, default='M', help='features [S, M]')
parser.add_argument('--target', type=str, default='OT', help='target feature')

parser.add_argument('--seq_len', type=int, default=96, help='input series length')
parser.add_argument('--label_len', type=int, default=48, help='help series length')
parser.add_argument('--pred_len', type=int, default=24, help='predict series length')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='prob sparse factor')

parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention [prob, full]')
parser.add_argument('--embed', type=str, default='fixed', help='embedding type [fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='control attention output')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

parser.add_argument('--itr', type=int, default=1, help='each params run iteration')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

parser.add_argument('--seq_diff', type=int, default=1)
parser.add_argument('--test_seq_diff', type=int, default=48)
parser.add_argument('--keep_ratio', type=float, default=0.8)
parser.add_argument('--target_res', type=str, default='30min', help='30min/6-hour/day/all')
parser.add_argument('--with_ts_delta', action='store_true')
parser.add_argument('--no_input_mask', dest='with_input_mask', action='store_false', help='set if include mask in input')
parser.add_argument('--single_input_res', action='store_true', help='only the target resolution in input')
parser.add_argument('--resolution_type', type=str, default='agg')

parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--expname_suffix', type=str, default='')

parser.add_argument('--load_ckpt', type=str, default='')
parser.add_argument('--notrain', action='store_true')

parser.add_argument('--chrono_arr_cats', type=str, default='mdwh')

args = parser.parse_args()

# initialize seeds
if args.seed > 0:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1]},
    'NYCTaxi_30d-7d-1stride':{'data':'sampled_dense_multires_seqs_30d-7d_diff1d.npz','T':None,'M':[2,2,2],'S':[1,1,1]},
    'NYCTaxi_48h-24h-1stride':{'data':'sampled_dense_multires_seqs_48h-24h_diff1d.npz','T':None,'M':[2,2,2],'S':[1,1,1]},
    'NYCTaxi_30d-7d':{'data':'sampled_dense_multires_seqs_30d-7d_diff1_node2.npz','T':None,'M':[2,2,2],'S':[1,1,1]},
    'NYCTaxi_48h-24h':{'data':'sampled_dense_multires_seqs_48h-24h_diff1_node2.npz','T':None,'M':[2,2,2],'S':[1,1,1]},
    'NYCTaxi_7d-7d-1stride':{'data':'sampled_dense_multires_seqs.npz','T':None,'M':[2,2,2],'S':[1,1,1]},
    'NYCTaxi_Multires_6h-1d':{'data':None,'T':None,'M':[1404,156,156],'S':None},
    'NYCTaxiGreen_Multires_6h-1d':{'data':None,'T':None,'M':[88*9*2,88*2,88*2],'S':None},
    'NYCTaxiFHV_Multires_6h-1d':{'data':None,'T':None,'M':[1404,156,156],'S':None},
    'SolarEnergy_Multires_6h-1d':{'data':None,'T':None,'M':[1323,147,147],'S':None},
    'SolarEnergy10min_Multires_1h-6h':{'data':None,'T':None,'M':[1323,147,147],'S':None},
    'SolarEnergyAr_Multires_6h-1d':{'data':None,'T':None,'M':[96*9,96,96],'S':None},
    'METR-LA_Multires_1h-4h':{'data':None,'T':None,'M':[9*228,228,228],'S':None},
    'PEMS-BAY_Multires_1h-4h':{'data':None,'T':None,'M':[9*357,357,357],'S':None},
    'ECL_Multires_6h-1d':{'data':None,'T':None,'M':[9*321,321,321],'S':None}
}

input_res_num = 3
if args.single_input_res:
    input_res_num = 1
output_res_num = 1
if args.target_res == 'all':
    output_res_num = 3
input_dim_each_f_n = (1 + int(args.with_ts_delta) + int(args.with_input_mask)) * input_res_num
output_dim_each_f_n = output_res_num

for dataname in ['NYCTaxi_Multires_6h-1d', 'NYCTaxiGreen_Multires_6h-1d', 'NYCTaxiFHV_Multires_6h-1d', 'SolarEnergy_Multires_6h-1d', 'SolarEnergy10min_Multires_1h-6h', 'SolarEnergyAr_Multires_6h-1d', 'METR-LA_Multires_1h-4h', 'PEMS-BAY_Multires_1h-4h', 'ECL_Multires_6h-1d']:
    f_n_num = data_parser[dataname]['M'][2]
    data_parser[dataname]['M'] = [f_n_num * input_dim_each_f_n, f_n_num * output_dim_each_f_n, f_n_num * output_dim_each_f_n]

# if not args.with_ts_delta:
#     data_parser['NYCTaxi_Multires_6h-1d']['M'][0] = 936
#     data_parser['NYCTaxiGreen_Multires_6h-1d']['M'][0] = 88*6*2
#     data_parser['NYCTaxiFHV_Multires_6h-1d']['M'][0] = 936
#     data_parser['SolarEnergy_Multires_6h-1d']['M'][0] = 882
#     data_parser['SolarEnergy10min_Multires_1h-6h']['M'][0] = 882
#     data_parser['SolarEnergyAr_Multires_6h-1d']['M'][0] = 6*96
#     data_parser['METR-LA_Multires_1h-4h']['M'][0] = 6*228
#     data_parser['PEMS-BAY_Multires_1h-4h']['M'][0] = 6*357
#     data_parser['ECL_Multires_6h-1d']['M'][0] = 6*321

# if args.target_res == 'all':
#     data_parser['NYCTaxi_Multires_6h-1d']['M'][1:3] = [468,468]
#     data_parser['NYCTaxiGreen_Multires_6h-1d']['M'][1:3] = [88*3*2,88*3*2]
#     data_parser['NYCTaxiFHV_Multires_6h-1d']['M'][1:3] = [468,468]
#     data_parser['SolarEnergy_Multires_6h-1d']['M'][1:3] = [441,441]
#     data_parser['SolarEnergy10min_Multires_1h-6h']['M'][1:3] = [441,441]
#     data_parser['SolarEnergyAr_Multires_6h-1d']['M'][1:3] = [3*96,3*96]
#     data_parser['METR-LA_Multires_1h-4h']['M'][1:3] = [3*228,3*228]
#     data_parser['PEMS-BAY_Multires_1h-4h']['M'][1:3] = [3*357,3*357]
#     data_parser['ECL_Multires_6h-1d']['M'][1:3] = [3*321,3*321]
#     if (args.seq_len < 48) or (args.label_len < 48):
#         data_parser['PEMS-BAY_Multires_1h-4h']['M'][1:3] = [2*357,2*357]
#         data_parser['PEMS-BAY_Multires_1h-4h']['M'][0] = 4*357

# if args.resolution_type == 'sample':
#     data_parser['NYCTaxi_Multires_6h-1d']['M'][0] = 2*156
#     data_parser['SolarEnergy_Multires_6h-1d']['M'][0] = 2*147
#     data_parser['SolarEnergyAr_Multires_6h-1d']['M'][0] = 2*96
#     data_parser['METR-LA_Multires_1h-4h']['M'][0] = 2*228
#     data_parser['PEMS-BAY_Multires_1h-4h']['M'][0] = 2*357


if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    print(data_info[args.features])
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

Exp = Exp_Informer

for ii in range(args.itr):
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_eb{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.embed, args.des, ii)

    exp = Exp(args)

    if not args.notrain:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)