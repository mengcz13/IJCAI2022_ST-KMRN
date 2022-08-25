import os
import torch
import numpy as np
import argparse
import time
from util import *
from base_models.mtgnn.multires_data import load_dataset as multires_load_dataset
from base_models.mtgnn.trainer import Trainer
from tqdm import tqdm
from copy import deepcopy

from multitask_model import MultitaskModel
from base_models import mtgnn_enc, mtgnn_dec

import wandb


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str,
                    default='data/METR-LA', help='data path')

parser.add_argument('--adj_data', type=str,
                    default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True,
                    help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,
                    help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool,
                    default=False, help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True,
                    help='whether to do curriculum learning')

parser.add_argument('--gcn_depth', type=int, default=2,
                    help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=207,
                    help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
# parser.add_argument('--subgraph_size', type=int, default=20, help='k')
# parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--subgraph_size', type=int, default=32, help='k')
parser.add_argument('--node_dim', type=int, default=64, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int,
                    default=1, help='dilation exponential')

parser.add_argument('--conv_channels', type=int,
                    default=32, help='convolution channels')
parser.add_argument('--residual_channels', type=int,
                    default=32, help='residual channels')
parser.add_argument('--skip_channels', type=int,
                    default=64, help='skip channels')
# parser.add_argument('--end_channels', type=int,
#                     default=128, help='end channels')

# multitask attn params
parser.add_argument('--mt_attn_embed_dim', type=int, default=2048)
parser.add_argument('--mt_attn_num_heads', type=int, default=8)
parser.add_argument('--mt_attn_dropout', type=float, default=0.1)


parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--out_dim', type=int, default=2, help='output dimension')
parser.add_argument('--seq_in_len', type=int, default=12,
                    help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=12,
                    help='output sequence length')

parser.add_argument('--layers', type=int, default=3, help='number of layers')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float,
                    default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--step_size1', type=int, default=2500, help='step_size')
parser.add_argument('--step_size2', type=int, default=100, help='step_size')


parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=101, help='random seed')
parser.add_argument('--save', type=str, default='./save/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')

parser.add_argument('--num_split', type=int, default=1,
                    help='number of splits for graphs')

parser.add_argument('--runs', type=int, default=1, help='number of runs')

parser.add_argument('--seq_diff', type=int, default=48)
parser.add_argument('--test_seq_diff', type=int, default=48)
parser.add_argument('--test_int', type=int, default=1)

parser.add_argument('--eval_saved_model', type=str, default=None)

parser.add_argument('--target_res', type=str,
                    default='30min', help='30min/6-hour/day/all')
parser.add_argument('--expname_suffix', type=str, default='')
parser.add_argument('--resolution_type', type=str, default='agg')

parser.add_argument('--single_res_input_output', action='store_true',
                    help='true if used for verification on existing exps')


args = parser.parse_args()
torch.set_num_threads(3)


def main(runid):
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    # dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    dataloader = multires_load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size, args)
    scaler = dataloader['scaler']

    # predefined_A = load_adj(args.adj_data)
    predefined_A = dataloader['predefined_A']
    args.num_nodes = predefined_A.shape[0]
    predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)

    args.in_dim = dataloader['in_dim']
    args.out_dim = dataloader['out_dim']
    res_num = dataloader['res_num']

    expname = 'Multitask-MTGNN'
    if args.expname_suffix != '':
        expname += '_{}'.format(args.expname_suffix)
    wandb.init(project='multires_st', config=args, name=expname)

    # if args.load_static_feature:
    #     static_feat = load_node_feature('data/sensor_graph/location.csv')
    # else:
    #     static_feat = None

    base_model_enc_list = []
    base_model_dec_list = []
    for seq_in_len, seq_out_len in zip(dataloader['x_lens'], dataloader['y_lens']):
        base_model_enc = mtgnn_enc(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, predefined_A=predefined_A,
                    dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim,
                    dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels,
                    # end_channels=args.end_channels,
                    end_channels=args.mt_attn_embed_dim,
                    seq_length=seq_in_len, in_dim=args.in_dim, out_dim=None,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha,
                    layer_norm_affline=True)
        base_model_enc_list.append(base_model_enc)

        base_model_dec = mtgnn_dec(
            # in_channels=args.end_channels,
            # end_channels=args.end_channels,
            in_channels=args.mt_attn_embed_dim,
            end_channels=args.mt_attn_embed_dim,
            seq_out_len=seq_out_len,
            out_dim=args.out_dim
        )
        base_model_dec_list.append(base_model_dec)
    model = MultitaskModel(
        enc_base_model_list=base_model_enc_list,
        dec_base_model_list=base_model_dec_list,
        self_attn_embed_dim=args.mt_attn_embed_dim,
        self_attn_num_heads=args.mt_attn_num_heads,
        dropout=args.mt_attn_dropout
    )

    wandb.watch(model)

    print(args)
    # print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip,
                     args.step_size1, args.seq_out_len, scaler, device, args.cl)

    save_dir = os.path.join(args.save, wandb.run.id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def run_test(epoch):
        outputs = []
        # realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = []

        for iter, (xlist, ylist) in tqdm(enumerate(dataloader['test_loader']), total=len(dataloader['test_loader'])):
            testx = [torch.Tensor(x).to(device) for x in xlist]
            testy = [torch.Tensor(y) for y in ylist]
            with torch.no_grad():
                preds = engine.model(testx)
            outputs.append([p.detach().cpu() for p in preds])
            realy.append([y.cpu() for y in testy])

        yhat_list = [torch.cat(output, dim=0) for output in list(zip(*outputs))]
        realy_list = [torch.cat(y, dim=0) for y in list(zip(*realy))]

        for res_id, (pred, realy) in enumerate(zip(yhat_list, realy_list)):
            print(pred.shape, realy.shape)
            tmae, tmape, trmse = metric(pred, realy)
            wandb.log({
                'Res-{} Test MAE'.format(res_id): tmae,
                'Res-{} Test MAPE'.format(res_id): tmape,
                'Res-{} Test RMSE'.format(res_id): trmse,
                'Epoch': epoch
            })

            np.save(os.path.join(save_dir, 'res-{}-pred.npy'.format(res_id)), pred.numpy())
            np.save(os.path.join(save_dir, 'res-{}-true.npy'.format(res_id)), realy.numpy())

    if args.eval_saved_model is not None:
        engine.model.load_state_dict(torch.load(args.eval_saved_model))
        run_test(epoch=0)
    else:
        print("start training...", flush=True)
        his_loss = []
        val_time = []
        train_time = []
        minl = 1e5
        # dataloader['train_loader'].shuffle()
        for i in range(1, args.epochs+1):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            # dataloader['train_loader'].shuffle()
            for iter, (xlist, ylist) in tqdm(enumerate(dataloader['train_loader']), total=len(dataloader['train_loader'])):
                # if iter > 3:
                #     break
                trainx = [torch.Tensor(x).to(device) for x in xlist]
                trainy = [torch.Tensor(y).to(device) for y in ylist]
                if iter % args.step_size2 == 0:
                    perm = np.random.permutation(range(args.num_nodes))
                num_sub = int(args.num_nodes/args.num_split)
                for j in range(args.num_split):
                    if j != args.num_split-1:
                        id = perm[j * num_sub:(j + 1) * num_sub]
                    else:
                        id = perm[j * num_sub:]
                    id = torch.tensor(id).to(device)
                    tx = [x[:, :, id, :] for x in trainx]
                    ty = [y[:, :, id, :] for y in trainy]
                    metrics = engine.train(tx, ty, id)
                    train_loss.append(metrics[0])
                    train_mape.append(metrics[1])
                    train_rmse.append(metrics[2])
                if iter % args.print_every == 0:
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(log.format(
                        iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
                    wandb.log({
                        'Step Train Loss': train_loss[-1],
                        'Step Train MAPE': train_mape[-1],
                        'Step Train RMSE': train_rmse[-1],
                    })
            t2 = time.time()
            train_time.append(t2-t1)
            # validation
            valid_loss = []
            valid_mape = []
            valid_rmse = []

            s1 = time.time()
            for iter, (xlist, ylist) in tqdm(enumerate(dataloader['val_loader']), total=len(dataloader['val_loader'])):
                # if iter > 3:
                #     break
                testx = [torch.Tensor(x).to(device) for x in xlist]
                testy = [torch.Tensor(y).to(device) for y in ylist]
                metrics = engine.eval(testx, testy)
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i, (s2-s1)))
            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse,
                             mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
            wandb.log({
                'Epoch': i,
                'Train Loss': mtrain_loss, 'Train RMSE': mtrain_rmse,
                'Valid Loss': mvalid_loss, 'Valid MAPE': mvalid_mape, 'Valid RMSE': mvalid_rmse
            })

            if mvalid_loss < minl:
                torch.save(engine.model.state_dict(), os.path.join(
                    save_dir, "best.pth"))
                minl = mvalid_loss

            # testing
            if (i % args.test_int == 0) or (i == args.epochs):
                print('Testing for epoch {}...'.format(i))
                ckpt = deepcopy(engine.model.state_dict())
                bestid = np.argmin(his_loss)
                engine.model.load_state_dict(torch.load(os.path.join(
                    save_dir, "best.pth")))
                run_test(epoch=i)
                engine.model.load_state_dict(ckpt)

        print(
            "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        print("Training finished")
        print("The valid loss on best model is",
              str(round(his_loss[bestid], 4)))


if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    for i in range(args.runs):
        main(i)
