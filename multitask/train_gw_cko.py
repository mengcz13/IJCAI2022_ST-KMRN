import os
import torch
import numpy as np
import pickle as pkl
import argparse
import time
import util
import base_models.gw_cko.multires_data as multires_data
from base_models.gw_cko.engine import trainer
from base_models.graph_wavenet.model import gwnet_enc, gwnet_dec
from base_models.gw_cko.multitask_model import MultitaskModel 
from base_models.gw_cko.engine import move_data_list

import wandb
from tqdm import tqdm
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--out_dim', type=int, default=2)
parser.add_argument('--seq_in_len', type=int, default=2)
parser.add_argument('--seq_out_len', type=int, default=2)
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./save',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--seq_diff', type=int, default=48)
parser.add_argument('--test_seq_diff', type=int, default=48, help='only used for test set to reduce num of seqs for saving')
parser.add_argument('--eval_saved_model', type=str, default=None)
parser.add_argument('--test_int', type=int, default=1, help='test every n epochs, default 1')

parser.add_argument('--patience', type=int, default=-1, help='work when set to positive')
parser.add_argument('--patience_th', type=float, default=1e-4)

parser.add_argument('--target_res', type=str, default='30min', help='30min/6-hour/day/all')
parser.add_argument('--resolution_type', type=str, default='agg')

parser.add_argument('--single_res_input_output', action='store_true', help='true if used for verification on existing exps')

# multitask attn params
parser.add_argument('--mt_attn_num_heads', type=int, default=8)
parser.add_argument('--mt_attn_dropout', type=float, default=0.1)

# up/down sampling options
parser.add_argument('--use_downsampling_pred', action='store_true')
parser.add_argument('--use_upsampling_pred', action='store_true')
parser.add_argument('--upsampler', type=str, default='simple')
parser.add_argument('--save_auxiliary_outputs', action='store_true')
parser.add_argument('--updown_fusion', type=str, default='param', help='param: c parameters for c sources; conv: b x t x n x c params from conv for c sources')
parser.add_argument('--agg_res_hdim', type=int, default=32)

# debugging flags
parser.add_argument('--debug', action='store_true')

# loss function
parser.add_argument('--loss_type', type=str, default='masked_mae')

# CKO params
parser.add_argument('--cko_lrate', type=float, default=1e-4)
parser.add_argument('--cko_wdecay', type=float, default=0)
parser.add_argument('--pstep', type=int, default=2, help='number of propagation steps in GNN model')
parser.add_argument('--nf_relation', type=int, default=120, help='length of relation encoding')
parser.add_argument('--nf_particle', type=int, default=100, help='length of object encoding')
parser.add_argument('--nf_effect', type=int, default=100, help='length of effect encoding')
parser.add_argument('--g_dim', type=int, default=32, help='dimention of latent linear dynamics')
parser.add_argument('--fit_type', default='structured',
                    help="what is the structure of AB matrix in koopman: structured | unstructured | diagonal")
parser.add_argument('--cko_lambda_loss_metric', type=float, default=0.3)
parser.add_argument('--I_factor', type=float, default=10)
parser.add_argument('--enc_dec_type', type=str, default='pn', help='pn: PropNet; mlp: MLP')
parser.add_argument('--chrono_arr_cats', type=str, default='mdwh', help='mdwh/mdwhm')
parser.add_argument('--chrono_sincos_emb', action='store_true')
parser.add_argument('--cko_dl_gate', action='store_true')
parser.add_argument('--cko_dl_gate_hdim', type=int, default=32)

parser.add_argument('--keep_ratio', type=float, default=0.8)

# ablation
parser.add_argument('--no_multitask_attn', dest='with_multitask_attn',  action='store_false')

args = parser.parse_args()


def main():
    #set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    # dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    dataloader = multires_data.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, args)
    adj_mx = multires_data.load_adj(dataloader['predefined_A'], args.adjtype)
    args.in_dim = dataloader['in_dim']
    args.out_dim = dataloader['out_dim']
    args.num_nodes = dataloader['predefined_A'].shape[0]
    attr_dim = dataloader['attr_dim']
    state_dim = dataloader['state_dim']
    action_dim = dataloader['action_dim']
    relation_dim = dataloader['relation_dim']

    wandb.init(project='multires_st', config=args, name='Multitask-GWplusCKO')

    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    base_model_enc_list = []
    base_model_dec_list = []
    for seq_in_len, seq_out_len in zip(dataloader['x_lens'], dataloader['y_lens']):
        base_model_enc = gwnet_enc(
            device, args.num_nodes, args.dropout, supports=supports,
            gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, 
            aptinit=adjinit, in_dim=args.in_dim, out_dim=None, 
            residual_channels=args.nhid, dilation_channels=args.nhid, 
            skip_channels=args.nhid * 8,
            end_channels=args.nhid * 16
        )
        base_model_enc_list.append(base_model_enc)

        base_model_dec = gwnet_dec(
            in_channels=args.nhid * 16,
            end_channels=args.nhid * 16,
            seq_out_len=seq_out_len,
            out_dim=args.out_dim
        )
        base_model_dec_list.append(base_model_dec)
    model = MultitaskModel(
        input_dim=args.in_dim,
        output_dim=args.out_dim,
        res_seq_lens=dataloader['y_lens'],
        enc_base_model_list=base_model_enc_list,
        dec_base_model_list=base_model_dec_list,
        self_attn_embed_dim=args.nhid * 16,
        self_attn_num_heads=args.mt_attn_num_heads,
        attr_dim=attr_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        relation_dim=relation_dim,
        fit_type=args.fit_type,
        g_dim=args.g_dim,
        nf_particle=args.nf_particle,
        nf_relation=args.nf_relation,
        nf_effect=args.nf_effect,
        residual=False,
        pstep=args.pstep,
        I_factor=args.I_factor,
        enc_dec_type=args.enc_dec_type,
        nodenum=args.num_nodes,
        cko_dl_gate=args.cko_dl_gate,
        cko_dl_gate_hdim=args.cko_dl_gate_hdim,
        dropout=args.mt_attn_dropout,
        use_downsampling_pred=args.use_downsampling_pred,
        use_upsampling_pred=args.use_upsampling_pred,
        upsampler=args.upsampler,
        updown_fusion=args.updown_fusion,
        agg_res_hdim=args.agg_res_hdim,
        downsampler_agg=('mean' if args.resolution_type == 'agg' else 'first'),
        with_multitask_attn=args.with_multitask_attn
    )
    print(model)
    wandb.watch(model)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = trainer(model, scaler, 
                     args.learning_rate, args.weight_decay, args.cko_lrate, args.cko_wdecay, device, loss_type=args.loss_type,
                     cko_lambda_loss_metric=args.cko_lambda_loss_metric)

    save_dir = os.path.join(args.save, wandb.run.id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def run_test(epoch):
        outputs = []
        # realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = []

        if args.save_auxiliary_outputs:
            aux_outputs = []

        engine.model.eval()
        for iter, (x_data_list, y_data_list) in tqdm(enumerate(dataloader['test_loader']), total=len(dataloader['test_loader'])):
            x_data_list = move_data_list(x_data_list, device)
            y_data_list = move_data_list(y_data_list, device)
            with torch.no_grad():
                outdict = engine.model(x_data_list, y_data_list)
                preds = outdict['pred']
            outputs.append([p.detach().cpu() for p in preds])
            realy.append([y[1][:, 1:].cpu() for y in y_data_list])
            if args.save_auxiliary_outputs:
                aux_output_np = {}
                for k in outdict:
                    if outdict[k] is None:
                        aux_output_np[k] = outdict[k]
                    else:
                        templ = []
                        for o in outdict[k]:
                            if o is not None:
                                templ.append(o.detach().cpu())
                            else:
                                templ.append(None)
                        aux_output_np[k] = templ
                aux_outputs.append(aux_output_np)

        yhat_list = [torch.cat(output, dim=0) for output in list(zip(*outputs))]
        realy_list = [torch.cat(y, dim=0) for y in list(zip(*realy))]

        if args.save_auxiliary_outputs:
            aux_outputs_dict = {}
            for k in aux_outputs[0]:
                if aux_outputs[0][k] is None:
                    aux_outputs_dict[k] = None
                else:
                    aux_outputs_dict[k] = []
                    for ts in zip(*[aux_output[k] for aux_output in aux_outputs]):
                        if ts[0] is None:
                            aux_outputs_dict[k].append(None)
                        else:
                            aux_outputs_dict[k].append(torch.cat(ts, dim=0).cpu().numpy())
            aux_outputs_dict['agg_weights'] = [torch.nn.functional.softmax(engine.model.agg_res_weights[ri], dim=0).detach().cpu().numpy() for ri in range(len(engine.model.res_seq_lens))]

        for res_id, (pred, realy) in enumerate(zip(yhat_list, realy_list)):
            print(pred.shape, realy.shape)
            tmae, tmape, trmse = util.metric(pred, realy)
            wandb.log({
                'Res-{} Test MAE'.format(res_id): tmae,
                'Res-{} Test MAPE'.format(res_id): tmape,
                'Res-{} Test RMSE'.format(res_id): trmse,
                'Epoch': epoch
            })

            np.save(os.path.join(save_dir, 'res-{}-pred.npy'.format(res_id)), pred.numpy())
            np.save(os.path.join(save_dir, 'res-{}-true.npy'.format(res_id)), realy.numpy())

        if args.save_auxiliary_outputs:
            # np.savez_compressed(os.path.join(save_dir, 'auxiliary_outputs.npz'), **aux_outputs_dict)
            with open(os.path.join(save_dir, 'auxiliary_outputs.pkl'), 'wb') as f:
                pkl.dump(aux_outputs_dict, f)

    if args.eval_saved_model is not None:
        engine.model.load_state_dict(torch.load(args.eval_saved_model))
        run_test(epoch=0)
    else:
        print("start training...",flush=True)
        his_loss =[]
        val_time = []
        train_time = []

        best_val_loss = np.inf
        best_val_epoch = 0

        # dataloader['train_loader'].shuffle()
        for i in range(1,args.epochs+1):
            #if i % 10 == 0:
                #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
                #for g in engine.optimizer.param_groups:
                    #g['lr'] = lr
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            for iter, (x_data_list, y_data_list) in tqdm(enumerate(dataloader['train_loader']), total=len(dataloader['train_loader'])):
                x_data_list = move_data_list(x_data_list, device)
                y_data_list = move_data_list(y_data_list, device)
                metrics = engine.train(x_data_list, y_data_list)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
                if iter % args.print_every == 0 :
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(log.format(iter, train_loss[-1]['total_loss'], train_mape[-1], train_rmse[-1]),flush=True)
                    wandb_log_dict = {
                        # 'Step Train Loss': train_loss[-1],
                        'Step Train MAPE': train_mape[-1],
                        'Step Train RMSE': train_rmse[-1],
                    }
                    for k in train_loss[-1]:
                        wandb_log_dict.update({'Step Train {}'.format(k): train_loss[-1][k]})
                    wandb.log(wandb_log_dict)
            t2 = time.time()
            train_time.append(t2-t1)
            #validation
            valid_loss = []
            valid_mape = []
            valid_rmse = []


            s1 = time.time()
            for iter, (x_data_list, y_data_list) in tqdm(enumerate(dataloader['val_loader']), total=len(dataloader['val_loader'])):
                x_data_list = move_data_list(x_data_list, device)
                y_data_list = move_data_list(y_data_list, device)
                metrics = engine.eval(x_data_list, y_data_list)
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i,(s2-s1)))
            val_time.append(s2-s1)
            # mtrain_loss = np.mean(train_loss)
            mtrain_loss = {}
            for k in train_loss[0]:
                mtrain_loss[k] = np.mean([tl[k] for tl in train_loss])
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            # mvalid_loss = np.mean(valid_loss)
            mvalid_loss = {}
            for k in valid_loss[0]:
                mvalid_loss[k] = np.mean([tl[k] for tl in valid_loss])
            engine.cko_scheduler.step(mvalid_loss['cko_loss'])

            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss['loss'])

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss['total_loss'], mtrain_mape, mtrain_rmse, mvalid_loss['loss'], mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
            wandb_log_dict = {
                'Epoch': i,
                # 'Train Loss': mtrain_loss,
                'Train MAPE': mtrain_mape, 'Train RMSE': mtrain_rmse,
                # 'Valid Loss': mvalid_loss,
                'Valid MAPE': mvalid_mape, 'Valid RMSE': mvalid_rmse
            }
            for k in mtrain_loss:
                wandb_log_dict.update({
                    'Train {}'.format(k): mtrain_loss[k],
                    'Valid {}'.format(k): mvalid_loss[k]
                })
            wandb.log(wandb_log_dict)
            
            if np.abs(mvalid_loss['loss']) <= 1e-6: # confronted with nan, exit
                early_stop_flag = True
            else:
                if mvalid_loss['loss'] < best_val_loss - args.patience_th:
                    best_val_loss = mvalid_loss['loss']
                    best_val_epoch = i
                    torch.save(engine.model.state_dict(), os.path.join(save_dir, "best.pth"))
                    no_dec_epoch_num = 0
                else:
                    no_dec_epoch_num = i - best_val_epoch
                    print('Val loss has not decreased for {} epochs (threshold {:.5f})'.format(no_dec_epoch_num, args.patience_th))
                
                early_stop_flag =  (args.patience > 0) and (no_dec_epoch_num > args.patience)

            #testing
            if (i % args.test_int == 0) or (i == args.epochs) or early_stop_flag:
                print('Testing for epoch {}...'.format(i))
                ckpt = deepcopy(engine.model.state_dict())
                bestid = np.argmin(his_loss)
                engine.model.load_state_dict(torch.load(os.path.join(save_dir, "best.pth")))
                run_test(epoch=i)
                engine.model.load_state_dict(ckpt)

            if early_stop_flag:
                break

        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        print("Training finished")
        print("The valid loss on best model is", str(round(his_loss[bestid],4)))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
