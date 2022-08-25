import os
import torch
import numpy as np
import argparse
import time
import util
import multires_data
import matplotlib.pyplot as plt
from engine import trainer

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
# TCN blocks and layers, kernel size
parser.add_argument('--blocks', type=int, default=4)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--kernel_size', type=int, default=2)

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
parser.add_argument('--save',type=str,default='./garage',help='save path')
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

parser.add_argument('--keep_ratio', type=float, default=0.8)

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

    wandb.init(project='multires_st', config=args, name='Graph-WaveNet')

    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None



    engine = trainer(scaler, args.in_dim, args.out_dim, args.seq_out_len, args.num_nodes, args.nhid, args.blocks, args.layers, args.kernel_size, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, args)

    save_dir = os.path.join(args.save, wandb.run.id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def run_test(epoch):
        outputs = []
        # realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = []

        for iter, (x, y) in tqdm(enumerate(dataloader['test_loader']), total=len(dataloader['test_loader'])):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)
            testy = torch.Tensor(y)
            with torch.no_grad():
                preds = engine.model(testx).transpose(1,3)
            outputs.append(preds.detach().cpu())
            realy.append(testy.cpu())

        yhat = torch.cat(outputs,dim=0)
        # yhat = yhat[:realy.size(0),...]

        realy = torch.cat(realy, dim=0)

        pred = yhat
        print(pred.shape, realy.shape)
        tmae, tmape, trmse = util.metric(pred, realy)
        wandb.log({
            'Test MAE': tmae, 'Test MAPE': tmape, 'Test RMSE': trmse, 'Epoch': epoch
        })

        np.save(os.path.join(save_dir, 'pred.npy'), yhat.numpy())
        np.save(os.path.join(save_dir, 'true.npy'), realy.numpy())

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
            for iter, (x, y) in tqdm(enumerate(dataloader['train_loader']), total=len(dataloader['train_loader'])):
                trainx = torch.Tensor(x).to(device)
                trainx= trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)
                metrics = engine.train(trainx, trainy)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
                if iter % args.print_every == 0 :
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
                    wandb.log({
                        'Step Train Loss': train_loss[-1],
                        'Step Train MAPE': train_mape[-1],
                        'Step Train RMSE': train_rmse[-1],
                    })
            t2 = time.time()
            train_time.append(t2-t1)
            #validation
            valid_loss = []
            valid_mape = []
            valid_rmse = []


            s1 = time.time()
            for iter, (x, y) in tqdm(enumerate(dataloader['val_loader']), total=len(dataloader['val_loader'])):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                metrics = engine.eval(testx, testy)
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i,(s2-s1)))
            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
            wandb.log({
                'Epoch': i,
                'Train Loss': mtrain_loss, 'Train MAPE': mtrain_mape, 'Train RMSE': mtrain_rmse,
                'Valid Loss': mvalid_loss, 'Valid MAPE': mvalid_mape, 'Valid RMSE': mvalid_rmse
            })

            if mvalid_loss < best_val_loss - args.patience_th:
                best_val_loss = mvalid_loss
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
                # engine.model.load_state_dict(torch.load(os.path.join(save_dir, "_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth")))
                engine.model.load_state_dict(torch.load(os.path.join(save_dir, "best.pth")))
                run_test(epoch=i)
                engine.model.load_state_dict(ckpt)

            if early_stop_flag:
                break

        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        print("Training finished")
        print("The valid loss on best model is", str(round(his_loss[bestid],4)))




    # amae = []
    # amape = []
    # armse = []
    # for i in range(args.seq_out_len):
    #     pred = yhat[:, i, :, :]
    #     real = realy[:, i, :, :]
    #     metrics = util.metric(pred,real)
    #     log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    #     print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
    #     amae.append(metrics[0])
    #     amape.append(metrics[1])
    #     armse.append(metrics[2])

    # log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    # print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    # torch.save(engine.model.state_dict(), os.path.join(save_dir, "_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth"))



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
