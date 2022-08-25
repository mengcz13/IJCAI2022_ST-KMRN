import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset, dataloader

import torch.nn.init as init

# from read_dataset import data_from_name
from multires_loader import MultiresDataset
from model import *
from tools import *
from train_multires import *

import os
import wandb

#==============================================================================
# Training settings
#==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--model', type=str, default='koopmanAE', metavar='N', help='model')
#
parser.add_argument('--alpha', type=int, default='1',  help='model width')
#
parser.add_argument('--dataset', type=str, default='flow_noisy', metavar='N', help='dataset')
#
parser.add_argument('--theta', type=float, default=2.4,  metavar='N', help='angular displacement')
#
parser.add_argument('--noise', type=float, default=0.0,  metavar='N', help='noise level')
#
parser.add_argument('--lr', type=float, default=1e-2, metavar='N', help='learning rate (default: 0.01)')
#
parser.add_argument('--wd', type=float, default=0.0, metavar='N', help='weight_decay (default: 1e-5)')
#
parser.add_argument('--epochs', type=int, default=600, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--batch', type=int, default=64, metavar='N', help='batch size (default: 10000)')
#
parser.add_argument('--batch_test', type=int, default=200, metavar='N', help='batch size  for test set (default: 10000)')
#
parser.add_argument('--plotting', type=bool, default=True, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--folder', type=str, default='test',  help='specify directory to print results to')
#
parser.add_argument('--lamb', type=float, default='1',  help='balance between reconstruction and prediction loss')
#
parser.add_argument('--nu', type=float, default='1e-1',  help='tune backward loss')
#
parser.add_argument('--eta', type=float, default='1e-2',  help='tune consistent loss')
#
parser.add_argument('--steps', type=int, default='8',  help='steps for learning forward dynamics')
#
parser.add_argument('--steps_back', type=int, default='8',  help='steps for learning backwards dynamics')
#
parser.add_argument('--bottleneck', type=int, default='6',  help='size of bottleneck layer')
#
parser.add_argument('--lr_update', type=int, nargs='+', default=[30, 200, 400, 500], help='decrease learning rate at these epochs')
#
parser.add_argument('--lr_decay', type=float, default='0.2',  help='PCL penalty lambda hyperparameter')
#
parser.add_argument('--backward', type=int, default=0, help='train with backward dynamics')
#
parser.add_argument('--init_scale', type=float, default=0.99, help='init scaling')
#
parser.add_argument('--gradclip', type=float, default=0.05, help='gradient clipping')
#
parser.add_argument('--pred_steps', type=int, default='1000',  help='prediction steps')
#
parser.add_argument('--seed', type=int, default='1',  help='seed value')
#

'''
for multires data
'''
parser.add_argument('--data', type=str, default='')
parser.add_argument('--seq_in_len', type=int, default=1440)
parser.add_argument('--seq_out_len', type=int, default=480)
parser.add_argument('--seq_diff', type=int, default=48)
parser.add_argument('--test_seq_diff', type=int, default=48, help='only used for test set to reduce num of seqs for saving')
parser.add_argument('--target_res', type=str, default='30min', help='30min/6-hour/day/all')
parser.add_argument('--keep_ratio', type=float, default=0.8)
parser.add_argument('--resolution_type', type=str, default='agg')
parser.add_argument('--single_res_input_output', action='store_true', help='true if used for verification on existing exps')


args = parser.parse_args()

wandb.init(project='multires_st', config=args, name='koopmanAE')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
set_seed(args.seed)
device = get_device()


## load multires data
train_data = MultiresDataset(
    args.data, 'train', args
)
val_data = MultiresDataset(
    args.data, 'val', args
)
test_data = MultiresDataset(
    args.data, 'test', args
)
train_dataloader = DataLoader(train_data, batch_size=args.batch, shuffle=True, collate_fn=train_data.collate_fn)
val_dataloader = DataLoader(val_data, batch_size=args.batch, shuffle=False, collate_fn=val_data.collate_fn)
test_dataloader = DataLoader(test_data, batch_size=args.batch, shuffle=False, collate_fn=test_data.collate_fn)
sample_seq_data, _ = next(iter(train_dataloader))
sample_seq_data, _ = next(iter(train_dataloader))
sample_attr, sample_states, sample_actions, sample_rel_attrs = sample_seq_data
fdim = sample_states.size(-1)
adim = sample_attr.size(-1)
nodenum = sample_states.size(-2)



#==============================================================================
# Model
#==============================================================================
model = koopmanAEwithattr(fdim, adim, nodenum, args.bottleneck, args.steps, args.steps_back, args.alpha, args.init_scale)
print('koopmanAE')
#model = torch.nn.DataParallel(model)
model = model.to(device)

wandb.watch(model)


#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('Total params: %.2fk' % (sum(p.numel() for p in model.parameters())/1000.0))
print('************')
print(model)

# model_folder = args.folder
model_folder = 'save/{}'.format(wandb.run.id)
if not os.path.exists(model_folder):
    os.makedirs(model_folder, exist_ok=True)

#==============================================================================
# Start training
#==============================================================================
model, optimizer, epoch_hist = train(model, train_dataloader, val_dataloader, test_dataloader,
                    lr=args.lr, weight_decay=args.wd, lamb=args.lamb, num_epochs = args.epochs,
                    learning_rate_change=args.lr_decay, epoch_update=args.lr_update,
                    nu = args.nu, eta = args.eta, backward=args.backward, steps=args.steps, steps_back=args.steps_back,
                    gradclip=args.gradclip, model_folder=model_folder)

# torch.save(model.state_dict(), args.folder + '/model'+'.pkl')
