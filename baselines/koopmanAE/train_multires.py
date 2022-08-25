from copy import deepcopy
import torch
from torch import nn
import numpy as np

from tools import *

from tqdm import tqdm

import os
import wandb


def eval(model, test_dataloader, criterion, steps, device, with_res=False):
    model.eval()
    if with_res:
        all_out, all_states = [], []
    loss_fwd_epoch = []
    with torch.no_grad():
        for batch_idx, (seq_data, _) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            attrs, states, _, _ = seq_data
            out, out_back = model(states[:, :1].to(device), attrs.to(device), mode='forward')
            for k in range(steps):
                if k == 0:
                    loss_fwd = criterion(out[k], states[:, k+1:k+2].to(device))
                else:
                    loss_fwd += criterion(out[k], states[:, k+1:k+2].to(device))
            loss_fwd_epoch.append(loss_fwd.item())
            if with_res:
                out = torch.cat(out, dim=1)[:, :-1]
                all_out.append(out.detach().cpu())
                all_states.append(states[:, 1:].detach().cpu())
    if with_res:
        all_out = torch.cat(all_out, dim=0).numpy()
        all_states = torch.cat(all_states, dim=0).numpy()
        return np.mean(loss_fwd_epoch), all_out, all_states
    else:
        return np.mean(loss_fwd_epoch)


def train(model, train_loader, val_loader, test_loader, lr, weight_decay, 
          lamb, num_epochs, learning_rate_change, epoch_update, 
          nu=0.0, eta=0.0, backward=0, steps=1, steps_back=1, gradclip=1, model_folder=None):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = get_device()
             
            
    def lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
                    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
                    if epoch in decayEpoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_decay_rate
                        return optimizer
                    else:
                        return optimizer
                        
                     
        

    criterion = nn.MSELoss().to(device)


    epoch_hist = []
    loss_hist = []
    epoch_loss = []

    best_val_loss_fwd = np.inf
                            
    for epoch in range(num_epochs):
        #print(epoch)
        # train
        model.train()
        for batch_idx, (seq_data, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            attrs, states, _, _ = seq_data
            model.train()
            out, out_back = model(states[:, :1].to(device), attrs.to(device), mode='forward')


            for k in range(steps):
                if k == 0:
                    loss_fwd = criterion(out[k], states[:, k+1:k+2].to(device))
                else:
                    loss_fwd += criterion(out[k], states[:, k+1:k+2].to(device))

            
            loss_identity = criterion(out[-1], states[:, 0:1].to(device)) * steps


            loss_bwd = 0.0
            loss_consist = 0.0

            loss_bwd = 0.0
            loss_consist = 0.0

            if backward == 1:
                out, out_back = model(states[:, -1:].to(device), attrs.to(device), mode='backward')
   

                for k in range(steps_back):
                    
                    if k == 0:
                        loss_bwd = criterion(out_back[k], states[:, -k-2:-k-1].to(device))
                    else:
                        loss_bwd += criterion(out_back[k], states[:, -k-2:-k-1].to(device))
                        
                               
                A = model.dynamics.dynamics.weight
                B = model.backdynamics.dynamics.weight

                K = A.shape[-1]

                for k in range(1,K+1):
                    As1 = A[:,:k]
                    Bs1 = B[:k,:]
                    As2 = A[:k,:]
                    Bs2 = B[:,:k]

                    Ik = torch.eye(k).float().to(device)

                    if k == 1:
                        loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                         torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
                    else:
                        loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                         torch.sum((torch.mm(As2, Bs2)-  Ik)**2) ) / (2.0*k)


                
                
                
#                Ik = torch.eye(K).float().to(device)
#                loss_consist = (torch.sum( (torch.mm(A, B)-Ik )**2)**1 + \
#                                         torch.sum( (torch.mm(B, A)-Ik)**2)**1 )
#   
                                        
                
    
            loss = loss_fwd + lamb * loss_identity +  nu * loss_bwd + eta * loss_consist

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip
            optimizer.step()           

        # schedule learning rate decay    
        lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)
        loss_hist.append(loss)                
        epoch_loss.append(epoch)

        # validation
        val_loss_fwd = eval(model, val_loader, criterion, steps, device, with_res=False)
        if val_loss_fwd < best_val_loss_fwd:
            best_val_loss_fwd = val_loss_fwd
            torch.save(model.state_dict(), os.path.join(model_folder, 'best.pkl'))

        # test
        last_weights = deepcopy(model.state_dict())
        best_weights = torch.load(os.path.join(model_folder, 'best.pkl'), map_location='cpu')
        model.load_state_dict(best_weights)
        test_loss_fwd, test_pred, test_true = eval(model, test_loader, criterion, steps, device, with_res=True)
        model.load_state_dict(last_weights)
        np.save(os.path.join(model_folder, 'pred.npy'), test_pred)
        np.save(os.path.join(model_folder, 'true.npy'), test_true)
        
        wandb.log({
            'train loss identity': loss_identity.item(),
            'train loss backward': loss_bwd.item(),
            'train loss consistent': loss_consist.item(),
            'train loss forward': loss_fwd.item(),
            'train loss': loss.item(),
            'val loss forward': val_loss_fwd,
            'test loss forward': test_loss_fwd,
            'Epoch': epoch
        })
        print('test loss fwd', test_loss_fwd)

        # torch.save(model.state_dict(), os.path.join(model_folder, 'epoch-{}.pkl'.format(epoch)))
        

        # if (epoch) % 20 == 0:
        #         print('********** Epoche %s **********' %(epoch+1))
                
        #         print("loss identity: ", loss_identity.item())
        #         if backward == 1:
        #             print("loss backward: ", loss_bwd.item())
        #             print("loss consistent: ", loss_consist.item())
        #         print("loss forward: ", loss_fwd.item())
        #         print("loss sum: ", loss.item())

        #         epoch_hist.append(epoch+1) 

        #         if hasattr(model.dynamics, 'dynamics'):
        #             w, _ = np.linalg.eig(model.dynamics.dynamics.weight.data.cpu().numpy())
        #             print(np.abs(w))


    if backward == 1:
        loss_consist = loss_consist.item()
                
                
    return model, optimizer, [epoch_hist, loss_fwd.item(), loss_consist]
