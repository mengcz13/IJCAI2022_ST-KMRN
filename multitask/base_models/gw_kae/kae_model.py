from torch import nn
import torch

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega


class encoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(encoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.N, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, b)

        self.norm1 = nn.LayerNorm(16 * ALPHA)
        self.norm2 = nn.LayerNorm(16 * ALPHA)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        # x = self.tanh(self.fc1(x))
        # x = self.tanh(self.fc2(x))        
        x = self.tanh(self.norm1(self.fc1(x)))
        x = self.tanh(self.norm2(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class decoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(decoderNet, self).__init__()

        self.m = m
        self.n = n
        self.b = b

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(b, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, m*n)

        self.norm1 = nn.LayerNorm(16 * ALPHA)
        self.norm2 = nn.LayerNorm(16 * ALPHA)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        # x = self.tanh(self.fc1(x)) 
        # x = self.tanh(self.fc2(x)) 
        # x = self.tanh(self.fc3(x))
        x = self.tanh(self.norm1(self.fc1(x)))
        x = self.tanh(self.norm2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, 1, self.m, self.n)
        return x



class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

        
    def forward(self, x):
        x = self.dynamics(x)
        return x


class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())     

    def forward(self, x):
        x = self.dynamics(x)
        return x




class koopmanAE(nn.Module):
    def __init__(self, m, n, b, steps, steps_back, alpha = 1, init_scale=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        
        self.encoder = encoderNet(m, n, b, ALPHA = alpha)
        self.dynamics = dynamics(b, init_scale)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.decoder = decoderNet(m, n, b, ALPHA = alpha)


    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous())) 
            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back

class koopmanAEwithattr(nn.Module):
    def __init__(self, fdim, adim, nodenum, b, steps, steps_back, alpha = 1, init_scale=1):
        super(koopmanAEwithattr, self).__init__()
        self.steps = steps
        self.steps_back = steps_back

        self.fdim = fdim
        self.adim = adim
        self.nodenum = nodenum
        
        self.encoder = encoderNet(nodenum, (fdim + adim), b, ALPHA = alpha)
        self.dynamics = dynamics(b, init_scale)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.decoder = decoderNet(nodenum, fdim, b + adim, ALPHA = alpha)


    def forward(self, x, attrs, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(torch.cat([x, attrs[:, :1]], dim=-1).contiguous())
        q = z.contiguous()

        
        if mode == 'forward':
            for si in range(min(self.steps, attrs.size(1))):
                q = self.dynamics(q)
                out.append(self.decoder(torch.cat([q, attrs[:, si+1:si+2, 0]], dim=-1)))

            out.append(self.decoder(torch.cat([z, attrs[:, :1, 0]], dim=-1).contiguous())) 
            return out, out_back    

        if mode == 'backward':
            for si in range(min(self.steps_back, attrs.size(1))):
                q = self.backdynamics(q)
                out_back.append(self.decoder(torch.cat([q, attrs[:, -si-2:-si-1, 0]], dim=-1)))
                
            out_back.append(self.decoder(torch.cat([z, attrs[:, -1:, 0]], dim=-1).contiguous()))
            return out_back, out_back


def kae_sysid_and_pred(model, seq_data, fit_data, backward):
    retdict = {}
    attrs, states, actions, rel_attrs = seq_data
    out_fw, _ = model(states[:, :1], attrs, mode='forward')
    retdict['out_fw'] = torch.cat(out_fw, dim=1)
    if backward == 1:
        _, out_bw = model(states[:, -1:], attrs, mode='backward')
        out_bw = out_bw[::-1]
        out_bw = out_bw[1:] + out_bw[:1]
        retdict['out_bw'] = torch.cat(out_bw, dim=1)
    return retdict


def kae_consist_lossfunc(A, B):
    K = A.shape[-1]

    for k in range(1,K+1):
        As1 = A[:,:k]
        Bs1 = B[:k,:]
        As2 = A[:k,:]
        Bs2 = B[:,:k]

        Ik = torch.eye(k).float().to(A.device)

        if k == 1:
            loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
        else:
            loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                torch.sum((torch.mm(As2, Bs2)-  Ik)**2) ) / (2.0*k)

    return loss_consist