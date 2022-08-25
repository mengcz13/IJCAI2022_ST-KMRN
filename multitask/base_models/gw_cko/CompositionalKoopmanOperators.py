from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from base_models.gw_cko.cko_utils import get_flat

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: [n_relations, input_size]
        Returns:
            [n_relations, output_size]
        """
        return self.model(x)


class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticleEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        return self.model(x)


class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        """
        Args:
            x: [n_relations/n_particles, input_size]
        Returns:
            [n_relations/n_particles, output_size]
        """
        if self.residual:
            x = self.relu(self.layer_norm(self.linear(x) + res))
        else:
            x = self.relu(self.layer_norm(self.linear(x)))

        return x


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        x = self.relu(self.linear_0(x))

        return self.linear_1(x)


class l2_norm_layer(nn.Module):
    def __init__(self):
        super(l2_norm_layer, self).__init__()

    def forward(self, x):
        """
        :param x:  B x D
        :return:
        """
        norm_x = torch.sqrt((x ** 2).sum(1) + 1e-10)
        return x / norm_x[:, None]


class PropagationNetwork(nn.Module):
    def __init__(self, attr_dim, state_dim, action_dim, relation_dim, 
                 nf_particle, nf_relation, nf_effect,
                 input_particle_dim=None, input_relation_dim=None, output_dim=None, action=True, tanh=False,
                 residual=False):

        super(PropagationNetwork, self).__init__()

        self.action = action

        if input_particle_dim is None:
            input_particle_dim = attr_dim + state_dim
            input_particle_dim += action_dim if action else 0

        if input_relation_dim is None:
            input_relation_dim = relation_dim + state_dim

        if output_dim is None:
            output_dim = state_dim

        nf_particle = nf_particle
        nf_relation = nf_relation
        nf_effect = nf_effect

        self.nf_effect = nf_effect

        self.residual = residual

        # (1) state
        self.obj_encoder = ParticleEncoder(input_particle_dim, nf_particle, nf_effect)

        # (1) state receiver (2) state_diff
        self.relation_encoder = RelationEncoder(input_relation_dim, nf_relation, nf_relation)

        # (1) relation encode (2) sender effect (3) receiver effect
        self.relation_propagator = Propagator(nf_relation + 2 * nf_effect, nf_effect)

        # (1) particle encode (2) particle effect
        self.particle_propagator = Propagator(2 * nf_effect, nf_effect, self.residual)

        # rigid predictor
        # (1) particle encode (2) set particle effect

        self.particle_predictor = ParticlePredictor(nf_effect, nf_effect, output_dim)

        if tanh:
            self.particle_predictor = nn.Sequential(
                self.particle_predictor, nn.Tanh()
            )

    def forward(self, attrs, states, actions, rel_attrs, pstep):
        """
        :param attrs: B x N x attr_dim
        :param states: B x N x state_dim
        :param actions: B x N x action_dim
        :param rel_attrs: B x N x N x relation_dim
        :param pstep: 1 or 2
        :return:
        """
        B, N = attrs.size(0), attrs.size(1)
        '''encode node'''
        obj_input_list = [attrs, states]
        if self.action:
            obj_input_list += [actions]

        tmp = torch.cat(obj_input_list, 2)
        obj_encode = self.obj_encoder(tmp.reshape(tmp.size(0) * tmp.size(1), tmp.size(2))).reshape(B, N, -1)

        '''encode edge'''
        rel_states = states[:, :, None, :] - states[:, None, :, :]
        receiver_attr = attrs[:, :, None, :].expand(-1, -1, N, -1)
        sender_attr = attrs[:, None, :, :].expand(-1, N, -1, -1)
        tmp = torch.cat([rel_attrs, rel_states, receiver_attr, sender_attr], 3)
        rel_encode = self.relation_encoder(tmp.reshape(B * N * N, -1)).reshape(B, N, N, -1)

        for i in range(pstep):
            '''calculate relation effect'''

            receiver_code = obj_encode[:, :, None, :].expand(-1, -1, N, -1)
            sender_code = obj_encode[:, None, :, :].expand(-1, N, -1, -1)
            tmp = torch.cat([rel_encode, receiver_code, sender_code], 3)
            rel_effect = self.relation_propagator(tmp.reshape(B * N * N, -1)).reshape(B, N, N, -1)

            '''aggregating relation effect'''

            rel_agg_effect = rel_effect.sum(2)

            '''calc particle effect'''
            tmp = torch.cat([obj_encode, rel_agg_effect], 2)
            obj_encode = self.particle_propagator(tmp.reshape(B * N, -1)).reshape(B, N, -1)

        obj_prediction = self.particle_predictor(obj_encode.reshape(B * N, -1)).reshape(B, N, -1)
        return obj_prediction


# MLP encoder/decoder
class MLP(nn.Module):
    def __init__(self, attr_dim, state_dim, action_dim, relation_dim, 
                 hidden_dim, layer_num, nodenum, decoding=False,
                 input_particle_dim=None, input_relation_dim=None, output_dim=None, action=True):
        super().__init__()

        self.action = action
        self.decoding = decoding

        if input_particle_dim is None:
            input_particle_dim = attr_dim + state_dim
            input_particle_dim += action_dim if action else 0

        if input_relation_dim is None:
            input_relation_dim = relation_dim + state_dim

        if output_dim is None:
            output_dim = state_dim

        layers = []
        for li in range(layer_num):
            if li == 0:
                l_input_dim = input_particle_dim
            else:
                l_input_dim = hidden_dim
            if li == layer_num - 1:
                l_output_dim = output_dim
            else:
                l_output_dim = hidden_dim
            layers.append(nn.Linear(l_input_dim, l_output_dim))
            if (li < layer_num - 1) or (not decoding):
                layers.append(nn.LayerNorm(l_output_dim))
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, attrs, states, actions, rel_attrs, pstep):
        """
        :param attrs: B x N x attr_dim
        :param states: B x N x state_dim
        :param actions: B x N x action_dim
        :param rel_attrs: B x N x N x relation_dim
        :param pstep: 1 or 2
        :return:
        """
        B, N = attrs.size(0), attrs.size(1)
        '''encode node'''
        obj_input_list = [attrs, states]
        if self.action:
            obj_input_list += [actions]

        if len(states.shape) > 2:
            tmp = torch.cat(obj_input_list, 2)
            tmp = tmp.flatten(-2, -1)
        else:
            flat_obj_input_list = [x.flatten(1, -1) for x in obj_input_list]
            tmp = torch.cat(flat_obj_input_list, 1)
        obj_encode = self.layers(tmp)
        if self.decoding:
            obj_encode = obj_encode.reshape(B, N, -1)

        return obj_encode


# ======================================================================================================================
class CompositionalKoopmanOperators(nn.Module, ABC):
    def __init__(self, attr_dim, state_dim, action_dim, relation_dim,
                fit_type, g_dim, nf_particle, nf_relation, nf_effect, nodenum,
                residual=False, enc_dec_type='pn'):
        super(CompositionalKoopmanOperators, self).__init__()

        self.stat = None

        self.residual = residual

        ''' state '''
        # we should not include action in state encoder
        input_particle_dim = attr_dim + state_dim
        input_relation_dim = state_dim + relation_dim + attr_dim * 2

        # print('state_encoder', 'node', input_particle_dim, 'edge', input_relation_dim)
        if enc_dec_type == 'pn':
            self.state_encoder = PropagationNetwork(
                attr_dim, state_dim, action_dim, relation_dim, nf_particle, nf_relation, nf_effect,
                input_particle_dim=input_particle_dim, input_relation_dim=input_relation_dim,
                output_dim=g_dim, action=False, tanh=True,  # use tanh to enforce the shape of the code space
                residual=residual)
        elif enc_dec_type == 'mlp':
            self.state_encoder = MLP(
                attr_dim, state_dim, action_dim, relation_dim,
                hidden_dim=128, layer_num=2, nodenum=nodenum, decoding=False,
                input_particle_dim=input_particle_dim*nodenum, input_relation_dim=input_relation_dim,
                output_dim=g_dim, action=False
            )
        else:
            raise NotImplementedError()

        # the state for decoding phase is replaced with code of g_dim
        input_particle_dim = attr_dim + g_dim
        input_relation_dim = g_dim + relation_dim + attr_dim * 2

        # print('state_decoder', 'node', input_particle_dim, 'edge', input_relation_dim)
        if enc_dec_type == 'pn':
            self.state_decoder = PropagationNetwork(
                attr_dim, state_dim, action_dim, relation_dim, nf_particle, nf_relation, nf_effect,
                input_particle_dim=input_particle_dim, input_relation_dim=input_relation_dim,
                output_dim=state_dim, action=False, tanh=False,
                residual=residual)
        elif enc_dec_type == 'mlp':
            self.state_decoder = MLP(
                attr_dim, state_dim, action_dim, relation_dim,
                hidden_dim=128, layer_num=2, nodenum=nodenum, decoding=True,
                input_particle_dim=g_dim+attr_dim*nodenum, input_relation_dim=input_relation_dim,
                output_dim=state_dim*nodenum, action=False 
            )

        ''' dynamical system coefficient: A and B '''
        self.A = None
        self.B = None
        if fit_type == 'structured':
            self.system_identify = self.fit
            self.simulate = self.rollout
            self.step = self.linear_forward
        if fit_type == 'unstructured':
            self.system_identify = self.fit_unstructured
            self.simulate = self.rollout_unstructured
            self.step = self.linear_forward_unstructured
        elif fit_type == 'diagonal':
            self.system_identify = self.fit_diagonal
            self.simulate = self.rollout_diagonal
            self.step = self.linear_forward_diagonal

    def to_s(self, attrs, gcodes, rel_attrs, pstep):
        """ state decoder """
        return self.state_decoder(attrs=attrs, states=gcodes, actions=None, rel_attrs=rel_attrs, pstep=pstep)

    def to_g(self, attrs, states, rel_attrs, pstep):
        """ state encoder """
        return self.state_encoder(attrs=attrs, states=states, actions=None, rel_attrs=rel_attrs, pstep=pstep)

    @staticmethod
    def get_aug(G, rel_attrs):
        """
        :param G: B x T x N x D
        :param rel_attrs:  B x N x N x R
        :return:
        """
        B, T, N, D = G.size()
        R = rel_attrs.size(-1)

        sumG_list = []
        for i in range(R):
            ''' B x T x N x N '''
            adj = rel_attrs[:, :, :, i][:, None, :, :].expand(-1, T, -1, -1)
            sumG = torch.bmm(
                adj.reshape(B * T, N, N),
                G.reshape(B * T, N, D)
            ).reshape(B, T, N, D)
            sumG_list.append(sumG)

        augG = torch.cat(sumG_list, 3)

        return augG

    # structured A

    def fit(self, G, H, U, rel_attrs, I_factor):
        """
        :param G: B x T x N x D
        :param H: B x T x N x D
        :param U: B x T x N x a_dim
        :param rel_attrs: B x N x N x R (relation_dim) rel_attrs[i,j] ==> receiver i, sender j
        :param I_factor: scalor
        :return:
        A: B x R D x D
        B: B x R a_dim x D
        s.t.
        H = augG @ A + augU @ B
        """

        ''' B x R: sqrt(# of appearance of block matrices of the same type)'''
        rel_weights = torch.sqrt(rel_attrs.sum(1).sum(1))
        rel_weights = torch.clamp(rel_weights, min=1e-8)

        bs, T, N, D = G.size()
        R = rel_attrs.size(-1)
        a_dim = U.size(3)

        ''' B x T x N x R D '''
        augG = self.get_aug(G, rel_attrs)
        ''' B x T x N x R a_dim'''
        augU = self.get_aug(U, rel_attrs)

        augG_reweight = augG.reshape(bs, T, N, R, D) / rel_weights[:, None, None, :, None]
        augU_reweight = augU.reshape(bs, T, N, R, a_dim) / rel_weights[:, None, None, :, None]

        ''' B x TN x R(D + a_dim)'''
        GU_reweight = torch.cat([augG_reweight.reshape(bs, T * N, R * D),
                                 augU_reweight.reshape(bs, T * N, R * a_dim)], 2)

        '''B x (R * D + R * a_dim) x D'''
        AB_reweight = torch.bmm(
            self.batch_pinv(GU_reweight, I_factor),
            H.reshape(bs, T * N, D)
        )
        self.A = AB_reweight[:, :R * D].reshape(bs, R, D, D) / rel_weights[:, :, None, None]
        self.B = AB_reweight[:, R * D:].reshape(bs, R, a_dim, D) / rel_weights[:, :, None, None]

        self.A = self.A.reshape(bs, R * D, D)
        self.B = self.B.reshape(bs, R * a_dim, D)

        fit_err = H.reshape(bs, T * N, D) - torch.bmm(GU_reweight, AB_reweight)
        fit_err = torch.sqrt((fit_err ** 2).mean())

        return self.A, self.B, fit_err

    def linear_forward(self, g, u, rel_attrs):
        """
        :param g: B x N x D
        :param u: B x N x a_dim
        :param rel_attrs: B x N x N x R
        :return:
        """
        ''' B x N x R D '''
        aug_g = self.get_aug(G=g[:, None, :, :], rel_attrs=rel_attrs)[:, 0]
        ''' B x N x R a_dim'''
        aug_u = self.get_aug(G=u[:, None, :, :], rel_attrs=rel_attrs)[:, 0]

        new_g = torch.bmm(aug_g, self.A) + torch.bmm(aug_u, self.B)
        return new_g

    def rollout(self, g, u_seq, T, rel_attrs):
        """
        :param g: B x N x D
        :param u_seq: B x T x N x a_dim
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        g_list = []
        for t in range(T):
            g = self.linear_forward(g, u_seq[:, t], rel_attrs)
            g_list.append(g[:, None, :, :])
        return torch.cat(g_list, 1)

    # unstructured large A

    def fit_unstructured(self, G, H, U, I_factor, rel_attrs=None):
        """
        :param G: B x T x N x D
        :param H: B x T x N x D
        :param U: B x T x N x a_dim
        :param I_factor: scalor
        :return: A, B
        s.t.
        H = catG @ A + catU @ B
        """
        if len(G.size()) > 3:
            bs, T, N, D = G.size()
            G = G.reshape(bs, T, -1)
            H = H.reshape(bs, T, -1)
            U = U.reshape(bs, T, -1)
        else:
            U = U[:, :, 0, :]
        G_U = torch.cat([G, U], 2)
        A_B = torch.bmm(
            self.batch_pinv(G_U, I_factor),
            H
        )

        # self.A = A_B[:, :N * D]
        # self.B = A_B[:, N * D:]
        self.A = A_B[:, :G.shape[-1]]
        self.B = A_B[:, G.shape[-1]:]

        fit_err = H - torch.bmm(G_U, A_B)
        fit_err = torch.sqrt((fit_err ** 2).mean())

        return self.A, self.B, fit_err

    def linear_forward_unstructured(self, g, u, rel_attrs=None):
        if len(g.size()) > 2:
            B, N, D = g.size()
            a_dim = u.size(-1)
            g = g.reshape(B, 1, N * D)
            u = u.reshape(B, 1, N * a_dim)
            new_g = torch.bmm(g, self.A) + torch.bmm(u, self.B)
            return new_g.reshape(B, N, D)
        else:
            g = g.unsqueeze(1)
            u = u[:, 0, :].unsqueeze(1)
            new_g = torch.bmm(g, self.A) + torch.bmm(u, self.B)
            new_g = new_g.squeeze(1)
            return new_g

    def rollout_unstructured(self, g, u_seq, T, rel_attrs=None):
        g_list = []
        for t in range(T):
            g = self.linear_forward_unstructured(g, u_seq[:, t])
            if len(g.size()) > 2:
                g_list.append(g[:, None, :, :])
            else:
                g_list.append(g[:, None, :])
        return torch.cat(g_list, 1)

    # shared small A

    def fit_diagonal(self, G, H, U, I_factor, rel_attrs=None):
        bs, T, N, D = G.size()
        a_dim = U.size(3)

        G_U = torch.cat([G, U], 3)

        '''B x (D + a_dim) x D'''
        A_B = torch.bmm(
            self.batch_pinv(G_U.reshape(bs, T * N, D + a_dim), I_factor),
            H.reshape(bs, T * N, D)
        )
        self.A = A_B[:, :D]
        self.B = A_B[:, D:]

        fit_err = H.reshape(bs, T * N, D) - torch.bmm(G_U.reshape(bs, T * N, D + a_dim), A_B)
        fit_err = torch.sqrt((fit_err ** 2).mean())

        return self.A, self.B, fit_err

    def linear_forward_diagonal(self, g, u, rel_attrs=None):
        new_g = torch.bmm(g, self.A) + torch.bmm(u, self.B)
        return new_g

    def rollout_diagonal(self, g, u_seq, T, rel_attrs=None):
        g_list = []
        for t in range(T):
            g = self.linear_forward_diagonal(g, u_seq[:, t])
            g_list.append(g[:, None, :, :])
        return torch.cat(g_list, 1)

    @staticmethod
    def batch_pinv(x, I_factor):

        """
        :param x: B x N x D (N > D)
        :param I_factor:
        :return:
        """

        B, N, D = x.size()

        if N < D:
            x = torch.transpose(x, 1, 2)
            N, D = D, N
            trans = True
        else:
            trans = False

        x_t = torch.transpose(x, 1, 2)

        I = torch.eye(D)[None, :, :].expand(B, -1, -1).to(x.device)

        x_pinv = torch.bmm(
            torch.inverse(torch.bmm(x_t, x) + I_factor * I),
            x_t
        )

        if trans:
            x_pinv = torch.transpose(x_pinv, 1, 2)

        return x_pinv


def regularize_state_Soft(states, rel_attrs, stat):
    """
    :param states: B x N x state_dim
    :param rel_attrs: B x N x N x relation_dim
    :param stat: [xxx]
    :return new states: B x N x state_dim
    """
    states_denorm = denormalize([states], [stat[1]], var=True)[0]
    states_denorm_acc = denormalize([states.clone()], [stat[1]], var=True)[0]

    rel_attrs = rel_attrs[0]

    rel_attrs_np = rel_attrs.detach().cpu().numpy()

    def get_rel_id(x):
        return np.where(x > 0)[0][0]

    B, N, state_dim = states.size()
    count = Variable(torch.FloatTensor(np.zeros((1, N, 1, 8))).to(states.device))

    for i in range(N):
        for j in range(N):

            if i == j:
                assert get_rel_id(rel_attrs_np[i, j]) % 9 == 0  # rel_attrs[i, j, 0] == 1
                count[:, i, :, :] += 1
                continue

            assert torch.sum(rel_attrs[i, j]) <= 1

            if torch.sum(rel_attrs[i, j]) == 0:
                continue

            if get_rel_id(rel_attrs_np[i, j]) % 9 == 1:  # rel_attrs[i, j, 1] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 2  # rel_attrs[j, i, 2] == 1
                x0 = 1;
                y0 = 3
                x1 = 0;
                y1 = 2
                idx = 1
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 2:  # rel_attrs[i, j, 2] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 1  # rel_attrs[j, i, 1] == 1
                x0 = 3;
                y0 = 1
                x1 = 2;
                y1 = 0
                idx = 2
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 3:  # rel_attrs[i, j, 3] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 4  # rel_attrs[j, i, 4] == 1
                x0 = 0;
                y0 = 1
                x1 = 2;
                y1 = 3
                idx = 3
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 4:  # rel_attrs[i, j, 4] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 3  # rel_attrs[j, i, 3] == 1
                x0 = 1;
                y0 = 0
                x1 = 3;
                y1 = 2
                idx = 4
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 5:  # rel_attrs[i, j, 5] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 8  # rel_attrs[j, i, 8] == 1
                x = 0;
                y = 3
                idx = 5
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 8:  # rel_attrs[i, j, 8] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 5  # rel_attrs[j, i, 5] == 1
                x = 3;
                y = 0
                idx = 8
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 6:  # rel_attrs[i, j, 6] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 7  # rel_attrs[j, i, 7] == 1
                x = 1;
                y = 2
                idx = 6
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 7:  # rel_attrs[i, j, 7] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 6  # rel_attrs[j, i, 6] == 1
                x = 2;
                y = 1
                idx = 7
            else:
                AssertionError("Unknown rel_attr %f" % rel_attrs[i, j])

            if idx < 5:
                # if connect by two points
                x0 *= 2;
                y0 *= 2
                x1 *= 2;
                y1 *= 2
                count[:, i, :, x0:x0 + 2] += 1
                count[:, i, :, x1:x1 + 2] += 1
                states_denorm_acc[:, i, x0:x0 + 2] += states_denorm[:, j, y0:y0 + 2]
                states_denorm_acc[:, i, x0 + 8:x0 + 10] += states_denorm[:, j, y0 + 8:y0 + 10]
                states_denorm_acc[:, i, x1:x1 + 2] += states_denorm[:, j, y1:y1 + 2]
                states_denorm_acc[:, i, x1 + 8:x1 + 10] += states_denorm[:, j, y1 + 8:y1 + 10]

            else:
                # if connected by a corner
                x *= 2;
                y *= 2
                count[:, i, :, x:x + 2] += 1
                states_denorm_acc[:, i, x:x + 2] += states_denorm[:, j, y:y + 2]
                states_denorm_acc[:, i, x + 8:x + 10] += states_denorm[:, j, y + 8:y + 10]

    states_denorm = states_denorm_acc.view(B, N, 2, state_dim // 2) / count
    states_denorm = states_denorm.view(B, N, state_dim)

    return normalize([states_denorm], [stat[1]], var=True)[0]
    

def cko_sysid_and_pred(model, seq_data, fit_data, pstep, I_factor):
    attrs, states, actions, rel_attrs = seq_data
    attrs_2, states_2, actions_2, rel_attrs_2 = fit_data
    # print('attrs', attrs.shape)           bs x len_seq x num_obj x attr_dim
    # print('states', states.shape)         bs x len_seq x num_obj x state_dim
    # print('actions', actions.shape)       bs x len_seq x num_obj x action_dim
    # print('rel_attrs', rel_attrs.shape)   bs x len_seq x num_obj x num_obj x rel_dim

    data = [attrs, states, actions, rel_attrs]

    # T = args.len_seq
    T = attrs.shape[1] - 1 # seq_data: [last frame of fit_data, output data]
    bs = len(attrs)

    """
    flatten fit data
    """
    time_step = attrs_2.shape[1]
    attrs_flat = get_flat(attrs_2)
    states_flat = get_flat(states_2)
    actions_flat = get_flat(actions_2)
    rel_attrs_flat = get_flat(rel_attrs_2)

    g = model.to_g(attrs_flat, states_flat, rel_attrs_flat, pstep)
    g = g.view(torch.Size([bs, time_step]) + g.size()[1:])

    """
    fit A with fit data
    !!! need to force that rel_attrs in one group to be the same !!!
    """
    G_tilde = g[:, :-1]
    H_tilde = g[:, 1:]
    U_left = actions_2[:, :-1]

    G_tilde = get_flat(G_tilde, keep_dim=True)
    H_tilde = get_flat(H_tilde, keep_dim=True)
    U_left = get_flat(U_left, keep_dim=True)

    A, B, fit_err = model.system_identify(G=G_tilde, H=H_tilde, U=U_left,
                                            rel_attrs=rel_attrs[:1, 0], I_factor=I_factor)

    model.A = model.A.expand(bs, -1, -1)
    model.B = model.B.expand(bs, -1, -1)

    meter_fit_error_val = fit_err.item()

    """
    forward on sequential data
    """

    attrs_flat = get_flat(attrs)
    states_flat = get_flat(states)
    actions_flat = get_flat(actions)
    rel_attrs_flat = get_flat(rel_attrs)

    g = model.to_g(attrs_flat, states_flat, rel_attrs_flat, pstep)

    permu = np.random.permutation(bs * (T + 1))
    split_0 = permu[:bs * (T + 1) // 2]
    split_1 = permu[bs * (T + 1) // 2:]
    min_size = min(split_0.shape[0], split_1.shape[0])
    split_0 = split_0[:min_size]
    split_1 = split_1[:min_size]

    # dist_g = torch.mean((g[split_0] - g[split_1]) ** 2, dim=(1, 2))
    dist_g = torch.mean((g[split_0].flatten(1, -1) - g[split_1].flatten(1, -1)) ** 2, dim=1)
    dist_s = torch.mean((states_flat[split_0] - states_flat[split_1]) ** 2, dim=(1, 2))
    scaling_factor = 10
    loss_metric = torch.abs(dist_g * scaling_factor - dist_s).mean()

    g = g.view(torch.Size([bs, T + 1]) + g.size()[1:])

    """
    rollout 0 -> 1 : T + 1
    """
    U_for_pred = actions[:, : T]
    G_for_pred = model.simulate(T=T, g=g[:, 0], u_seq=U_for_pred, rel_attrs=rel_attrs[:, 0])

    ''' rollout time: T // 2 + 1, T '''
    data_for_ae = [x[:, :T + 1] for x in data]
    data_for_pred = [x[:, 1:T + 1] for x in data]

    # decode state for auto-encoding

    ''' BT x N x 4 '''
    attrs_for_ae_flat = get_flat(data_for_ae[0])
    rel_attrs_for_ae_flat = get_flat(data_for_ae[3])
    decode_s_for_ae = model.to_s(attrs=attrs_for_ae_flat, gcodes=get_flat(g[:, :T + 1]),
                                    rel_attrs=rel_attrs_for_ae_flat, pstep=pstep)

    # decode state for prediction

    ''' BT x N x 4 '''
    attrs_for_pred_flat = get_flat(data_for_pred[0])
    rel_attrs_for_pred_flat = get_flat(data_for_pred[3])
    decode_s_for_pred = model.to_s(attrs=attrs_for_pred_flat, gcodes=get_flat(G_for_pred),
                                    rel_attrs=rel_attrs_for_pred_flat, pstep=pstep)

    loss_auto_encode = F.l1_loss(
        decode_s_for_ae, states[:, :T + 1].reshape(decode_s_for_ae.shape))
    loss_prediction = F.l1_loss(
        decode_s_for_pred, states[:, 1:].reshape(decode_s_for_pred.shape))

    # loss = loss_auto_encode + loss_prediction + loss_metric * args.lambda_loss_metric

    # meter_loss_metric_val = loss_metric.item()
    # meter_loss_ae_val = loss_auto_encode.item()
    # meter_loss_pred_val = loss_prediction.item()

    # meter_dist_g_val = dist_g.mean().item()
    # meter_dist_s_val = dist_s.mean().item()

    return {
        'decode_s_for_pred': decode_s_for_pred.reshape(states[:, 1:].shape),
        'loss_auto_encode': loss_auto_encode,
        'loss_prediction': loss_prediction,
        'loss_metric': loss_metric,
    }