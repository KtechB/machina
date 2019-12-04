"""
An example of Behavioral Cloning.modifi
"""
import numpy as np
import gym
import os
import pickle


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from machina import loss_functional as lf

from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol, DeterministicActionNoisePol
from machina.algos import behavior_clone
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import measure, set_device
from torch.nn.init import kaiming_uniform_, uniform_
from simple_net import PolNet, PolNetLSTM, VNet, DiscrimNet
import numpy as np
import torch
import torch.nn as nn

from machina.pols import BasePol
from machina.pds.gaussian_pd import GaussianPd
from machina.utils import get_device


class GaussianPol_VAE(BasePol):
    """
    Policy with Gaussian distribution.

    Parameters
    ----------
    observation_space : gym.Space
        observation's space
    action_space : gym.Space
        action's space
        This should be gym.spaces.Box
    net : torch.nn.Module
    rnn : bool
    normalize_ac : bool
        If True, the output of network is spreaded for action_space.
        In this situation the output of network is expected to be in -1~1.
    """

    def __init__(self, observation_space, action_space, net, rnn=False,
                 normalize_ac=True):
        BasePol.__init__(self, observation_space, action_space, net, rnn,
                         normalize_ac)
        self.pd = GaussianPd()
        self.to(get_device())

    def forward(self, obs, hs=None, h_masks=None):
        obs = self._check_obs_shape(obs)

        if self.rnn:
            time_seq, batch_size, *_ = obs.shape

            if hs is None:
                if self.hs is None:
                    self.hs = self.net.init_hs(batch_size)
                hs = self.hs

            if h_masks is None:
                h_masks = hs[0].new(time_seq, batch_size, 1).zero_()
            h_masks = h_masks.reshape(time_seq, batch_size, 1)

            mean, log_std, hs = self.net(obs, hs, h_masks)
            self.hs = hs
        else:
            mean, log_std , latent= self.net(obs)
        log_std = log_std.expand_as(mean)
        ac = self.pd.sample(dict(mean=mean, log_std=log_std))
        ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict(mean=mean, log_std=log_std, hs=hs, latent= latent)

    def deterministic_ac_real(self, obs, hs=None, h_masks=None):
        """
        action for deployment
        """
        obs = self._check_obs_shape(obs)

        if self.rnn:
            time_seq, batch_size, *_ = obs.shape
            if hs is None:
                if self.hs is None:
                    self.hs = self.net.init_hs(batch_size)
                hs = self.hs

            if h_masks is None:
                h_masks = hs[0].new(time_seq, batch_size, 1).zero_()
            h_masks = h_masks.reshape(time_seq, batch_size, 1)

            mean, log_std, hs = self.net(obs, hs, h_masks)
            self.hs = hs
        else:
            mean, log_std , latent= self.net(obs)
        mean_real = self.convert_ac_for_real(mean.detach().cpu().numpy())
        return mean_real, mean, dict(mean=mean, log_std=log_std, hs=hs, latent=latent)


def mini_weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(uniform_(m.weight.data, -3e-3, 3e-3))
        m.bias.data.fill_(0)


def weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(kaiming_uniform_(m.weight.data))
        m.bias.data.fill_(0)

## BC
def log_likelihood(pol, batch):
    obs = batch['obs']
    acs = batch['acs']
    _, _, pd_params = pol(obs)
    llh = pol.pd.llh(acs, pd_params)
    pol_loss = -torch.mean(llh)
    return pol_loss


def vae_loss(pol, batch):
    latent_classnum = pol.net.latent_classnum
    obs = batch['obs']
    acs = batch['acs']
    _, _, pd_params = pol(obs) # must be fixed of return num
    llh = pol.pd.llh(acs, pd_params)
    llh_loss = -torch.mean(llh)
    #print( pd_params["latent"])
    log_qy, qy = pd_params["latent"]
    kl_tmp = (qy * (log_qy - torch.log(torch.tensor(1.0 / latent_classnum)))).view(-1, pol.net.latent_dim, pol.net.latent_classnum)
    KL = torch.sum(torch.sum(kl_tmp, 2),1)

    elbo = llh_loss +  KL

    return elbo, llh_loss, KL



def update_pol(pol, optim_pol, batch):
    
    pol_loss, llh, KL = vae_loss(pol, batch)
    
    optim_pol.zero_grad()
    pol_loss.backward()
    optim_pol.step()
    return pol_loss.detach().cpu().numpy(),llh

def train(expert_traj, pol, optim_pol, seq_length=10, epoch=100):
    pol_losses = []
    llh_losses =[]
    
    iterater = expert_traj.random_batch_rnn(batch_size = 1, seq_length = seq_length, epoch=epoch)
    for batch in iterater:
        pol.net.randomize_latent_code()
        for sa_i in range(seq_length):
            loss_tmp = np.zeros((seq_length,2))
            sa = dict()
            sa["obs"] = batch["obs"][sa_i].view(batch["obs"].size()[2:])
            sa["acs"] = batch["acs"][sa_i].view(batch["acs"].size()[2:])

            pol_loss,llh = update_pol(pol, optim_pol, sa)
            loss_tmp[sa_i,0]= pol_loss
            loss_tmp[sa_i,1]= llh
        loss_mean = loss_tmp.mean(axis = 0)
        pol_losses.append(float(loss_mean[0]))
        llh_losses.append(float(loss_mean[1]))
    return dict(PolLoss=pol_losses, llh_loss = llh_losses)


def test(expert_traj, pol):
    pol.eval()
    pol_loss_mean= 0
    counter =0
    iterater = expert_traj.full_batch(epoch=1)
    for batch in iterater:
        with torch.no_grad():
            for sa_i in range(len(batch["obs"])):
                sa = dict()
                sa["obs"] = batch["obs"][sa_i]
                sa["acs"] = batch["acs"][sa_i]
                pol_loss = lf.log_likelihood(pol, sa)
                counter +=1
                pol_loss_mean += pol_loss
    pol_loss_mean = pol_loss_mean/counter
    return dict(Testllh_Loss=[float(pol_loss_mean.detach().cpu().numpy())])

def mini_weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(uniform_(m.weight.data, -3e-3, 3e-3))
        m.bias.data.fill_(0)


def weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(kaiming_uniform_(m.weight.data))
        m.bias.data.fill_(0)

def sample_gumbel(shape, eps=1e-20):
    U = torch.Tensor(shape).uniform_(0,1)#.cuda()
    return -(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y = (y_hard - y).detach() + y
    return y
def get_onehot(index_array, class_num):
    assert isinstance(index_array,torch.Tensor), "index_array must be torch.Tensor"
    assert len(index_array.size()) == 1, "get_onehot index must be 1dim array"
    return torch.eye(class_num)[index_array].view(1,-1)

    
    
class VAEPolNet(nn.Module):
    def __init__(self, observation_space, action_space, h1=200, h2=100, deterministic=False):
        super(VAEPolNet, self).__init__()

        self.deterministic = deterministic

        if isinstance(action_space, gym.spaces.Box):
            self.discrete = False
        else:
            self.discrete = True
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                self.multi = True
            else:
                self.multi = False

        self.state_dim = observation_space.shape[0]
        #self.action_dim = action_space.shape[0]
        self.latent_classnum = 3
        self.latent_dim = 1
        self.randomize_latent_code()
        self.update_latent_code = True

        tau0 = 1.0
        self.tau = torch.autograd.Variable(torch.tensor(tau0))
                

        self.fce1 = nn.Linear(self.state_dim + self.latent_classnum , 64)
        self.fce2 = nn.Linear(64, self.latent_classnum* self.latent_dim)
        #self.fce3 = nn.Linear
        self.fcd1 = nn.Linear(self.latent_classnum * self.latent_dim + self.state_dim, h1)
        self.fcd2 = nn.Linear(h1,h2)
        if not self.discrete:
            self.mean_layer = nn.Linear(h2, action_space.shape[0])
            if not self.deterministic:
                self.log_std_param = nn.Parameter(
                    torch.randn(action_space.shape[0])*1e-10 - 1)
            self.mean_layer.apply(mini_weight_init)
        else:
            if self.multi:
                self.output_layers = nn.ModuleList(
                    [nn.Linear(h2, vec) for vec in action_space.nvec])
                list(map(lambda x: x.apply(mini_weight_init), self.output_layers))
            else:
                self.output_layer = nn.Linear(h2, action_space.n)
                self.output_layer.apply(mini_weight_init)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

        tau0 = 1.0
        self.tau = torch.autograd.Variable(torch.tensor(tau0))

    def sample_latent_code(self):
        self.latent_code = torch.eye(self.latent_classnum)[torch.randint(low = 0, high= self.latent_classnum ,size =  (self.latent_dim,))].view(1,-1)

    def encode(self, x):
        input_and_latent_code  = torch.cat([x, self.latent_code],dim = 1)
        he1 = nn.ReLU()(self.fce1(input_and_latent_code))
        #he2 = self.relu(self.fce2(he1))
        he2 = self.fce2(he1)
        logits_y = he2.view(-1,self.latent_classnum )
        qy = self.softmax(logits_y)# each class likelyhood
        log_qy = torch.log(qy + 1e-20)
        return logits_y, log_qy, qy

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, x):
        # sample and reshape back (shape=(batch_size,N,K))
        # set hard=True for ST Gumbel-Softmax
        assert len(x.size()) ==2, "x can't be batch"
        ge = gumbel_softmax(z, self.tau, hard=False).view(-1, self.latent_dim, self.latent_classnum)
        
        concat_input = torch.cat([x,ge.view(1,self.latent_dim * self.latent_classnum)], dim = 1)
        hd1 = nn.ReLU()(self.fcd1(concat_input))
        hd2 = nn.ReLU()(self.fcd2(hd1))
        #hd3 = self.fcd3(hd2)
        return hd2, ge


    def get_latent_code(self):
        # return numpy label array
        return self.latent_code.view(self.latent_dim, self.latent_classnum).argmax(dim = -1).numpy()        
    def randomize_latent_code(self):
        self.latent_code = torch.eye(self.latent_classnum)[torch.randint(low = 0, high= self.latent_classnum ,size =  (self.latent_dim,))].view(1,-1)


    def forward(self, ob):
        logits_y, log_qy, qy = self.encode(ob)
        
        h , ge = self.decode(logits_y,ob)
        if self.update_latent_code:
            assert ge.size()[0]== 1, "this script can't deal with batch"
            self.latent_code = get_onehot(ge.argmax(dim = -1).view(-1), self.latent_classnum)

        if not self.discrete:
            mean = torch.tanh(self.mean_layer(h))
            if not self.deterministic:
                log_std = self.log_std_param.expand_as(mean)
                return mean, log_std, [log_qy, qy]
            else:
                return mean, [log_qy, qy]
        else:
            if self.multi:
                return torch.cat([torch.softmax(ol(h), dim=-1).unsqueeze(-2) for ol in self.output_layers], dim=-2), [log_qy, qy]
            else:
                return torch.softmax(self.output_layer(h), dim=-1), [log_qy, qy]



      