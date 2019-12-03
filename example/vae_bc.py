"""
An example of Behavioral Cloning.modifi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from machina import loss_functional as lf



model = VAE()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, log_qy, qy, data):
    sigmoid = nn.Sigmoid()
    kl_tmp = (qy * (log_qy - torch.log(torch.tensor(1.0 / self.latent_classnum)))).view(-1, N, K)
    KL = torch.sum(torch.sum(kl_tmp, 2),1)
    shape = data.size()
    #elbo = torch.sum(recon_x.log_prob(data.view(shape[0], shape[1] * shape[2] * shape[3])), 1) - KL
    data_ = data.view(shape[0], shape[1] * shape[2] * shape[3])
    # calculate binary cross entropy using explicit calculation rather than using pytorch distribution API
    bce = torch.sum(data_ * torch.log(sigmoid(recon_x)) + (1 - data_) * torch.log(1 - sigmoid(recon_x)), 1)
    elbo = bce - KL
    return torch.mean(-elbo), torch.mean(bce), torch.mean(KL)

ANNEAL_RATE=0.00003
MIN_TEMP=0.5

def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        px, log_qy, qy = model(data)
        recon_x = torch.distributions.bernoulli.Bernoulli(logits=px)
        #loss = loss_function(recon_x, log_qy, qy, data)
        loss, bce, KL = loss_function(px, log_qy, qy, data)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        #if batch_idx % 1000 == 1:
        #    tau = Variable(torch.tensor(np.maximum(tau0 * np.exp(-ANNEAL_RATE * batch_idx), MIN_TEMP)))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tBCE: {:.6f} \tKL: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0], bce.data[0], KL.data[0]))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    M = 100 * N
    np_y = np.zeros((M, K))
    np_y[range(M), np.random.choice(K, M)] = 1
    np_y = np.reshape(np_y, [100, N, K])

    px = model.decode(Variable(torch.tensor(np_y).cuda()))
    recon_x = torch.nn.Sigmoid()(px).detach().cpu().numpy()
    #recon_x = torch.distributions.Bernoulli(logits=px).sample()
    np_x = recon_x.reshape((10, 10, 28, 28))
    # split into 10 (1,10,28,28) images, concat along columns -> 1,10,28,280
    np_x = np.concatenate(np.split(np_x, 10, axis=0), axis=3)
    # split into 10 (1,1,28,280) images, concat along rows -> 1,1,280,280
    np_x = np.concatenate(np.split(np_x, 10, axis=1), axis=2)
    x_img = np.squeeze(np_x)
    plt.imshow(x_img, cmap=plt.cm.gray, interpolation='none')
    plt.show()



## BC
def log_likelihood(pol, batch):
    obs = batch['obs']
    acs = batch['acs']
    _, _, pd_params = pol(obs)
    llh = pol.pd.llh(acs, pd_params)
    pol_loss = -torch.mean(llh)
    return pol_loss

def vae_loss(pol, batch):
    
    obs = batch['obs']
    acs = batch['acs']
    _, _, pd_params = pol(obs)
    llh = pol.pd.llh(acs, pd_params)
    llh_loss = -torch.mean(llh)
    log_qy, qy = pd.params["latent"]
    kl_tmp = (qy * (log_qy - torch.log(torch.tensor(1.0 / self.latent_classnum)))).view(-1, N, K)
    KL = torch.sum(torch.sum(kl_tmp, 2),1)

    elbo = llh +  KL

    return elbo, llh, KL

def loss_function(recon_x, log_qy, qy, data):
    sigmoid = nn.Sigmoid()
    kl_tmp = (qy * (log_qy - torch.log(torch.tensor(1.0 / self.latent_classnum)))).view(-1, N, K)
    KL = torch.sum(torch.sum(kl_tmp, 2),1)
    shape = data.size()
    #elbo = torch.sum(recon_x.log_prob(data.view(shape[0], shape[1] * shape[2] * shape[3])), 1) - KL
    data_ = data.view(shape[0], shape[1] * shape[2] * shape[3])
    # calculate binary cross entropy using explicit calculation rather than using pytorch distribution API
    bce = torch.sum(data_ * torch.log(sigmoid(recon_x)) + (1 - data_) * torch.log(1 - sigmoid(recon_x)), 1)
    elbo = bce - KL
    return torch.mean(-elbo), torch.mean(bce), torch.mean(KL)


def update_pol(pol, optim_pol, batch):
    kl_tmp = (qy * (log_qy - torch.log(torch.tensor(1.0 / self.latent_classnum)))).view(-1, N, K)
    KL = torch.sum(torch.sum(kl_tmp, 2),1)
    
    pol_loss,llh, KL = vae_loss(pol, batch)
    
    optim_pol.zero_grad()
    pol_loss.backward()
    optim_pol.step()
    return pol_loss.detach().cpu().numpy()


def train(expert_traj, pol, optim_pol, batch_size):
    pol_losses = []
    iterater = expert_traj.iterate_once(batch_size)
    for batch in iterater:
        pol_loss = update_pol(pol, optim_pol, batch)
        pol_losses.append(pol_loss)
    return dict(PolLoss=pol_losses)


def test(expert_traj, pol):
    pol.eval()
    iterater = expert_traj.full_batch(epoch=1)
    for batch in iterater:
        with torch.no_grad():
            pol_loss = lf.log_likelihood(pol, batch)
    return dict(TestPolLoss=[float(pol_loss.detach().cpu().numpy())])

import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, uniform_
import torch.nn.functional as F
import gym

def mini_weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(uniform_(m.weight.data, -3e-3, 3e-3))
        m.bias.data.fill_(0)


def weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(kaiming_uniform_(m.weight.data))
        m.bias.data.fill_(0)

def sample_gumbel(shape, eps=1e-20):
    U = torch.Tensor(shape).uniform_(0,1).cuda()
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
    return torch.eye(class_num)[index_array]

    
    
class VAEPolNet(nn.Module):
    def __init__(self, observation_space, action_space, h1=200, h2=100, deterministic=False):
        super(PolNet, self).__init__()

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
        self.tau = Variable(torch.tensor(tau0))
                

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
        self.tau = Variable(torch.tensor(tau0))

    def sample_latent_code():
        self.latent_code = torch.eye(self.latent_class_num)[torch.randint(low = 0, high= self.latent_classnum ,size =  (self.latent_dim,))]

    def encode(self, x):
        input_and_latent_code  = torch.cat([x, self.latent_code],dim = 0)
        he1 = nn.Relu(self.fce1(input_and_latent_code))
        #he2 = self.relu(self.fce2(he1))
        he2 = self.fce2(he1)
        logits_y = he2.view(-1,self.latent_classnum )
        qy = self.softmax(logits_y)# each class likelyhood
        log_qy = torch.log(qy + 1e-20)
        return logits_y, log_qy, qy

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, x):
        # sample and reshape back (shape=(batch_size,N,K))
        # set hard=True for ST Gumbel-Softmax
        assert len(x.size()) ==1, "x can't be batch"
        ge = gumbel_softmax(z, self.tau, hard=False).view(-1, self.latent_dim, self.latent_classnum)
        
        concat_input = torch.cat([x,ge.view(self.latent_dim * self.latent_classnum)], dim = 0)
        hd1 = nn.ReLU(self.fcd1(concat_input))
        hd2 = nn.ReLU(self.fcd2(hd1))
        #hd3 = self.fcd3(hd2)
        return hd2, ge


    def get_latent_code(self):
        # return numpy label array
        return self.latent_code.view(self.latent_dim, self.latent_classnum).argmax(dim = -1).numpy()        
    def randomize_latent_code(self):
        self.latent_code = torch.eye(self.latent_class_num)[torch.randint(low = 0, high= self.latent_classnum ,size =  (self.latent_dim,))]


    def forward(self, ob):
        logits_y, log_qy, qy = self.encode(ob)
        
        h , ge = self.decode(logits_y)
        if self.update_latent_code:
            assert ge.size()[0]== 1, "this script can't deal with batch"
            self.latent_code = get_onehot(ge.argmax(dim = -1).view(-1), class_num)

        if not self.discrete:
            mean = torch.tanh(self.mean_layer(h))
            if not self.deterministic:
                log_std = self.log_std_param.expand_as(mean)
                return [mean, log_std], [log_qy, qy]
            else:
                return mean
        else:
            if self.multi:
                return torch.cat([torch.softmax(ol(h), dim=-1).unsqueeze(-2) for ol in self.output_layers], dim=-2), [log_qy, qy]
            else:
                return torch.softmax(self.output_layer(h), dim=-1), [log_qy, qy]
import argparse
import json
import os
import copy
from pprint import pprint
import pickle

import numpy as np
import torch
import gym

from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol, DeterministicActionNoisePol
from machina.algos import behavior_clone
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import measure, set_device

from simple_net import PolNet, PolNetLSTM, VNet, DiscrimNet

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage',
                    help='Directory name of log.')
parser.add_argument('--env_name', type=str,
                    default='Pendulum-v0', help='Name of environment.')
parser.add_argument('--c2d', action='store_true',
                    default=False, help='If True, action is discretized.')
parser.add_argument('--record', action='store_true',
                    default=False, help='If True, movie is saved.')
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_epis', type=int,
                    default=100000000, help='Number of episodes to run.')
parser.add_argument('--num_parallel', type=int, default=4,
                    help='Number of processes to sample.')
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')

parser.add_argument('--expert_dir', type=str, default='../data/expert_epis')
parser.add_argument('--expert_fname', type=str,
                    default='Pendulum-v0_100epis.pkl')

parser.add_argument('--max_epis_per_iter', type=int,
                    default=10, help='Number of episodes in an iteration.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--pol_lr', type=float, default=1e-4,
                    help='Policy learning rate.')
parser.add_argument('--h1', type=int, default=32)
parser.add_argument('--h2', type=int, default=32)

parser.add_argument('--tau', type=float, default=0.001,
                    help='Coefficient of target function.')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount factor.')
parser.add_argument('--lam', type=float, default=1,
                    help='Tradeoff value of bias variance.')

parser.add_argument('--train_size', type=int, default=0.7,
                    help='Size of training data.')
parser.add_argument('--check_rate', type=int, default=0.05,
                    help='Rate of performance check per epoch.')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--deterministic', action='store_true',
                    default=False, help='If True, policy is deterministic.')
args = parser.parse_args()

device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

if not os.path.exists(args.log):
    os.makedirs(args.log)

with open(os.path.join(args.log, 'args.json'), 'w') as f:
    json.dump(vars(args), f)
pprint(vars(args))

if not os.path.exists(os.path.join(args.log, 'models')):
    os.makedirs(os.path.join(args.log, 'models'))

np.random.seed(args.seed)
torch.manual_seed(args.seed)

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)
logger.add_tensorboard_output(args.log)

env = GymEnv(args.env_name, log_dir=os.path.join(
    args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)
if args.c2d:
    env = C2DEnv(env)

observation_space = env.observation_space
action_space = env.action_space

pol_net = PolNet(observation_space, action_space)
if isinstance(action_space, gym.spaces.Box):
    pol = GaussianPol(observation_space, action_space, pol_net)
elif isinstance(action_space, gym.spaces.Discrete):
    pol = CategoricalPol(observation_space, action_space, pol_net)
elif isinstance(action_space, gym.spaces.MultiDiscrete):
    pol = MultiCategoricalPol(observation_space, action_space, pol_net)
else:
    raise ValueError('Only Box, Discrete, and MultiDiscrete are supported')

sampler = EpiSampler(env, pol, num_parallel=args.num_parallel, seed=args.seed)
optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)

with open(os.path.join(args.expert_dir, args.expert_fname), 'rb') as f:
    expert_epis = pickle.load(f)
train_epis, test_epis = ef.train_test_split(
    expert_epis, train_size=args.train_size)
train_traj = Traj()
train_traj.add_epis(train_epis)
train_traj.register_epis()
test_traj = Traj()
test_traj.add_epis(test_epis)
test_traj.register_epis()
expert_rewards = [np.sum(epi['rews']) for epi in expert_epis]
expert_mean_rew = np.mean(expert_rewards)
logger.log('expert_score={}'.format(expert_mean_rew))
logger.log('num_train_epi={}'.format(train_traj.num_epi))

max_rew = -1e6

for curr_epoch in range(args.epoch):
    result_dict = behavior_clone.train(
        train_traj, pol, optim_pol,
        args.batch_size
    )
    test_result_dict = behavior_clone.test(test_traj, pol)

    for key in test_result_dict.keys():
        result_dict[key] = test_result_dict[key]

        if curr_epoch % int(args.check_rate * args.epoch) == 0 or curr_epoch == 0:
            with measure('sample'):
                paths = sampler.sample(
                    pol, max_epis=args.max_epis_per_iter)
            rewards = [np.sum(path['rews']) for path in paths]
            mean_rew = np.mean([np.sum(path['rews']) for path in paths])
            logger.record_results_bc(args.log, result_dict, score_file,
                                     curr_epoch, rewards,
                                     plot_title=args.env_name)

        if mean_rew > max_rew:
            torch.save(pol.state_dict(), os.path.join(
                args.log, 'models', 'pol_max.pkl'))
            torch.save(optim_pol.state_dict(), os.path.join(
                args.log, 'models', 'optim_pol_max.pkl'))
            max_rew = mean_rew

        torch.save(pol.state_dict(), os.path.join(
            args.log, 'models', 'pol_last.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(
            args.log, 'models', 'optim_pol_last.pkl'))

del sampler
