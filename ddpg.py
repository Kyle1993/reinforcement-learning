import torch
import numpy as np
import random
import torch.nn.functional as F
import gym

from copy import deepcopy
from collections import namedtuple
from torch import nn
from torch import optim
from torch.autograd import Variable
from itertools import count
from random_process import OrnsteinUhlenbeckProcess


############## default setting (total reward converge to 120~ within 100 episodes)##################
# GAMMA = 0.99            # discount rate
# TAN = 0.001             # update speed between double_network
# EPSILON = 1             # random search
# MEMORY_SIZE = int(1e6)
# BATCH_SIZE = 128
# WARM_UP = 5*BATCH_SIZE  # start the training after WARM_UP steps
#
# ACTOR_LR = 5e-4         # 1e-4 is ok
# CRITIC_LR = 5e-3        # 1e-3 is ok
# MAX_STEP = 200          # max_step in each episode
# TEST_ITERVAL = 1        # start test every TEST_ITERVAL steps
# TEST_NUM = 1
# RENDER = False
# RENDER_NUM = 100        # begin render after RENDER_NUM episode
# RANDOM_SEED = 123
# D_EPSILON = 1./10000    # epsilon decay rate
# use rele && init_w && no grad clip && no bn


hyperparameters = {
    'GAMMA' : 0.99,            # discount rate
    'TAN' : 0.001,             # update speed between double_network
    'EPSILON' : 1,             # random search
    'MEMORY_SIZE' : int(1e6),
    'BATCH_SIZE' : 128,

    'ACTOR_LR' : 5e-4,
    'CRITIC_LR' : 5e-3,
    'MAX_STEP' : 200,          # max_step in each episode
    'RANDOM_SEED' : 123,
    'D_EPSILON' : 1./10000     # epsilon decay rate
     }

hyperparameters_train = deepcopy(hyperparameters)
hyperparameters_train['MODE'] = 'train'
hyperparameters_test = deepcopy(hyperparameters)
hyperparameters_test['MODE'] = 'test'

WARM_UP = 5 * hyperparameters['BATCH_SIZE']  # start the training after WARM_UP steps
TEST_ITERVAL = 1                             # start test every TEST_ITERVAL steps
TEST_NUM = 1
RENDER = False                               # if render
RENDER_NUM = 100                             # begin render after RENDER_NUM episode

# from hyperboard import Agent
# HBagent = Agent(username='jianglibin',password='1993610',address='127.0.0.1',port=5000)
# test_record = HBagent.register(hyperparameters_train,'reward')
# train_record = HBagent.register(hyperparameters_test,'reward')

Memory_Unit = namedtuple('Memory_Unit',('state','action','next_state','reward','not_done'))



class Memory():
    def __init__(self,max_len):
        self.max_len = max_len
        self.current_index = -1
        self.counter = 0
        self.memory = [None for _ in range(self.max_len)]

    def append(self,*args):
        self.current_index = (self.current_index+1) % self.max_len
        self.memory[self.current_index] = Memory_Unit(*args)
        self.counter += 1

    def sample(self,batch_size=64):
        if batch_size > self.max_len:
            raise RuntimeError()
        batch = random.sample(self.memory[:min(self.counter,self.max_len)],batch_size)
        batchs = Memory_Unit(*zip(*batch))

        state_batch = np.asarray(batchs.state,dtype=np.float32)
        action_batch = np.asarray(batchs.action,dtype=np.float32)
        next_state_batch = np.asarray(batchs.next_state,dtype=np.float32)
        reward_batch = np.asarray(batchs.reward,dtype=np.float32)
        done_batch = np.asarray(batchs.not_done,dtype=np.int32)

        return state_batch,action_batch,next_state_batch,reward_batch,done_batch


class Actor(nn.Module):
    def __init__(self,state_size,action_size,h1_size=400,h2_size=300,init_w=3e-3):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(state_size,h1_size)
        self.bn1 = nn.BatchNorm1d(h1_size)
        self.fc2 = nn.Linear(h1_size,h2_size)
        self.bn2 = nn.BatchNorm1d(h2_size)
        self.fc3 = nn.Linear(h2_size,action_size)
        self.init_weight(init_w)

    def init_weight(self,init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w,init_w)

    def forward(self, s):
        # out = F.tanh(self.bn1(self.fc1(s)))
        # out = F.tanh(self.bn2(self.fc2(out)))
        out = F.relu(self.fc1(s))
        out = F.relu(self.fc2(out))
        out = F.tanh(self.fc3(out))
        return out

class Critic(nn.Module):
    def __init__(self,state_size,action_size,h1_size=400,h2_size=400,init_w=3e-3):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state_size,h1_size)
        self.bn1 = nn.BatchNorm1d(h1_size)
        self.fc2 = nn.Linear(h1_size+action_size,h2_size)
        self.bn2 = nn.BatchNorm1d(h2_size)
        self.fc3 = nn.Linear(h2_size,1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, s,a):
        # out = F.tanh(self.bn1(self.fc1(s)))
        out = F.relu(self.fc1(s))
        out = torch.cat([out,a],1)
        # out = F.tanh(self.bn2(self.fc2(out)))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class DDPG(object):
    def __init__(self,state_size,action_size,memory_size,batch_size=128,tan=0.001,actor_lr=0.001,critic_lr=0.001,epsilon=1.):


        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.tan = tan
        self.warmup = WARM_UP
        self.epsilon = epsilon
        self.epsilon_decay = hyperparameters['D_EPSILON']

        self.actor = Actor(state_size,action_size)
        self.actor_target = Actor(state_size,action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic = Critic(state_size,action_size)
        self.critic_target = Critic(state_size,action_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.memory = Memory(memory_size)
        self.criterion = nn.MSELoss()


        self.random_process = OrnsteinUhlenbeckProcess(size=action_size, theta=0.15, mu=0., sigma=0.2)

        copy_parameter(self.actor,self.actor_target)
        copy_parameter(self.critic,self.critic_target)

    def train(self):

        # if not warm up
        if self.memory.counter < self.warmup:
            return

        # get batch
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.memory.sample(self.batch_size)
        action_batch = action_batch.reshape((-1,self.action_size))
        reward_batch = reward_batch.reshape((-1,1))
        done_batch = done_batch.reshape((-1,1))

        # update critic
        nsb = Variable(torch.from_numpy(next_state_batch).float(),volatile=True)  # next_state_batch
        nab = self.actor_target(nsb)   # next_action_batch
        next_q = self.critic_target(nsb,nab)
        next_q.volatile = False

        rb = Variable(torch.from_numpy(reward_batch).float())  # reward_batch
        db = Variable(torch.from_numpy(done_batch).float()) # if next state is None, next_q should be 0, which means q = r
        q_target = rb + hyperparameters['GAMMA']*db*next_q

        sb_grad = Variable(torch.from_numpy(state_batch).float())  # state_batch with grad, mean output need grad
        ab = Variable(torch.from_numpy(action_batch).float())  # action_batch
        q_eval = self.critic(sb_grad,ab)

        value_loss = self.criterion(q_eval,q_target)
        self.critic.zero_grad()
        value_loss.backward()
        # nn.utils.clip_grad_norm(self.critic.parameters(),0.8)
        self.critic_optimizer.step()

        # update actor
        sb_grad = Variable(torch.from_numpy(state_batch).float())  # state_batch
        aab = self.actor(sb_grad)  # actor_action_batch

        q = self.critic(sb_grad,aab)
        policy_loss = torch.mean(-q)
        self.actor.zero_grad()
        policy_loss.backward()
        # nn.utils.clip_grad_norm(self.actor.parameters(),0.8)
        self.actor_optimizer.step()

        # update parameter between two network
        update_parameter(self.critic_target,self.critic,self.tan)
        update_parameter(self.actor_target,self.actor,self.tan)

    def select_action(self,s,is_train=True,decay_e=True):
        if self.memory.counter < self.warmup:
            action = env.action_space.sample()[0]
            # action = random.uniform(-2.,2.)
            return action
        state = Variable(torch.FloatTensor([s]).float())
        action = self.actor(state).squeeze(1).data.numpy()
        action += is_train * max(self.epsilon,0) * self.random_process.sample()
        action = float(np.clip(action,-1.,1.)[0])

        if decay_e:
            if self.memory.counter > self.warmup:
                self.epsilon -= self.epsilon_decay
        return action

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

def copy_parameter(target,source):
    for tp,sp in zip(target.parameters(),source.parameters()):
        tp.data.copy_(sp.data)

def update_parameter(target,source,tau):
    for tp,sp in zip(target.parameters(),source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)

def test(env,agent,episode):
    total_step = 0
    total_reward = 0
    for i in range(TEST_NUM):
        state = env.reset()
        for test_step in range(hyperparameters['MAX_STEP']):
            if RENDER:
                env.render()
            action = agent.select_action(state, is_train=False, decay_e=False)
            next_state, reward, done, info = env.step([action])
            total_reward += reward
            state = next_state
            if done:
                break
        total_step += test_step
    # HBagent.append(test_record,episode,total_reward/TEST_NUM)
    print('Epsiode {}\tAvg_length :{}\tAvg_reward :{}\tEpsilon :{}'.format(episode, total_step / TEST_NUM, total_reward / TEST_NUM,agent.epsilon))

class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)


################ mian process #################

env = NormalizedEnv(gym.make('Pendulum-v0'))
# env = gym.make('Pendulum-v0')

np.random.seed(hyperparameters['RANDOM_SEED'])
env.seed(hyperparameters['RANDOM_SEED'])
torch.manual_seed(hyperparameters['RANDOM_SEED'])

nb_state = env.observation_space.shape[0]
nb_action = env.action_space.shape[0]

ddpg_agent = DDPG(nb_state,nb_action,hyperparameters['MEMORY_SIZE'],batch_size=hyperparameters['BATCH_SIZE'],tan=hyperparameters['TAN'],actor_lr=hyperparameters['ACTOR_LR'],critic_lr=hyperparameters['CRITIC_LR'],epsilon=hyperparameters['EPSILON'])


for episode in count(0):
    train_total_reward = 0
    state = env.reset()
    if episode > RENDER_NUM:
        RENDER = True
    for step in range(hyperparameters['MAX_STEP']):
        action = ddpg_agent.select_action(state,decay_e=True)
        next_state,reward,done,info = env.step([action])
        train_total_reward += reward

        # in this game, the last state shuold not append to memory, because it's finished by the length of game, not state
        not_done = not done
        if not_done:
            ddpg_agent.memory.append(state,action,next_state,reward,not_done)
        ddpg_agent.train()

        state = next_state
        if done:
            break

    if episode % TEST_ITERVAL == 0:
        # HBagent.append(train_record,episode,train_total_reward/TEST_NUM)
        test(env,ddpg_agent,episode)






