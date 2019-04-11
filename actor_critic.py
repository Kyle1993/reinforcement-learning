import gym
import numpy as np
from itertools import count
from collections import namedtuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


gamma = 0.99
actor_lr = 1e-5
critic_lr = 1e-4
max_step = 200
batch_size = 128

print_interval = 10
test_num = 3

RENDER = False
render_num = 2500


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Memory():
    def __init__(self,max_len,batch_size):
        self.max_len = max_len
        self.current_index = -1
        self.counter = 0
        self.memory = [None for _ in range(self.max_len)]
        self.batch_size = batch_size

    def append(self, state, action_prob, action, next_state, reward, done):
        self.current_index = (self.current_index + 1) % self.max_len
        self.memory[self.current_index] = (state, action_prob, action, next_state, reward, done)
        self.counter += 1

    def sample(self):
        batch = random.sample(self.memory[:min(self.counter, self.max_len)], self.batch_size)
        batch = list(zip(*batch))

        state_batch = np.asarray(batch[0], dtype=np.float32)  # batch * myself_num * enemy_num+1 * state_size
        actionprob_batch = np.asarray(batch[1], dtype=np.float32)
        action_batch = np.asarray(batch[2],dtype=np.int)
        next_state_batch = np.asarray(batch[3], dtype=np.float32)
        reward_batch = np.asarray(batch[4], dtype=np.float32)
        done_batch = np.asarray(batch[5], dtype=np.int32)

        return state_batch, actionprob_batch, action_batch, next_state_batch, reward_batch, done_batch

class Actor(nn.Module):

    def __init__(self,state_szie,action_size,hidden_size=40,init_w=3e-2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_szie,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)
        # self.init_weight(init_w)

    def forward(self,s):
        out = F.relu(self.fc1(s))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out),dim=1)
        return out

    def init_weight(self,init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w,init_w)

class Critic(nn.Module):

    def __init__(self,input_size,hidden_size=40,output_size=1,init_w=3e-2):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size,4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,1)
        # self.init_weight(init_w)

    def forward(self,s):
        out = F.relu(self.fc1(s))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def init_weight(self,init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

class Actor_Critic():
    def __init__(self,nb_state,nb_action,gamma,critic_lr,actor_lr,init_w=3e-3):
        self.memory = Memory(int(1e5),128)
        self.gamma = gamma
        self.critic = Critic(nb_state,init_w=init_w)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)

        self.actor = Actor(nb_state,nb_action,init_w=init_w)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)

    # in test case, only return the action with max prob
    def select_action(self,state,is_train=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(Variable(state))
        if not is_train:
            return torch.max(probs,1,keepdim=True)[1].data[0,0]
        return probs

    def train(self):
        if self.memory.counter<3*batch_size:
            return
        state_batch,actionprob_batch,action_batch,next_state_batch,reward_batch,done_batch = self.memory.sample()

        state_batch = Variable(torch.from_numpy(state_batch).view(-1,nb_state)).float()
        actions_prob_batch = Variable(torch.from_numpy(actionprob_batch).view(-1,nb_aciton)).float()
        action_batch = Variable(torch.from_numpy(action_batch).view(-1,1))
        next_state_batch = Variable(torch.from_numpy(next_state_batch).view(-1,nb_state)).float()
        reward_batch = Variable(torch.from_numpy(reward_batch).view(-1,1)).float()
        done_batch = Variable(torch.from_numpy(done_batch).view(-1,1)).float()

        ########## TD_error ##########
        v_eval = self.critic(state_batch)
        v_next = self.critic(next_state_batch)
        v_target = reward_batch+self.gamma*done_batch*v_next
        v_target.detach_()
        td_error = (v_target-v_eval).detach()

        #---------------update critic---------------------
        value_loss = torch.nn.functional.mse_loss(v_eval,v_target)
        self.optimizer_critic.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic.parameters(),0.8)
        self.optimizer_critic.step()

        #----------------update actor----------------------
        self.optimizer_actor.zero_grad()
        log_softmax_actions = torch.log(self.actor(state_batch))
        policy_loss = - torch.mean(log_softmax_actions.gather(1,action_batch)* td_error)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm(self.actor.parameters(),0.8)
        self.optimizer_actor.step()


# only chose the action with max prob
def test(env,agent,episode):
    total_length = 0
    total_reward = 0
    for i in range(test_num):
        state = env.reset()
        for test_step in range(max_step):
            if RENDER:
                env.render()
            action = agent.select_action(state,is_train=False)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        total_length += test_step
    print('Episode {}\tAvg_Length:{:5f}\tAvg_Reward:{:5f}'.format(episode, total_length / test_num,
                                                                  total_reward / test_num))


env = gym.make('CartPole-v0')

nb_aciton = env.action_space.n
nb_state = env.observation_space.shape[0]

actor_critic = Actor_Critic(nb_state,nb_aciton,gamma,critic_lr,actor_lr)

for i_episode in count(1):
    state = env.reset()
    if i_episode > render_num:
        RENDER = True
    for t in range(max_step):
        # print('----------------------')
        actions_prob = actor_critic.select_action(state)
        action = actions_prob.multinomial(1).data[0,0]
        next_state, reward, done, _ = env.step(action)

        actor_critic.memory.append(state,actions_prob[0].data.numpy(),action,next_state,reward,float(not done))
        actor_critic.train()
        # during the train, the parameter always be NaN, I can't fix it
        if np.isnan(np.sum(list(actor_critic.actor.fc1.parameters())[0].data.numpy())):
            print('NAN')
        if done:
            break
        state = next_state

    if i_episode%print_interval == 0:
        test(env,actor_critic,i_episode)

