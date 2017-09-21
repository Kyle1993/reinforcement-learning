import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


gamma = 0.99
actor_lr = 1e-2
critic_lr = 1e-2
max_step = 10000

print_interval = 10
test_num = 3

RENDER = False
render_num = 200

Memory_unit = namedtuple('Memory_unit', ['state','action','next_state', 'reward'])

class Actor(nn.Module):

    def __init__(self,state_szie,action_size,hidden_size=40,init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_szie,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)
        self.init_weight(init_w)

    def forward(self,s):
        out = F.relu(self.fc1(s))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out))
        return out

    def init_weight(self,init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w,init_w)

class Critic(nn.Module):

    def __init__(self,input_size,hidden_size=40,output_size=1,init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)
        self.init_weight(init_w)

    def forward(self,s):
        out = F.relu(self.fc1(s))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def init_weight(self,init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

class Advantage_Actor_Critic():
    def __init__(self,nb_state,nb_action,critic_lr,actor_lr,init_w=3e-3):
        self.memory = []
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

    def discount_reward(self,r, gamma,final_r):
        discounted_r = np.zeros_like(r)
        running_add = final_r
        for t in reversed(range(0, len(r))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r


    def train_episode(self):

        batch = Memory_unit(*zip(*self.memory))

        state_batch = Variable(torch.Tensor(batch.state).view(-1,nb_state))
        action_batch = Variable(torch.LongTensor(batch.action).view(-1,1))
        # next_state_batch = Variable(torch.FloatTensor(batch.next_state).view(-1,nb_state))
        # reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1,1))

        ################# Advantage ################
        #---------------update critic---------------------
        v_eval = self.critic(state_batch)
        q_target = Variable(torch.FloatTensor(self.discount_reward(batch.reward,0.99,1))).unsqueeze(1)

        value_loss = torch.nn.functional.mse_loss(v_eval,q_target)
        self.optimizer_critic.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic.parameters(),0.8)
        self.optimizer_critic.step()

        #----------------update actor----------------------
        self.optimizer_actor.zero_grad()
        log_softmax_actions = torch.log(self.actor(state_batch))
        v_eval = self.critic(state_batch)
        q_target = Variable(torch.FloatTensor(self.discount_reward(batch.reward,0.99,1))).unsqueeze(1)
        advantage = q_target - v_eval
        policy_loss = - torch.mean(log_softmax_actions.gather(1,action_batch)* advantage)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm(self.actor.parameters(),0.8)
        self.optimizer_actor.step()

        self.memory = []

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



def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


env = gym.make('CartPole-v0')

nb_aciton = env.action_space.n
nb_state = env.observation_space.shape[0]

a2c = Advantage_Actor_Critic(nb_state,nb_aciton,critic_lr,actor_lr)

for i_episode in count(1):
    state = env.reset()
    if i_episode > render_num:
        RENDER = True
    for t in range(max_step):
        actions_prob = a2c.select_action(state)
        action = actions_prob.multinomial(1).data[0,0]
        next_state, reward, done, _ = env.step(action)

        a2c.memory.append(Memory_unit(state,action,next_state,reward))

        if done:
            break
        state = next_state

    a2c.train_episode()

    if i_episode%print_interval == 0:
        test(env,a2c,i_episode)


