import numpy as np
import torch
from torch import nn
import random
import gym

class Network(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_1 =  nn.Linear(input_dim, 32)
        self.linear_2 = nn.Linear(32, 32)
        self.linear_3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()


    def forward(self, input):
        hidden = self.linear_1(input)
        hidden = self.relu(hidden)
        hidden = self.linear_2(input)
        hidden = self.relu(hidden)
        output = self.linear_3(input)
        return output  

class DQNAgent(nn.Module):

    def __init__(self, state, dim, action_n):
        super().__init__(state_dim, action_n)
        self.state_dim = state_dim
        self.action_n  = action_n
        self.epsilon = 1
        self.memory_size = 10000    
        self.memory = [] 
        self.q = Network(state_dim, action_n)
        self.actions = np.arrange(action_n)
        self.batch_size = 64

# epsilon-greedy
    def get_action(self, state):
        state = torch.FloatTensor(state)
        argmax_action = np.argmax(self.q(state))
        probs = np.ones(self.action_n) * self.epsilon / self.action_n
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(actions, p=probs)
        return action

    def fit(state, action, reward, next_state):
        self.memory.append([state, action, reward, done, next_state])
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        batch = random.sample(self.memory, self.batch_size)
        
        