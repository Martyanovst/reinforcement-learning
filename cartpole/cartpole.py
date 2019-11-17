import gym
import gym_maze
import random
import time
import numpy as np
import torch
from torch import nn

session_n = 100
session_len = 20000
q_param = 0.9 
size = 100

class Agent():
    def __init__(self, states_dim, actions_n):
        self.states_dim = states_dim
        self.actions_n = actions_n
        self.actions = np.arange(actions_n)
        self.linear_1 = nn.Linear(states_dim, 50)
        self.linear_2 = nn.Linear(50, 20)
        self.linear_3 = nn.Linear(20, actions_n)
        self.relu = nn.Relu()
        self.epsilon  = 1
        self.epsilon_dec = 0.95
        self.epsilon_min = 0.5
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.1)


    def forward(self, input):
        hidden = self.linear_1(input)
        hidden = self.relu(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.relu(hidden)
        hidden = self.linear_3(hidden)
        return nn.Softmax(hidden)

    def get_action(self, state):
        action_prob = (1 - self.epsilon) * self.forward(state) + self.epsilon / self.actions_n
        action = np.random.choice(self.actions, p=action_prob)
        return int(action)

    def update_policy(self, elite_sessions):
        elite_states = []
        elite_actions = []

        for session in elite_sessions:
            states, actions, total_rewards = session
            elite_states.extend(states)
            elite_actions.extend(actions)

        
        elite_states = torch.tensor(elite_states)
        elite_actions = torch.tensor(elite_actions)
        loss = nn.Log(elite_states, elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


env = gym.make("CartPole-v0")

agent = Agent(4, 2)

def get_state(obs):
    return int(obs[0] * size + obs[1])

def get_session(session_len):
    obs = env.reset()
    states = []
    actions = []
    total_reward = 0
    for t in range(session_len):
        # state = get_state(obs)
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    session = [states, actions, total_reward]
    return session

def get_elite_sessions(sessions, q_param):
    total_rewards = np.array([session[2] for session in sessions])
    print(np.mean(total_rewards))
    quantile = np.quantile(total_rewards, q_param)
    elite_sessions = []
    for session in sessions:
        states, actions, total_reward = session
        if total_reward > quantile:
            elite_sessions.append(session)
    return elite_sessions


for _ in range(20):
    sessions = [get_session(session_len) for _ in range(session_n)]
    elite_sessions = get_elite_sessions(sessions, q_param)
    if len(elite_sessions) > 0:
        agent.update_policy(elite_sessions)

done = False
state = env.reset()
while not done:
    state, reward, done, _ = env.step(agent.get_action(get_state(state))) 
    env.render()
    time.sleep(0.1)







