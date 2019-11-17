from frozen_lake import FrozenLakeEnv
import numpy as np


env = FrozenLakeEnv()
GAMMA = 0.99


def get_q(v):
    q = {}
    for s in env.get_all_states():
        q[s] = {}
        for a in env.get_possible_actions(s):
            q[s][a] = 0
            for next_s in env.get_next_states(s, a):
                q[s][a] += env.get_transition_prob(s,a,next_s) * (env.get_reward(s, a, next_s) + GAMMA * v[next_s])
    return q

def policy_evaluation(pi, iteration_n=100):
    v = {s: 0 for s in env.get_all_states()}
    for _ in range(iteration_n):
        q = get_q(v)
        new_v = {}
        for s in env.get_all_states():
            new_v[s] = 0
            for a in env.get_possible_actions(s):
                new_v[s] +=  pi[s][a] * q[s][a]
        v = new_v
    return v

def policy_improvement(v):
    q = get_q(v)
    pi = {}
    for s in env.get_all_states():
        pi[s] = {}
        if len(env.get_possible_actions(s)) != 0:
            q_max = max(q[s][a] for a in env.get_possible_actions(s))
            there_was_max = False
            for a in env.get_possible_actions(s):
                if q[s][a] == q_max and not there_was_max:
                    pi[s][a] = 1
                    there_was_max = True
                else:
                    pi[s][a] = 0
    return pi

def policy_iteration(iteration_n=20):
    pi = {s: {a : 1/4 for a in env.get_possible_actions(s)} for s in env.get_all_states()}
    for i in range(iteration_n):
        v = policy_evaluation(pi)
        pi = policy_improvement(v)
    return pi

def test(pi, test_n=100, step_n = 100):
    rewards = np.zeros(test_n)
    for i in range(test_n):
        state = env.reset()
        for t in range(step_n):
            prob = [pi[state][a] for a in env.get_possible_actions(state)]
            action = np.random.choice(env.get_possible_actions(state), p=prob)
            next_state, reward, done, _ = env.step(action)
            rewards[i] += reward
            state = next_state
            if done:
                break
    return np.mean(rewards)

def value_iteration(iteration_n=100):
    v = {s: 0 for s in env.get_all_states()}
    for _ in range(iteration_n):
        q = get_q(v)
        for s in env.get_all_states():
            if len(env.get_possible_actions(s)) != 0:
                v[s] = max(q[s][a] for a in env.get_possible_actions(s))
    return policy_improvement(v)


pi1 = policy_iteration()
print(pi1)
print(test(pi1))

pi2 = value_iteration()
print(pi2)
print(test(pi2))
print(pi1 == pi2)