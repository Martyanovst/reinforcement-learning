def SARSA(env, episode_n=30, t_max=50):
    state_n = env.observation_space.n
    action_n = env.action_space.n
    Q = np.zeros((state_n, action_n))
    total_rewards = np.zeros(episode_n)
    for episode in range(episode_n):
        s = env.reset()
        for t in range(t_max):
            next_s, r, done, _ = env.step(a)
           total_rewards[episode] += r
            next_a = get_epsilon_greedy_action(Q[next_s])
            Q[s][a] += alpha * (r + gamma * Q[next_s][next_a] - Q[s][a])
            rewards[t] = r
            s, a = next_s, next_a
            if done:
                break
    return total_rewards
