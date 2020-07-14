#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:44:38 2020
https://www.datamachinist.com/reinforcement-learning/part-6-q-learning-for-continuous-state-problems/
@author: ar
"""
import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time


######################################
# CRIAR E VISUALIZAR ENTORNO
######################################
# n_actions = env.action_space.n
# n_states = env.observation_space.shape[0]
# print("Action space size: ", n_actions)
# print("State space size: ", n_states)

# print('states high value:')
# print(env.observation_space.high[0])
# print(env.observation_space.high[1])
# print(env.observation_space.high[2])
# print(env.observation_space.high[3])

# print('states low value:')
# print(env.observation_space.low[0])
# print(env.observation_space.low[1])
# print(env.observation_space.low[2])
# print(env.observation_space.low[3])


######################################
# Q-LEARNING E FUNCOES DE DISCRETIZACAO
######################################
def Q_learning_train_test(buckets,  # define o numero de buckets para cada estado valor (x, x', theta, theta')
                          n_episodes,  # numero de episodios
                          n_steps,  # numero de maximo de steps por episodio
                          min_alpha,  # taxa de aprendizado
                          min_epsilon,  # taxa de explotacao
                          gamma,  # fator de desconto
                          ada_divisor):  # taxa de decaimento para os parametro de alpha e epsilon
    # exp_name,   # nome do experimento
    # exp_color): # color da linha do experimento no grafico
    # # HYPERPARAMETERS
    buckets = (1, 1, 6, 12)     
    # n_episodes = 1000           # Total train episodes
    # n_steps = 200               # Max steps per episode
    # min_alpha = 0.1             # learning rate
    # min_epsilon = 0.1           # exploration rate
    # gamma = 1                   # discount factor
    # ada_divisor = 25            # decay rate parameter for alpha and epsilon

    env = gym.make('CartPole-v0')
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    # define upper and lower bounds for each state value
    upper_bounds = [
        env.observation_space.high[0],
        0.5,
        env.observation_space.high[2],
        math.radians(50)
    ]
    lower_bounds = [
        env.observation_space.low[0],
        -0.5,
        env.observation_space.low[2],
        -math.radians(50)]

    # INITIALISE Q MATRIX
    Q = np.zeros(buckets + (n_actions,))
    print(np.shape(Q))

    def discretize(obs):
        ''' discretise the continuous state into buckets '''
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def epsilon_policy(state, epsilon):
        ''' choose an action using the epsilon policy '''
        exploration_exploitation_tradeoff = np.random.random()
        if exploration_exploitation_tradeoff <= epsilon:
            action = env.action_space.sample()  # exploration
        else:
            action = np.argmax(Q[state])  # exploitation
        return action

    def greedy_policy(state):
        ''' choose an action using the greedy policy '''
        return np.argmax(Q[state])

    def update_q(current_state, action, reward, new_state, alpha):
        ''' update the Q matrix with the Bellman equation '''
        Q[current_state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[current_state][action])

    def get_epsilon(t):
        ''' decrease the exploration rate at each episode '''
        if ada_divisor == 0:
            return min_epsilon  # epsilon fixo
        else:
            return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))  # adaptativo

    def get_alpha(t):
        ''' decrease the learning rate at each episode '''
        if ada_divisor == 0:
            return min_alpha  # alpha fixo
        else:
            return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))  # adaptativo

    # TRAINING PHASE
    df_results = pd.DataFrame(columns=['episode', 'alpha', 'epsilon', 'n_steps', 'episode_rewards'])

    for episode in range(n_episodes):
        current_state = env.reset()
        current_state = discretize(current_state)

        alpha = get_alpha(episode)
        epsilon = get_epsilon(episode)

        episode_rewards = 0

        for t in range(n_steps):
            # env.render()

            action = epsilon_policy(current_state, epsilon)
            new_state, reward, done, _ = env.step(action)
            # discreticacao do novo estado
            new_state = discretize(new_state)
            update_q(current_state, action, reward, new_state, alpha)
            current_state = new_state

            # increment the cumulative reward
            episode_rewards += reward

            # at the end of the episode
            if done:
                # print('Episode:{}/{} Total steps: {} Total reward: {}'.format(episode, n_episodes, t, episode_rewards))
                break

        # append the episode cumulative reward to the reward list
        df_results.loc[len(df_results)] = [episode, alpha, epsilon, t, episode_rewards]

    # TEST PHASE
    # current_state = env.reset()
    # current_state = discretize(current_state)
    # episode_rewards = 0

    # for t in range(n_steps):
    #     env.render()
    #     action = greedy_policy(current_state)
    #     new_state, reward, done, _ = env.step(action)
    #     new_state = discretize(new_state)
    #     update_q(current_state, action, reward, new_state, alpha)
    #     current_state = new_state
    #     episode_rewards += reward

    #     # at the end of the episode
    #     if done:
    #         #print('Test episode finished with a total reward of: {}'.format(episode_rewards))
    #         break

    # env.close()
    return df_results


######################################
# EXPERIMENTOS
######################################
def meanResults(df, n):
    df_len = len(df)
    count = 0
    dfs = []
    e = 1
    while True:
        if count > df_len - 1:
            break
        start = count
        count += n
        # print("%s : %s" % (start, count))
        x = df.iloc[start: count]
        dfs.append([e, x['episode_rewards'].mean()])
        e += 1
    df = pd.DataFrame(dfs)
    df.columns = ['episode', 'episode_rewards']
    # print(df)
    return df


def exp_q_learning(nr, buckets, n_episodes, batch, n_steps, alpha_epsilon, ada_divisor, gamma):
    print('INICIO qlearning ', datetime.datetime.now().time())
    start_time = time.time()
    # config grafico
    plt.style.use('seaborn')
    palette = plt.get_cmap('tab20')
    plt.figure(figsize=(8, 7), dpi=100)
    plt.xlabel('Episodios (x' + str(batch) + ')')
    plt.ylabel('Recompensas por episodio')
    df_results = pd.DataFrame()
    num = 0
    for (alpha, epsilon) in alpha_epsilon:
        num += 1
        name_exp = 'qlearning' + str(nr) + '_' + 'epis' + str(n_episodes) + '_batch' + str(batch) + '_alpha' + str(
            alpha) + '_epsi' + str(epsilon) + '_T' + str(int(time.time()))
        print(name_exp)
        df_results = Q_learning_train_test(buckets, n_episodes, n_steps, alpha, epsilon, gamma, ada_divisor)
        df_results = meanResults(df_results, batch)
        df_results.to_csv(str(num) + name_exp +'.csv', index=False)
        plt.title('Q-Learning: Curva de evolução de aprendizado', loc='center', fontsize=12,
                  fontweight=0)  # alpha: ' + str(alpha) + ' e epsilon: ' + epsilon
        x = df_results['episode']
        y = df_results['episode_rewards']
        #plt.plot(x, y, linewidth=2.5, dashes=[int(alpha * 20 + 3), 2], color=palette(num),
        #         label='a: ' + str(alpha) + ', e: ' + str(epsilon))
        plt.plot(x, y, linewidth=2.5, color=palette(num),
                 label='a: ' + str(alpha) + ', ep: ' + str(epsilon) )
    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.axvline(10, color='r', ls="dotted")
    plt.axhline(175, color='r', ls="dotted")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 4))
    plt.tight_layout()
    end_time = time.time()
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print('Tempo total: ', '%02d:%02d:%02d' % (h, m, s))
    print('FIM', datetime.datetime.now().time())
    plt.savefig(name_exp+'.png', dpi=100)
    plt.show()


if __name__ == "__main__":
    
    # name_exp = 'q-learning-teste basico'    
    # df_results = Q_learning_train_test(buckets = (1,1,6,12), n_episodes = 1000, n_steps = 200, min_alpha = 0.89, min_epsilon = 0.1, gamma = 1, ada_divisor = 0)
    
    # ''' q-learning '''
    # alpha_epsilon= [(0.0025,0.1),(0.5,0.1),(0.3,0.1),(0.9,0.9)]
    # exp_q_learning(nr=30, buckets=(1, 1, 6, 12), n_episodes=10000, batch=100, n_steps=200,  alpha_epsilon =  alpha_epsilon, ada_divisor=0, gamma=1)

    # alpha_epsilon= [(0.0025,0.1),(0.1,0.5),(0.3,0.1),(0.9,0.1)]
    # exp_q_learning(nr=35, buckets=(1, 1, 6, 12), n_episodes=10000, batch=100, n_steps=200,  alpha_epsilon =  alpha_epsilon, ada_divisor=0, gamma=1)

    alpha_epsilon= [(0.9, 0.9)]
    exp_q_learning(nr=35, buckets=(1, 1, 6, 12), n_episodes=10000, batch=100, n_steps=200,  alpha_epsilon =  alpha_epsilon, ada_divisor=0, gamma=1)


    # exp_q_learning(nr=31, buckets=(1, 1, 6, 12), n_episodes=10000, batch=100, n_steps=200, alpha_epsilon =  alpha_epsilon, ada_divisor=0, gamma=1)

    # exp_q_learning(nr=32, buckets=(1, 1, 6, 12), n_episodes=10000, batch=100, n_steps=200, alpha_epsilon =  [(0.1,0.1)], ada_divisor=25, gamma=1)
    # exp_q_learning(nr=32, buckets=(1, 1, 6, 12), n_episodes=10000, batch=100, n_steps=200, alpha_epsilon = [(0.1,0.1)], ada_divisor=25, gamma=0.8)
