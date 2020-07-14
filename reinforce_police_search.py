#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:13:01 2020

@author: ar
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime
import pandas as pd
import time

######################################
# CRIAR E VISUALIZAR ENTORNO
######################################
# Create gym and seed numpy
env = gym.make('CartPole-v0')


######################################
# REINFORCE
######################################

def reinforce(env, n_episodes, alpha, gamma):
    n_actions = env.action_space.n
    np.random.seed(1)

    # Inicialice teta(w) arbitrariamente
    w = np.random.rand(4, 2)

    # Keep stats for fin_actionsl print of graph
    episode_rewards = []
    steps = []

    # Our policy that maps state to action parameterized by w
    def policy(state, w):
        z = state.dot(w)
        exp = np.exp(z)
        return exp / np.sum(exp)

    # Vectorized softmax Jacobian
    def softmax_grad(softmax):
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    # Repita para todo episodio
    for e in range(n_episodes):

        state = env.reset()[None, :]

        grads = []
        rewards = []

        n_steps = 0
        # guardar a recomensa por episodio
        score = 0

        while True:

            # Uncomment to see your model train in real time (slower)
            # env.render()

            # Sample from policy and take action in environment
            # i. calculo da recompensa acumulada descontada a partir do tempo t
            probs = policy(state, w)
            action = np.random.choice(n_actions, p=probs[0])
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[None, :]

            # Compute gradient and save with reward in memory for our weight updates
            # ii. calculo da gradiente da politica
            dsoftmax = softmax_grad(probs)[action, :]
            dlog = dsoftmax / probs[0, action]
            grad = state.T.dot(dlog[None, :])

            grads.append(grad)
            rewards.append(reward)

            score += reward
            # print('Steps: ', str(n_steps))
            n_steps += 1
            # Dont forget to update your old state to the new state
            state = next_state

            if done:
                break

        # Weight update
        for i in range(len(grads)):
            # Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
            w += alpha * grads[i] * sum([r * (gamma ** r) for t, r in enumerate(rewards[i:])])

        # Append for logging and print
        episode_rewards.append(score)
        steps.append(n_steps)
    # print("EP: " + str(e) + " Reward Accuml: " + str(score) + "         " + " n_steps: " + str(n_steps),end="\r", flush=False)
    return episode_rewards, steps


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


def exp_reinforce(nr, n_episodes, batch, alphas, ganmas):
    print('INICIO reinforce ', datetime.datetime.now().time())
    start_time = time.time()
    # config grafico
    plt.style.use('seaborn')
    palette = plt.get_cmap('tab20')
    plt.figure(figsize=(8, 7), dpi=100)
    plt.xlabel('Episodios (x' + str(batch) + ')')
    plt.ylabel('Recompensas por episodio')
    df_results = pd.DataFrame()
    num = 0
    for alpha in alphas:
        for ganma in ganmas:
            num += 1
            name_exp = 'reinforce'+ str(nr) +'_epis' + str(n_episodes) + '_batch' + str(batch) + '_alpha' + str(
                alpha) + '_ganma' + str(ganma) + '_T' + str(int(time.time()))
            print(name_exp)
            episode_rewards, steps = reinforce(env, n_episodes, alpha, ganma)
            df_results = pd.DataFrame({'episode': np.arange(n_episodes), 'episode_rewards': episode_rewards})
            df_results = meanResults(df_results, batch)
            df_results.to_csv(str(num) + name_exp + '_df_results.csv', index=False)
            plt.title('REINFORCE: Curva de evolução de aprendizado', loc='center', fontsize=12,
                      fontweight=0)
            x = df_results['episode']
            y = df_results['episode_rewards']
            #plt.plot(x, y, linewidth=2.5, dashes=[int(alpha * 20 + 3), 2], color=palette(num),
            #         label='a: ' + str(alpha) + ', e: ' + str(epsilon))
            plt.plot(x, y, linewidth=2.5, color=palette(num),
                     label='a: ' + str(alpha) + ', g: ' + str(ganma))
                     #label='a: ' + str(alpha) + ', ep: ' + str(epsilon))
    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.axvline(10, color='r', ls="dotted")
    plt.axhline(175, color='r', ls="dotted")
    plt.ticklabel_format(style='sci', axis='x')# , scilimits=(0, 4))
    plt.tight_layout()
    end_time = time.time()
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print('Tempo total: ', '%02d:%02d:%02d' % (h, m, s))
    print('FIM', datetime.datetime.now().time())
    plt.savefig(name_exp + str('%02d%02d%02d' % (h, m, s)) + '_graph.png', dpi=100)
    plt.show()


if __name__ == "__main__":
    exp_reinforce(nr=30, n_episodes=10000, batch=100,
                    alphas=[0.0025, 0.3, 0.5, 0.9],
                    ganmas=[1])
#alpha_epsilon= [(0.0025,0.1),(0.5,0.1),(0.3,0.1),(0.9,0.9)]
    # exp_reinforce(nr=30, n_episodes=10000, batch=100,
    #                 alphas=[0.00025, 0.3, 0.6, 0.8],
    #                 ganmas=[1])

    # exp_reinforce(nr=30, n_episodes=10000, batch=100,
    #                 alphas=[0.0025],
    #                 ganmas=[1])

# Hyperparameters
# n_episodes = 1000
# alpha = 0.0025
# gamma = 1#0.99
# episode_rewards,steps=reinforce(env, n_episodes, alpha, gamma)
# plt.plot(np.arange(n_episodes),episode_rewards)
# plt.plot(np.arange(n_episodes),steps)
# plt.savefig('reinforce ep '+str(n_episodes)+'-steps '+'-a '+'-g '+str(alpha)+str(gamma)+'.png', dpi=300)
# plt.show()
# env.close()
