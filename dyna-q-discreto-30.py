# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:20:02 2020
https://github.com/MJeremy2017/Reinforcement-Learning-Implementation/blob/master/DynaMaze/DynaMaze.py
@author: vcardenas.local
"""

import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time

# ROWS = 6
# COLS = 9
# S = (2, 0)
# G = (0, 8)
# BLOCKS = [(1, 2), (2, 2), (3, 2), (0, 7), (1, 7), (2, 7), (4, 5)]
# ACTIONS = ["left", "up", "right", "down"]
# ------------------------------------------------------------------------------
# 5. Models
# ------------------------------------------------------------------------------

# Implementation of a Table Lookup Model as showed by David Silver in
# COMPM050/COMPGI13 Lecture 8, slide 15
class TableLookupModel:
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
         #Q =    np.zeros(buckets + (n_actions,))
        self.N = np.zeros((nS, nA)) # Keep track of the number of times (s,a)
                                    # has appeared
        self.SprimeCounter = np.zeros((nS, nA, nS)) # Number of times (s,a)
                                                    # resulted in s'
        self.Rcounter = np.zeros((nS, nA)) # Total reward obtained by (s,a)
        self.observedStates = [] # states that have appeared before
        self.observedActions = [[] for i in range(nS)] # actions observed before
                                                       # at every state
        self.terminalStates = [] # No knowledge about terminal states assumed

    # Experience is considered as a tuple of (state, action, reward, state_prime)
    def addExperience(self, experience):
        s, a, r, s_prime = experience
        self.N[s][a] += 1
        self.SprimeCounter[s][a][s_prime] += 1
        self.Rcounter[s][a] += r
        if not s in self.observedStates: self.observedStates.append(s)
        if not a in self.observedActions[s]: self.observedActions[s].append(a)

    # Samples the resulting state of (s,a)
    def sampleStatePrime(self, state, action):
        # If there is no information about (s,a), then sample randomly
        if self.N[state][action] == 0: return np.random.choice(range(self.nS))

        prob = self.SprimeCounter[state][action] / self.N[state][action]
        return np.random.choice(range(self.nS), p = prob)

    # Samples the resulting reward of (s,a)
    def sampleReward(self, state, action):
        # If there is no information about (s,a), then return a fixed reward
        if self.N[state][action] == 0: return 0

        return self.Rcounter[state][action] / self.N[state][action]

    # Sample a random state that has been observed before
    def sampleRandState(self):
        return np.random.choice(self.observedStates)

    # Sample a random action previously observed in a given state
    def sampleRandAction(self, state):
        return np.random.choice(self.observedActions[state])

    # Give model knowledge about terminal states
    def addTerminalStates(self, term_states):
        self.terminalStates = term_states

    # Check wether a state is terminal (assuming model has knowledge about
    # terminal states)
    def isTerminal(self, state):
        return state in self.terminalStates
#######################


def DynaAgentPlay(buckets,  # define o numero de buckets para cada estado valor (x, x', theta, theta')
                n_episodes,  # numero de episodios
                n_steps,  # numero de maximo de steps por episodio
                min_alpha,  # taxa de aprendizado
                min_epsilon,  # taxa de explotacao
                gamma,  # fator de desconto
                ada_divisor):  # taxa de decaimento para os parametro de alpha e epsilon
    buckets = (1,1,6,12)    
    env = gym.make('CartPole-v0')
    n_actions = env.action_space.n # nro de acoes
    n_states = env.observation_space.shape[0]# nro de estados
    state_actions = []             # armazena estado acao
    
    # inicialize Q(s,a) e Modelo(s,a) para todo estado s e acao a
    Q = np.zeros(buckets + (n_actions,))
    model = {} # Initialize model
    N = {} #Keep track of the number of times (s,a) has appeared
    SprimeCounter = {} # Number of times (s,a) resulted in s'
    Rcounter = {} #Total reward obtained by (s,a)

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
        
    def update_q(current_state, action, reward, new_state, alpha):
        ''' update the Q matrix with the Bellman equation '''
        Q[current_state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[current_state][action])
        
    # Experience is considered as a tuple of (state, action, reward, state_prime)
    def addExperience(current_state, action, reward, new_state):
        if current_state not in model.keys():
            model[current_state] = {}
            N[current_state] = {action:0}
            SprimeCounter[current_state] = {action:{new_state:0}}
            Rcounter[current_state] = {action:0}
        if action not in N[current_state].keys():
            N[current_state] = {action:0}
            SprimeCounter[current_state] = {action:{new_state:0}}
            Rcounter[current_state] = {action:0}
        if new_state not in SprimeCounter[current_state][action].keys():
            SprimeCounter[current_state][action] = {new_state:0}
        
        model[current_state][action] = (reward, new_state) # self.model[self.state][action] = (reward, new_State) ex (0, 0, 2, 6): {1: (1.0, (0, 0, 2, 4)), 0: (1.0, (0, 0, 2, 8))}
        N[current_state][action] += 1
        #print(N)
        SprimeCounter[current_state][action][new_state] += 1
        Rcounter[current_state][action] += reward
        #if not s in self.observedStates: self.observedStates.append(s)
        #if not a in self.observedActions[s]: self.observedActions[s].append(a)


    # Samples the resulting reward of (s,a)
    def sampleReward(state, action):
        # If there is no information about (s,a), then return a fixed reward
        if action not in N[state].keys() or N[state][action] == 0: 
            return 0
        return Rcounter[state][action] / N[state][action]
    
    # Samples the resulting state of (s,a)
    def sampleStatePrime(state, action):
        # If there is no information about (s,a), then sample randomly  
        reward, new_State = model[state][action]
        return new_State
        # prob = SprimeCounter[state][action][new_State] / N[state][action]
        # return np.random.choice([new_State,], p = [prob,1-prob])
        # #
        
     

    
    # #randomState =  #[np.random.uniform(low=-2.4, high=2.4),np.random.uniform(low=-1000000, high=1000000),np.random.uniform(low=-2.4, high=2.4), np.random.uniform(low=-1000000, high=1000000)]
        # #randomState = discretize(randomState)     
        # if N[state][action] == 0: 
        #     return new_State #np.random.choice(range(nS))        
        
        # prob = SprimeCounter[state][action][new_State] / N[state][action]
        
        # if prob > 0.5:
        #     return new_State
        # else:
        #     return model[state][action]

    
    
    
    #n_steps_per_episode = []  
    df_results = pd.DataFrame(columns=['episode', 'alpha', 'epsilon', 'n_steps', 'episode_rewards'])
    
    
    for episode in range(n_episodes):        
        #print('dyna ', episode)
        current_state = env.reset()
        current_state = discretize(current_state)       
        alpha = get_alpha(episode)
        epsilon = get_epsilon(episode)
        
        episode_rewards = 0
        state_actions = []        
        # repetir em quanto o pole ainda esteja em pe
        for t in range(n_steps): # self.maze.end:
            # 1. escolha uma estado e uma acao real
            action = epsilon_policy(current_state, epsilon)          
            state_actions.append((current_state, action))
            
            # 2. observe uma recompensa resultante a experiencia real
            new_state, reward, done, _ = env.step(action) #nxtState = self.maze.nxtPosition(action)
            #reward = self.maze.giveReward()
            # discreticacao do novo estado
            new_state = discretize(new_state)
            
            # 3. atualice o valor de Q-value com Q-learning            
            update_q(current_state, action, reward, new_state, alpha)
            #self.Q_values[self.state][action] += self.alpha*(reward + np.max(list(self.Q_values[nxtState].values())) - self.Q_values[self.state][action])

            # increment the cumulative reward
            episode_rewards += reward
            
            # 4. atualice o modelo do ambiente com esta experiencia real
            # if current_state not in model.keys():
            #     model[current_state] = {}
            # model[current_state][action] = (reward, new_state)
            # current_state = new_state #self.state = nxtState
            addExperience(current_state, action, reward, new_state)
            
            current_state = new_state #self.state = nxtState
            ###### DYNA  com simulacao
            # 5. repita n vezes para atualizar o Q-valor aleatoriamente
            #if (t > 2000):
            for temp in range(5):# n_steps):
                # escolha um estado hipotetic entre os estadosobservados
                rand_idx = np.random.choice(range(len(model.keys())))
                _state = list(model)[rand_idx] # lista das keys do modelo key=(s)
                
                # escolha uma acao hipotetica entre as acoes observadas
                rand_idx = np.random.choice(range(len(model[_state].keys()))) # retorna as acoes registradas em model para esse state
                _action = list(model[_state])[rand_idx]
                
                # simule recompensa e seguinte estado resultante com o modelo do ambiente!!!
                #_reward, _new_state = model[_state][_action]
                _reward = sampleReward(_state, _action)
                _new_state = sampleStatePrime(_state, _action)
            
                # aplique aprendizado por reforço a esta experiencia hipotética
                update_q(_state, _action, _reward, _new_state, alpha)
                
                #Q_values[_state][_action] += self.alpha*(_reward + np.max(list(self.Q_values[_nxtState].values())) - self.Q_values[_state][_action])       
    
            # end of game
            if done:
                #print('Episode:{}/{} Total steps: {} Total reward: {}'.format(episode, n_episodes, t, episode_rewards))
                break
            #print('Episode:{}/{} Total steps: {} Total reward: {}'.format(episode, n_episodes, t, episode_rewards))
        # append the episode cumulative reward to the reward list
        df_results.loc[len(df_results)] = [episode, alpha, epsilon, t, episode_rewards]
        
        #n_steps_per_episode.append(len(state_actions))      
        #self.reset()
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

def combAlphaEp(alphas, epsilons):
    alpha_epsilon = []
    for alpha in alphas:
        for epsilon in epsilons:
            alpha_epsilon.append((alpha,epsilon))
    return alpha_epsilon

def exp_dyna_q(nr, buckets, n_episodes, batch, n_steps, alpha_epsilon, ada_divisor, gamma):
    print('INICIO dyna ', datetime.datetime.now().time())
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
        name_exp = 'dynaq' + str(nr) + '_' + 'epis' + str(n_episodes) + '_batch' + str(batch) + '_alpha' + str(
            alpha) + '_epsi' + str(epsilon) + '_T' + str(int(time.time()))
        print(name_exp)
        df_results = DynaAgentPlay(buckets, n_episodes, n_steps, alpha, epsilon, gamma, ada_divisor)
        df_results = meanResults(df_results, batch)
        df_results.to_csv(str(num)+'dynaaaa df_results'+name_exp+'.csv', index=False)
        #plt.savefig('results/graph_'+name_exp+'.png', dpi=100)
        plt.title('Dyna-Q: Curva de evolução de aprendizado', loc='center', fontsize=12,
                  fontweight=0)  # alpha: ' + str(alpha) + ' e epsilon: ' + epsilon
        x = df_results['episode']
        y = df_results['episode_rewards']
        #plt.plot(x, y, linewidth=2.5, dashes=[int(alpha * 20 + 3), 2], color=palette(num),
        #         label='a: ' + str(alpha) + ', e: ' + str(epsilon))
        plt.plot(x, y, linewidth=2.5, color=palette(num),
                 label='a: ' + str(alpha) + ', ep: ' + str(epsilon))
    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    #plt.legend(loc='lower right', mode="expand", borderaxespad=0.)
    plt.axvline(10, color='r', ls="dotted")
    plt.axhline(175, color='r', ls="dotted")
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 4))
    plt.tight_layout()
    end_time = time.time()
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print('Tempo total: ', '%02d:%02d:%02d' % (h, m, s))
    print('FIM', datetime.datetime.now().time())
    plt.savefig('graph_dyna.png', dpi=100)
    plt.show()



if __name__ == "__main__":
    # comparison
    # alpha_epsilon = combAlphaEp(alphas=[0.25, 0.5, 0.9],  epsilons=[0.001, 0.5, 0.9])
    # exp_dyna_q(nr=12, buckets=(1, 1, 6, 12), n_episodes=10000, batch=100, n_steps=300, alpha_epsilon=alpha_epsilon, ada_divisor=0, gamma=1)
    
    # alpha_epsilon = combAlphaEp(alphas=[0.25, 0.3, 0.6, 0.8],  epsilons=[0.0001, 0.3, 0.6, 0.8])
    # exp_dyna_q(nr=12, buckets=(1, 1, 6, 12), n_episodes=10000, batch=100, n_steps=300, alpha_epsilon=alpha_epsilon, ada_divisor=0, gamma=1)
    
    # alpha_epsilon = combAlphaEp(alphas=[0.1, 0.1, 0.1, 0.1],  epsilons=[0.1, 0.1, 0.1, 0.1, 0.1])
    # exp_dyna_q(nr=12, buckets=(1, 1, 6, 12), n_episodes=10000, batch=100, n_steps=300,alpha_epsilon=alpha_epsilon, ada_divisor=25, gamma=1)
    
    alpha_epsilon= [(0.3,0.1)]
    exp_dyna_q(nr=30, buckets=(1, 1, 6, 12), n_episodes=10000, batch=100, n_steps=200,alpha_epsilon=alpha_epsilon, ada_divisor=0, gamma=1)

    # alpha_epsilon = combAlphaEp(alphas=[0.9],  epsilons=[0.1])
    # exp_dyna_q(nr=31, buckets=(5, 5, 6, 12), n_episodes=10000, batch=100, n_steps=200,alpha_epsilon=alpha_epsilon, ada_divisor=0, gamma=1)

    # alpha_epsilon = combAlphaEp(alphas=[0.3, 0.0025],  epsilons=[0.1, 0.5 ,0.9])
    # exp_dyna_q(nr=32, buckets=(1, 1, 2, 2), n_episodes=10000, batch=100, n_steps=200,alpha_epsilon=alpha_epsilon, ada_divisor=0, gamma=1)