# Q-Learning (off-policy TD control) for estimating pi=pi*
import gym
import matplotlib.pyplot as plt
import numpy as np
import time
import os
"""
6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    
state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
          5 * 5 * 5 * 4 = 500

Rewards:
    per-step : -1,
    delivering the passenger : +20,
    executing "pickup" and "drop-off" actions illegally : -10
"""
env = gym.make('Taxi-v3')
n_states = env.observation_space.n  # 500
n_actions = env.action_space.n      # 6

#Algorithm parameter: stepsize alpha (0,1], small e > 0
gamma = 0.99  # time decay
alpha = 0.9  # learning rate
epsilon = 0.7 # exploration start
epsilon_final = 0.1
epsilon_decay = 0.9999
#Initialize Q(s,a) for all s, a arbitrarily except Q(terminal,.)=0
Q = np.zeros([n_states, n_actions])

n_episodes = 1000
rendering = False
scores = []  # agent 가 episode 별로 얻은 score 기록
steps = []  # agent 가 spisode 별로 목표를 찾아간 step 수 변화 기록
greedy = [] # epsilon delay history 기록

#Loop for each episode:
for episode in range(n_episodes):
    #Initialize S
    s = env.reset()
    step = 0
    score = 0
    #Loop for each step of episode:
    done = False
    while not done:
        step += 1
        #Choose A from S using policy derived from Q(e.g.e-greedy)
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            random_values = Q[s] + np.random.rand(n_actions) / 1000
            a = np.argmax(random_values)          
        
        if epsilon > epsilon_final:
            epsilon *= epsilon_decay
        
        #Take action A, observe R, S'
        s_, r, done, _ = env.step(a)
        
        if rendering and (episode > n_episodes/2):
            os.system('cls')  # console clear
            env.render()
            time.sleep(0.5)
        
        #Q(S,A) <- Q(S,A) + alpha[R+gamma*max_aQ(S',a)-Q(S,A)]
        Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[s_] - Q[s, a]))
        
        #S <- S'
        s = s_ 
        score += r
        
    steps.append(step)
    scores.append(score)
    greedy.append(epsilon)
    
    if episode % 100 == 0:
        print("last 100 평균 score is %s" % np.mean(scores[-100:]))
    
print('평균 steps : {}'.format(np.mean(steps)))

plt.bar(np.arange(len(steps)), steps)
plt.title("Steps of Taxi-v3- gamma: {}, alpha: {}".format(
             gamma, alpha))
plt.xlabel('episode')
plt.ylabel('steps per episode')
plt.show()

plt.bar(np.arange(len(scores)), scores)
plt.title("Scores of Taxi-v3- gamma: {}, alpha: {}".format(
                    gamma, alpha))
plt.xlabel('episode')
plt.ylabel('score per episode')
plt.show()

plt.bar(np.arange(len(greedy)), greedy)
plt.title("Epsilon decay history - epsilon: {}, decay: {}".format(
                    epsilon, epsilon_decay))
plt.xlabel('episode')
plt.ylabel('epsilon per episode')
plt.show()

