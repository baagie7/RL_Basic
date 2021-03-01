#Tabular TD(0) for estimating V_pi
import numpy as np
import gym
from collections import defaultdict

env = gym.make('CartPole-v0')
#Algorithm parameter: step size alpha (0, 1]
alpha = 0.1
gamma = 0.99
num_episodes = 10000
#Tabular 형식으로 해결하기 위해 continuous state를 이산화
states = np.linspace(-0.2094, 0.2094, 10)
#Initialize V(s), arbitrarily except V(terminal)=0
V = defaultdict(float)

#Input: the policy pi to be evaluated
def pi(state):
    action = 0 if state < 5 else 1
    return action

#Loop for each episode:
for _ in range(num_episodes):
    #Initialize s
    observation = env.reset()
    s = np.digitize(observation[2], states)
    done = False
    #Loop for each step of episode
    while not done:
        #A <- action given by pi for s
        a = pi(s)
        #Take action A, observe R, S'
        observation_, r, done, _ = env.step(a)
        s_ = np.digitize(observation_[2], states)
        #V(s) <- V(s)+alpha[r+gamma*V(s')-V(s)]
        V[s] += alpha*(r + gamma * V[s_] - V[s])
        #S <- S
        s = s_
  
for s, v in sorted(V.items()):
    print("State {} Value = {:.2f}".format(s, v))
        
        
