#Sarsa(on-policy TD control) algorithm for estimating q*
import gym 
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
sys.path.append('./lib')
import lib

#Algorithm parameters:step size ALPHA(0, 1], small e>0
GAMMA = 1.0
ALPHA = 0.5
EPSILON = 0.1
NUM_EPISODES = 300

#env = gym.make('GridWorld-v0')
env = gym.make('WindyGridWorld-v0')

#Initialize Q(s, a) for all s, a arbitrarily
# except that Q(terminal, .)=0
num_actions = env.action_space.n
Q = defaultdict(lambda: np.zeros(num_actions))

episode_lengths = []
#Loop for each episode:
for _ in range(NUM_EPISODES):
    #Initialize S
    s = env.reset()
    length_current_episode = 1
    # Choose A from S using policy dreived from Q (e.g.e-greedy)
    p = np.random.random()
    if p < EPSILON:
        a = np.random.choice(num_actions)
    else:
        a = np.argmax(Q[s])
    #Loop for each step of episodes:
    done = False
    while not done:
        # take action A, observe R, S'
        s_, r, done, _ = env.step(a)
        length_current_episode += 1
        # Choose A' from S' using policy dreived from Q (e.g.e-greedy)
        p = np.random.random()
        if p < EPSILON:
            a_ = np.random.choice(num_actions)
        else:
            a_ = np.argmax(Q[s_])
        # Q(S, A) <- Q(S, A) + alpha[R+gammaQ(S',A')-Q(S,A)]
        Q[s][a] += ALPHA * (r + GAMMA * Q[s_][a_] - Q[s][a])
        #S <- S', A <- A'
        a= a_
        s = s_
        #until S is terminal
        
    episode_lengths.append(length_current_episode)
    
plt.style.use('grayscale')
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(episode_lengths, label='Episode Lengths')
ax.grid(True)
ax.legend(loc='right')
ax.set_title('Number of actions before termination')
ax.set_xlabel('Episode')
ax.set_ylabel('Number of Actions')

plt.show()
