#Value Iteration
"""
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

nA = 4
nS = 4*4 = 16
P = {s: {a: [] for a in range(nA)} for s in range(nS)}
env.P[0][0] 
{0: {0: [(0.3333333333333333, 0, 0.0, False), --> (P[s'], s', r, done)
         (0.3333333333333333, 0, 0.0, False),
         (0.3333333333333333, 4, 0.0, False)],
"""
import gym
import numpy as np

env = gym.make('FrozenLake-v0')
#env = gym.make('FrozenLake8x8-v0')

GAMMA = 1.0
THETA = 1e-5

num_states = env.nS
num_actions = env.nA
transitions = env.P 

V = np.zeros(num_states)

# Value Iteration
while True:
    delta = 0
    for s in range(num_states):
        old_value = V[s]
        new_action_values = np.zeros(num_actions)
        for a in range(num_actions):
            # sum over s', r
            for prob, s_, r, _ in transitions[s][a]:
                new_action_values[a] += prob * (r + GAMMA * V[s_])
        V[s] = np.max(new_action_values)
        
        delta = np.maximum(delta, np.abs(V[s] - old_value))
        
    if delta < THETA:
        break
    
# extract optimal policy using action value
pi = np.zeros((num_states, num_actions))

for s in range(num_states):
    e = np.zeros(env.env.nA)                       
                                        
    for a in range(env.env.nA):             
        q=0                               
        P = np.array(env.env.P[s][a])
        (x,y) = np.shape(P)
        
        for i in range(x):
            s_= int(P[i][1])
            p = P[i][0]
            r = P[i][2]
            q += p*(r+GAMMA*V[s_])
            e[a] = q
            
    m = np.argmax(e)
    pi[s][m] = 1

print("Optimal Value = \n", V.reshape(4, 4))
print("Optimal Policy = \n", pi)
print("Optimal Action = \n", np.argmax(pi, axis=1).reshape(4, 4))

