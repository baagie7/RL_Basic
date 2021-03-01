#Iterative Policy Evaluation
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

GAMMA = 1.0
THETA = 1e-5

policy = np.ones([env.nS, env.nA]) * 0.25

num_states = env.nS
num_actions = env.nA
transitions = env.P 

# initialize an array V(s) = 0 for all s in S+
V = np.zeros(num_states)


while True:
    delta = 0
    for s in range(num_states):
        new_value = 0
        #update rule : V(s) = sum(pi(a|s)*sum(p(s,a)*[r + gamma*v(s')]))
        for a, prob_action in enumerate(policy[s]):
            # sum over s', r
            for prob, s_, reward, _ in transitions[s][a]:
                new_value += prob_action * prob * (reward + GAMMA * V[s_])
        delta = max(delta, np.abs(new_value - V[s]))
        V[s] = new_value
    if delta < THETA:
        break

print("수렴한 Optimal Value = \n", V.reshape(4, 4))
