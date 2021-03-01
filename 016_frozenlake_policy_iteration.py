#Policy Iteration
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

# 1. initialize an array V(s) = 0 for all s in S+
# and arbitrary pi(s) for all a in A+ for all s in S+
V = np.zeros(num_states)
pi = np.ones([env.nS, env.nA]) * 0.25

policy_stable = False

cnt = 0
 
while not policy_stable:
    cnt += 1
    
    while True:  # Iterative Policy Evaluation
        delta = 0
        for s in range(num_states):
            new_value = 0
            #update rule : V(s) = sum(pi(a|s)*sum(p(s,a)*[r + gamma*v(s')]))
            for a, prob_a in enumerate(pi[s]):
                # sum over s', r
                for prob, s_, r, _ in transitions[s][a]:
                    new_value += prob_a * prob * (r + GAMMA * V[s_])
            delta = max(delta, np.abs(new_value - V[s]))
            V[s] = new_value
        if delta < THETA:
            break

    #while True:   # Policy Improvement
    policy_stable = True
        
    for s in range(num_states):
        old_action = np.argmax(pi[s])
        # update rule : pi_s = argmax_a(sum(p(s',r|s,a)*[r + gamma*V(s')]))
        new_action_values = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, s_, r, _ in transitions[s][a]:
                new_action_values[a] += \
                        prob * (r + GAMMA * V[s_])
        new_action = np.argmax(new_action_values)
        if new_action != old_action:
            policy_stable = False
        pi[s] = np.eye(num_actions)[new_action]

print("Optimal Value = \n", V.reshape(4, 4))
print("Optimal Policy = \n", pi)
print("Optimal Action = \n", np.argmax(pi, axis=1).reshape(4, 4))
