import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

# LEFT - 0, DOWN - 1, RIGHT - 2, UP - 3

# SFFF       (S: starting point, safe)
# FHFH       (F: frozen surface, safe)
# FFFH       (H: hole, fall to your doom)
# HFFG       (G: goal, where the frisbee is located)

# Simple deterministic Policy
policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1,
          8: 2, 9: 1, 10: 1, 13: 2, 14: 2}

# new deterministic environment
register(id='FrozenLakeNoSlippery-v0',
         entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={'map_name': '4x4', 'is_slippery': False},
         )

#env = gym.make('FrozenLakeNoSlippery-v0')  # deterministic environment

env = gym.make('FrozenLake-v0')  # stochastic environment

n_games = 1000
win_pct = []
scores = []

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = policy[obs]  # deterministic
        obs, reward, done, info = env.step(action)
        print(info)
        score += reward
    scores.append(score)

    if i % 10:
        average = np.mean(scores[-10:])
        win_pct.append(average)

plt.plot(win_pct)
plt.xlabel('episode')
plt.ylabel('success ratio')
plt.show()
