import gym
import numpy as np
import matplotlib.pyplot as plt

# LEFT - 0, DOWN - 1, RIGHT - 2, UP - 3

# SFFF       (S: starting point, safe)
# FHFH       (F: frozen surface, safe)
# FFFH       (H: hole, fall to your doom)
# HFFG       (G: goal, where the frisbee is located)

# No Policy - stochastic
env = gym.make('FrozenLake-v0')
n_games = 1000
win_pct = []
scores = []

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0
    
    while not done:
      action = env.action_space.sample()
      obs, reward, done, info = env.step(action)
      score += reward
      
      env.render()
      
    scores.append(score)
    
    if i % 10:
      average = np.mean(scores[-10:])
      win_pct.append(average)
    
plt.plot(win_pct)
plt.xlabel('episode')
plt.ylabel('success ratio')
plt.show()