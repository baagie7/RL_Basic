import gym
env = gym.make('CartPole-v0')
env.reset()

for _ in range(500):
    env.render()
    # take a random action
    env.step(env.action_space.sample())

env.close()
