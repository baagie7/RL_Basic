import numpy as np
import gym
import time

class Agent:
    """
    observation(cart 위치, cart 속도, pole 각도, pole 각속도)에 가중치)
    agent의 parameter를 (np.random.rand(4) * 2 - 1)로 random초기화 하면 rand()가 
    [0, 1)이므로, [-1, 1)의 (4,) vector가 된다. observation의 shape도 (4,)이므로 
    matmul 은 음수 혹은 양수가 random 하게 생성.                                                                
    """
    def __init__(self):
        self.parameters = None # agent 가 policy 를 정하기 위한 parameter

    def policy(self, observations):  # agent 의 policy
        return 0 if np.matmul(self.parameters, observations) < 0 else 1

env = gym.make('CartPole-v0')
best_params = None
best_reward = 0
agent = Agent()

for episode in range(100):
    # random 하게 agent의 parameters 설정
    agent.parameters = np.random.rand(4) * 2 - 1
    observation = env.reset()
    total_reward = 0

    for i in range(1000):
        action = agent.policy(observation)
        env.render()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(f'episode {episode} 넘어짐 - after {i} step')
            break

    if total_reward > best_reward:

        best_reward = total_reward
        best_params = agent.parameters

        if total_reward >= 200:
            print(f'episode {episode} 에서 200 step 달성')
            break

    if episode+1 % 10 == 0:
        print(total_reward)

print(f"Best Parameter = {best_params}")    # optimal policy 출력
env.close()

