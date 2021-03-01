# Q-Learning (off-policy TD control) for estimating pi=pi*
"""
option 1 : FrozenLake-v0 (is_slippery : True) 를 이용하여 
           non-deterministic(stochastic) world 구현
env.observation_space.n : 16
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
env.action_space.n : 4
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
action 선택은 greedy, epsilon greedy, random noise 방식의 3 가지 비교
deterministic, stochastic 환경별로 ALPHA 값과 action 선택 policy 에 
따라 다른 결과 나오는 것을 비교

option 2 : 새로운 environment 를 등록하여 deterministic world 생성
"""
import sys
import gym 
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

OPTION = 2  # 1: FrozenLake-v0, 2: DeterministicFrozenLake-v0
#Algorithm patameters: step size alpha, small e>0
GAMMA = 0.99
epsilon = 0.3
num_episodes = 10000  # option 1 인 경우 더 많은 episode 필요

if OPTION == 1:
    env = gym.make('FrozenLake-v0')   # default는 is_slippery: True   
    ALPHA = 0.85
elif OPTION == 2:
    register(
        id="DeterministicFrozenLake-v0",
        entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4',
                'is_slippery': False}
    )
    env = gym.make('DeterministicFrozenLake-v0')
    ALPHA = 1.0 
else:
    print("Invalid Option")
    sys.exit(1)
    
reward_history = []
# Initialize Q(s,a) arbitrarily except that Q(terminal, .)=0
Q = np.zeros([env.observation_space.n, env.action_space.n])  #16x4

#env.render()

# Loop fpr eacj episode:
for _ in range(num_episodes):
    # Initialize S
    s = env.reset()  
      
    total_reward = 0
    # Loop for each step of episode:
    done = False
    while not done:       
        # Choose A from S using policy derived from Q (eg. e-greedy)
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else : 
            a = np.argmax(Q[s, :])

        # Ttake action A, observe R, S'
        s_, r, done, _ = env.step(a)  
        #env.render()
        Q[s, a] = Q[s, a] + ALPHA * (r + GAMMA * np.max(Q[s_,:]) - Q[s,a])
        total_reward += r
        # S <-- S'
        s = s_
        # until S is terminal

    reward_history.append(total_reward)

print("Option : {}".format("FrozenLake-v0" if OPTION ==
                           1 else "DeterministicFrozenLake-v0"))
print("GAMMA = {}, epsilon = {}, ALPHA = {}".format(GAMMA, epsilon, ALPHA))
print("Success rate : " + str(sum(reward_history) / num_episodes))
print("Final Q-Table Values")
print(Q)

plt.bar(range(len(reward_history)), reward_history, color="blue")
plt.title("{}: GAMMA{}, epsilon{}, ALPHA{}".format(
    "FrozenLake-v0" if OPTION==1 else "DeterministicFrozenLake-v0", GAMMA, epsilon, ALPHA))
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.show()
