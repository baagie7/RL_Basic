# Suntton p.92 First-Visit MC predictions, for estimating V ~ v_pi
# card 조합에 따른 State Value estimate

import gym 
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

#state : (player card sum, dealer open card, usable_ace 보유) ex) (6, 1, False) 
stick_threshold = 19
win_cnt = 0
lose_cnt = 0
draw_cnt = 0
num_episodes = 100000
GAMMA = 1  # no discount
#policy pi to be evaluated

env = gym.make("Blackjack-v0")

# Input: a policy pi to be evaluated
def pi(state):
    # player card 가 stick_threshold 이상이면 무조건 stick
    # else 이면 hit 하는 전략
    return 0 if state[0] >= stick_threshold else 1  #0:stick, 1:hit

# Initialize
V = defaultdict(float)
Returns = defaultdict(list)

for _ in range(num_episodes):
    #Generate an episode following pi: S0,A0,R1,S1,A1,R2,..ST-1,AT-1,RT
    episode = []
    s = env.reset()
    while True:
        a = pi(s)  # 정책 pi 를 따름
        s_, r, done, _ = env.step(a)
        # s_ : (sum_hand(player), dealer open card, usable_ace 보유)
        episode.append((s, a, r))
        if done:
            if r == 1:
                win_cnt += 1
            elif r == -1:
                lose_cnt += 1
            else:
                draw_cnt += 1
            break
        s = s_
    #G <- 0
    G = 0
    #Loop for each step of spisode: t=T-1,T-2,...,0
    for s, a, r in episode[::-1]:
        # G <- gamma*G + R_(t+1)
        G = GAMMA * G + r  
        visited_states = []
        #Unless S_t appears in S_0, S_1,...S_(t-1):
            #  Append G to Returns(S_t)
            #  V(S_t) <- average(Returns(S_t))
        if s not in visited_states:
            Returns[s].append(G)
            V[s] = np.mean(Returns[s])
            visited_states.append(s)     

print('stick threshold = {}'.format(stick_threshold))
print("win ratio = {:.2f}%".format(win_cnt/num_episodes*100))
print("lose ratio = {:.2f}%".format(lose_cnt/num_episodes*100))
print("draw ratio = {:.2f}%".format(draw_cnt/num_episodes*100))
    
sample_state = (21, 3, True)
print("state {}의 가치 = {:.2f}".format(sample_state, V[sample_state]))
sample_state = (4, 1, False)
print("state {}의 가치 = {:.2f}".format(sample_state, V[sample_state]))
    
#시각화
X, Y = np.meshgrid(
    np.arange(1, 11),    # dealer가 open 한 card
    np.arange(12, 22))   # player가 가진 card 합계
         
#V[(sum_hand(player), dealer open card, usable_ace 보유)]
no_usable_ace = np.apply_along_axis(lambda idx: V[(idx[1], idx[0], False)], 
                                    2, np.dstack([X, Y]))
usable_ace    = np.apply_along_axis(lambda idx: V[(idx[1], idx[0], True)], 
                                    2, np.dstack([X, Y]))
    
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3), 
                               subplot_kw={'projection': '3d'})

ax1.plot_surface(X, Y, usable_ace, cmap=plt.cm.YlGnBu_r)
ax1.set_xlabel('Dealer open Cards')
ax1.set_ylabel('Player Cards')
ax1.set_zlabel('MC Estimated Value')
ax1.set_title('Useable Ace')

ax0.plot_surface(X, Y, no_usable_ace, cmap=plt.cm.YlGnBu_r)
ax0.set_xlabel('Dealer open Cards')
ax0.set_ylabel('Player Cards')
ax0.set_zlabel('MC Estimated Value')
ax0.set_title('No Useable Ace')
    
plt.show()
