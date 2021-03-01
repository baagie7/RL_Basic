import numpy as np
import torch


class MultiArmedBandit:
    """
    multi-armed bandit (k-armed bandit)은 단일 state 로 구성
    k 가지의 action이 존재하고 각각의 reward 가 다름
    Agent는 e-greedy 방법에 따라 bandit으로부터 reward를 얻고 best arm 을 update.
    다음의 k-armed bandit을 구현
    """
    def __init__(self):
        self.p1 = [0.65, 0.35]
        self.p3 = [0.9, 0.1]

    def pull(self, arm):
        if arm == 0:
            return np.random.choice([10, 28], 1, p=self.p1)
        elif arm == 1:
            return np.random.choice(range(5, 14), 1)
        elif arm == 2:
            return np.random.choice([11, 87], 1, p=self.p3)


class Agent:
    def __init__(self, num_arms=3):
        self.num_arms = num_arms
        self.best_arm = np.argmax(np.random.rand(num_arms))
        print("initial best arm is {}".format(self.best_arm))

    def random_or_predict(self, epsilon):
        if np.random.rand() < epsilon:   # exploration
            return np.random.randint(self.num_arms)
        else:
            return self.best_arm   # exploit

env = MultiArmedBandit()
agent = Agent()
num_iter = [10, 100, 1000, 10000]
EPSILON = 0.3
num_arms = 3

for iters in num_iter:
    arm_rewards = np.zeros(num_arms)
    arm_selected = np.zeros(num_arms)

    for _ in range(iters):
        selected_arm = agent.random_or_predict(EPSILON)
        reward = env.pull(selected_arm)

        arm_rewards[selected_arm] += reward
        arm_selected[selected_arm] += 1

    agent.best_arm = np.argmax(arm_rewards / arm_selected)

    print()
    print("final best arm is {} when {}".format(agent.best_arm, iters))
    print("\tarm 선택 횟수 {}".format(arm_selected))
    for i, (rewards, selected) in enumerate(zip(arm_rewards, arm_selected)):
        if selected == 0:
            print("\t평균 rewards/{} no selection".format(i))
        else:
            print("\t평균 rewards/{} {:.2f}".format(i, (rewards / selected)))
