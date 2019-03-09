import gym
import numpy as np
import torch
import random
import collections


class DriverAction:
    def __init__(self):
        self.action_option = np.array([[-0.5, 0, 0],
                                       [-0.5, 0.5, 0],
                                       [-0.5, 0, 0.5],
                                       [0.5, 0, 0],
                                       [0.5, 0.5, 0],
                                       [0.5, 0, 0.5],
                                       [0, 0.5, 0],
                                       [0, 0, 0.5]])
        self.action_n = self.action_option.shape[0]

    def get_action(self, action_id):
        return self.action_option[action_id, :]


class DriverAnalysis(torch.nn.Module):
    def __init__(self, action_n):
        super(DriverAnalysis, self).__init__()
        self.cnn_part = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=24,
                            kernel_size=3, stride=1),
            # Output (96-3+1) * (96-3+1) * 24 = 94 * 94 * 24
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            # Output 47 * 47 * 24

            torch.nn.Conv2d(in_channels=24, out_channels=48,
                            kernel_size=6, stride=1),
            # Output (47-6+1) * (47-6+1) * 48 = 42 * 42 *48
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3)
            # Output 14 * 14 * 48
        )
        self.output_part = torch.nn.Linear(in_features=(14 * 14 * 48), out_features=action_n)

    def forward(self, x):
        x = self.cnn_part(x)
        x = x.view(x.size(0), -1)
        return self.output_part(x)


class DriverAgent:
    def __init__(self, observation_space):
        self.action_option = DriverAction()
        self.observation_space = observation_space
        self.memory = collections.deque()
        self._build_model()

    def _build_model(self):
        self.nn_model = DriverAnalysis(self.action_option.action_n)

    def act(self, state):
        action_id = random.randint(0, (self.action_option.action_n - 1))
        return self.action_option.get_action(action_id)

    def record(self, state, reward):
        pass


def run():
    env = gym.make("CarRacing-v0")
    action_range = []
    for i in range(env.action_space.high.shape[0]):
        action_range.append([env.action_space.high[i], env.action_space.low[i]])

    state = env.reset()
    driver = DriverAgent(env.observation_space)
    total = 0
    for e in range(1000):
        env.render()
        action = driver.act(state)
        state, reward, done, _ = env.step(action)
        total += reward
        print("e = %d, gas = %f, reward = %f, done = %d, total = %f" % (e, action[1], reward, done, total))
        if done:
            break
        # time.sleep(0.1)


if __name__ == "__main__":
    run()
