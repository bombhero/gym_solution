import gym
import numpy as np
import torch
import random
import collections
import os
import matplotlib.pyplot as plt


class DriverAction:
    def __init__(self):
        self.action_option = np.array([[-0.5, 0, 0],
                                       # [-0.5, 0.5, 0],
                                       # [-0.5, 0, 0.5],
                                       [0.5, 0, 0],
                                       # [0.5, 0.5, 0],
                                       # [0.5, 0, 0.5],
                                       [0, 0.5, 0],
                                       [0, 0, 0.5],
                                       [0, 0, 0]])
        self.action_n = self.action_option.shape[0]

    def get_action(self, action_id):
        return self.action_option[action_id, :]


class DriverAnalysis(torch.nn.Module):
    def __init__(self, action_n):
        super(DriverAnalysis, self).__init__()
        self.first_cnn = torch.nn.Conv2d(in_channels=3, out_channels=24,
                                         kernel_size=5, stride=1)
        self.cnn_part = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=24,
                            kernel_size=5, stride=1, padding=0),
            # Output (96-5+1) * (96-5+1) * 24 = 92 * 92 * 24
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            # Output 46 * 46 * 24

            torch.nn.Conv2d(in_channels=24, out_channels=48,
                            kernel_size=5, stride=1, padding=0),
            # Output (46-5+1) * (46-5+1) * 48 = 42 * 42 *48
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3)
            # Output 14 * 14 * 48
        )
        self.output_part = torch.nn.Linear(in_features=(14 * 14 * 48), out_features=action_n)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.expand([1, x.shape[0], x.shape[1], x.shape[2]])
        x = x.permute(0, 3, 1, 2)
        x = self.cnn_part(x)
        x = x.view(x.size(0), -1)
        return self.output_part(x)


class DriverAgent:
    def __init__(self, observation_space, action_n, training=True):
        self.action_option = DriverAction()
        self.observation_space = observation_space
        self.action_n = action_n
        self.memory = collections.deque()
        if training:
            self.random_min = 0.1
            self.random_threshold = 1
        else:
            self.random_min = 0
            self.random_threshold = 0
        self.random_decay = 0.99
        self.gamma = 0.95
        self.model_file = "save/racer.pkl"
        if os.path.exists(self.model_file):
            self.nn_model = torch.load(self.model_file)
        else:
            self.nn_model = self._build_model()
        print(self.nn_model)

    def _build_model(self):
        return DriverAnalysis(self.action_option.action_n)

    def act(self, state):
        random_value = random.random()
        if random_value < self.random_threshold:
            action_id = random.randint(0, (self.action_n - 1))
            if self.random_threshold > self.random_min:
                self.random_threshold = self.random_threshold * self.random_decay
        else:
            action_id = self.recommend(state)
        return action_id

    def remember(self, state, reward, action_id, next_state):
        self.memory.append((state, reward, action_id, next_state))

    def replay(self, max_batch_size):
        # if self.random_min == 0:
        #     return
        if len(self.memory) > max_batch_size:
            batch_size = max_batch_size
        else:
            batch_size = len(self.memory)
        min_batch = random.sample(self.memory, batch_size)
        scenarios = None
        targets = None
        for (state, reward, action_id, next_state) in min_batch:
            target_f = self.predict(state)
            target_f[0][action_id] = reward + self.gamma * np.amax(self.predict(next_state)[0])
            if scenarios is None:
                scenarios = np.array([state])
                targets = target_f
            else:
                scenarios = np.concatenate((scenarios, np.array([state])), axis=0)
                targets = np.concatenate((targets, target_f), axis=0)
        self.training(state, target_f)

    def predict(self, state):
        self.nn_model.eval()
        tensor_x = torch.from_numpy(np.float32(state))
        result = self.nn_model(tensor_x)
        return result.data.numpy()

    def recommend(self, state):
        result = self.predict(state)
        action_id = np.argmax(result, axis=1)[0]
        print("predict %d" % action_id)
        return action_id

    def training(self, x, y):
        self.nn_model.train()
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.01)
        tensor_x = torch.autograd.Variable(torch.from_numpy(np.float32(x)))
        tensor_y = torch.autograd.Variable(torch.from_numpy(np.float32(y)))

        for _ in range(10):
            optimizer.zero_grad()
            prediction = self.nn_model(tensor_x)
            loss = loss_func(prediction, tensor_y)
            loss.backward()
            optimizer.step()

    def load(self):
        self.nn_model = torch.load(self.model_file)

    def save(self):
        torch.save(self.nn_model, self.model_file)


def run():
    env = gym.make("CarRacing-v0")
    action_range = []
    for i in range(env.action_space.high.shape[0]):
        action_range.append([env.action_space.high[i], env.action_space.low[i]])

    action_option = DriverAction()
    remember_state = np.zeros([96, 96, 3])
    for r in range(100):
        if r % 10 == 0:
            driver = DriverAgent(env.observation_space, action_option.action_n, training=False)
        else:
            driver = DriverAgent(env.observation_space, action_option.action_n, training=True)
        state = env.reset()
        total = 0
        if driver.random_min > 0:
            driver.random_threshold = 1
        for e in range(1000):
            env.render()
            if e < 50:
                action_id = 2
            else:
                action_id = driver.act(state)
            gray_state = 0.299 * state[:, :, 0] + 0.587 * state[:, :, 1] + 0.114 * state[:, :, 2]
            remember_state[:, :, 2] = remember_state[:, :, 1]
            remember_state[:, :, 1] = remember_state[:, :, 0]
            remember_state[:, :, 0] = gray_state
            # plt.title(e)
            # plt.imshow(gray_state)
            # plt.pause(0.1)
            next_state, reward, done, _ = env.step(action_option.get_action(action_id))
            if reward < 0:
                reward *= 3
            if done or (total < -1):
                reward = -10
            if e >= 50:
                driver.remember(remember_state, reward, action_id, next_state)
            if done or (total < -1):
                break
            if (e > 64) and (e % 10 == 0):
                driver.replay(128)
                driver.replay(256)
            if e % 100 == 0:
                driver.save()
            total += reward
            print("e = %d, r= %.1f, g= %.1f, b= %.1f, reward = %.2f, done = %d, total = %f" % (e,
                                                                            action_option.get_action(action_id)[0],
                                                                            action_option.get_action(action_id)[1],
                                                                            action_option.get_action(action_id)[2],
                                                                            reward, done, total))
            state = next_state
            # time.sleep(0.1)
        driver.save()


if __name__ == "__main__":
    run()
