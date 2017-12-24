# -*- coding: utf-8 -*-
import random
import gym
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000
recall_num = 3


class DQNAgent:
    def __init__(self, recall_num, state_size, action_size):
        self.recall_num = recall_num
        self.state_size = state_size
        self.scene_size = recall_num * state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(100, input_dim=self.scene_size, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, scene, action, reward, next_state, done):
        self.memory.append((scene, action, reward, next_state, done))

    def act(self, scene):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(scene)
        return np.argmax(act_values[0])  # returns action

    def gen_next_scene(self, current_scene, next_state):
        sceneX = np.reshape(current_scene, [self.recall_num, self.state_size])
        next_sceneX = np.concatenate((sceneX, next_state), axis=0)[1:, :]
        return np.reshape(next_sceneX, [1, self.scene_size])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        trainX = np.array([[]])
        trainY = np.array([[]])
        for scene, action, reward, next_state, done in minibatch:
            target = reward
            next_scene = self.gen_next_scene(scene, next_state)
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_scene)[0]))
            target_f = self.model.predict(scene)
            target_f[0][action] = target
            if trainX.size == 0:
                trainX = scene
                trainY = target_f
            else:
                trainX = np.concatenate((trainX, scene), axis=0)
                trainY = np.concatenate((trainY, target_f), axis=0)
        self.model.fit(trainX, trainY, epochs=batch_size, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        print("Load %s successful." % name)

    def save(self, name):
        self.model.save_weights(name)


class Scene:
    def __init__(self, recall_num, state_size):
        self.recall_num = recall_num
        self.state_size = state_size
        self.scene = np.zeros([recall_num, state_size], dtype="float64")

    def reset(self):
        self.scene = np.zeros([recall_num, state_size], dtype="float64")

    def add_state(self, state):
        s = np.reshape(state, [1, self.state_size])
        self.scene = np.concatenate((self.scene, s), axis=0)[1:, :]

    def output_scene(self):
        return np.reshape(self.scene, [1, (self.recall_num * self.state_size)])

    def get_scene_size(self):
        return self.recall_num * self.state_size


def main():
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("state = %d, action = %d." % (state_size, action_size))
    agent = DQNAgent(recall_num, state_size, action_size)
    if os.path.exists("./save/mcar-dqn.h5"):
        agent.load("./save/mcar-dqn.h5")
    done = False
    batch_size = 128

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        recent_scene = Scene(recall_num, state_size)
        recent_scene.add_state(state)
        for time in range(500):
            # env.render()
            position_list = np.ones([recall_num, 1]) * state[0, 0]
            action = agent.act(recent_scene.output_scene())
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_p, next_v = next_state[0, :]
            position_list = np.concatenate((position_list, np.array([[next_p]])), axis=0)[1:, :]
            r1 = abs(next_p - (-0.5)) * np.var(position_list, axis=0)[0] * 100
            r2 = abs(next_v)
            if next_p > 0.5:
                reward = 10
            else:
                reward = (r1 * 1) if not done else -10
            agent.remember(recent_scene.output_scene(), action, reward, next_state, done)
            recent_scene.add_state(next_state)
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 10 == 0:
            agent.save("./save/mcar-dqn.h5")


if __name__ == "__main__":
    for _ in range(2):
        main()
