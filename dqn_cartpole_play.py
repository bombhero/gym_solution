# -*- coding: utf-8 -*-
import gym
import os
import numpy as np
from dqn_cartpole import DQNAgent


def DQNplay():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    if os.path.exists("./save/cartpole-dqn.h5"):
        agent.load("./save/cartpole-dqn.h5")
    else:
        print("Can\'t find NN file.")
        return

    # No random action.
    agent.epsilon = 0
    for e in range(10):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = np.reshape(next_state, [1, state_size])
            if done:
                print("episode: %d/10, score: %d, e: %.2f" % (e, time, agent.epsilon))
                break


if __name__ == "__main__":
    DQNplay()
