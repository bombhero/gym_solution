# -*- coding: utf-8 -*-
import gym
import os
import numpy as np
from dqn_mcar import DQNAgent
from dqn_mcar import Scene
from dqn_mcar import recall_num


def DQNplay():
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(recall_num, state_size, action_size)
    if os.path.exists("./save/mcar-dqn.h5"):
        agent.load("./save/mcar-dqn.h5")
    else:
        print("Can\'t find NN file.")
        return

    # No random action.
    agent.epsilon = 0
    for e in range(10):
        current_scene = Scene(recall_num, state_size)
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        current_scene.add_state(state)
        for time in range(500):
            env.render()
            action = agent.act(current_scene.output_scene())
            next_state, reward, done, _ = env.step(action)
            state = np.reshape(next_state, [1, state_size])
            current_scene.add_state(state)
            if done:
                print("episode: %d/10, score: %d, e: %.2f" % (e, time, agent.epsilon))
                break


if __name__ == "__main__":
    DQNplay()
