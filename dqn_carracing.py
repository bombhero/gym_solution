import gym
import numpy
import torch
import random
import time


def main():
    env = gym.make("CarRacing-v0")
    action_range = []
    for i in range(env.action_space.high.shape[0]):
        action_range.append([env.action_space.high[i], env.action_space.low[i]])

    state = env.reset()
    total = 0
    for e in range(10000):
        env.render(mode="state_pixels")
        action = numpy.zeros([3])
        action[0] = random.random() * 2 - 1
        action[1] = random.random()
        action[2] = random.random()
        state, reward, done, _ = env.step(action)
        total += reward
        print("e = %d, reward = %f, done = %d, total = %f" % (e, reward, done, total))
        if done:
            break
        # time.sleep(0.1)


if __name__ == "__main__":
    main()
