import gym
import gym_gomoku
import time
env = gym.make('Gomoku19x19-v0') # default 'beginner' level opponent policy

env.reset()
env.render("human")
# place a single stone, black color first
env.step(15)

# play a game
env.reset()
for _ in range(20):
    # sample without replacement
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    time.sleep(1)
    env.render("human")
    if done:
        print("Game is Over")
        break
