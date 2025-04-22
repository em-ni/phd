# Example of using sofagym
import gym
import sofagym.envs
import numpy as np

env = gym.make("Pendulum-v1")
env.seed(42)
observation = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()

env.close()
