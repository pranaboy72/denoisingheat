import gym
import d4rl

env = gym.make('maze2d-open-v1')
env.reset()
try:
    while True:
        env.step(env.action_space.sample())
        env.render()
except:
    env.close()