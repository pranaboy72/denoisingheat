import gym
import d4rl

env = gym.make('maze2d-umaze-v1')
env.reset()
env.step(env.action_space.sample())
env.render()