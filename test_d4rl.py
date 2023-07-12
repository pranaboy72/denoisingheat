import gym
import d4rl

env = gym.make('maze2d-open-v1')
env.reset()
try:
    while True:
        ob, _, _, _ = env.step(env.action_space.sample())
        print(ob)
        env.render()
except:
    env.close()