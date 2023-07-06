import utils
import argparse

import gym
import d4rl

from temporal import TemporalUnet

parser = argparse.ArgumentParser(description='Env Args')
parser.add_argument('--env-name', default='maze2d-large-v1')

args = parser.parse_args()

# Environment 
env = gym.make('maze2d-umaze-v1')
                                            # maze2d-umaze-v1
observation_dim = env.observation_space[2]  # (-inf,inf,(4,),float64)
action_dim = env.action_space[2]            # (-1.0,1.0,(2,),float32)


# Model
diffuser = TemporalUnet(
    
    
)