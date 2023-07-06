import utils
import argparse

import gym
import d4rl

from temporal import TemporalUnet

parser = argparse.ArgumentParser(description='Env Args')
## Env
parser.add_argument('--env-name', default='maze2d-umaze-v1')

## Model
parser.add_argument('--horizon', default=128)
parser.add_argument('--n_diffusion_steps', default=64)
parser.add_argument('--action_weight', default=1)
parser.add_argument('--loss_weights', default=None)
parser.add_argument('--loss_discount', default=1)
parser.add_argument('--predict_epsilon', default=False)
parser.add_argument('--dim_mults', default=(1, 4, 8))

## training
parser.add_argument('n_steps_per_epoch', default=10000)
parser.add_argument('loss_type',default='l2')
p

args = parser.parse_args()

# Environment 
env = gym.make('maze2d-umaze-v1')
                                            # maze2d-umaze-v1
observation_dim = env.observation_space  # (-inf,inf,(4,),float64)
action_dim = env.action_space            # (-1.0,1.0,(2,),float32)

# Model
diffuser = TemporalUnet(
    args.horizon,
       
)
