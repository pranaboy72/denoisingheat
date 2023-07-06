import argparse

import gym
import d4rl
import torch

from temporal import TemporalUnet
from diffusion import GaussianDiffusion
from utils.training import Trainer
from utils.serialization import *
from utils.arrays import report_parameters, batchify
from utils.setup import watch
from datasets.sequence import GoalDataset
from datasets.normalization import LimitsNormalizer

parser = argparse.ArgumentParser(description='Env Args')
## Env
parser.add_argument('--env-name', default='maze2d-umaze-v1')

## Dataset
parser.add_argument('--termination_penalty',default=None)
parser.add_argument('--preprocess_fns',default=['maze2d_set_terminals'])
parser.add_argument('--clip_denoised', default=True)
parser.add_argument('--use_padding', default=False)
parser.add_argument('--max_path_length', default=40000)

## Serialization
parser.add_argument('--log_base', default='logs')
parser.add_argument('--prefix', default='diffusion/')
parser.add_argument('--exp_name', default=watch(diffusion_args_to_watch))

## Model
parser.add_argument('--horizon', default=128)
parser.add_argument('--n_diffusion_steps', default=64)
parser.add_argument('--action_weight', default=1)
parser.add_argument('--loss_weights', default=None)
parser.add_argument('--loss_discount', default=1)
parser.add_argument('--predict_epsilon', default=False)
parser.add_argument('--dim_mults', default=(1, 4, 8))

## training
parser.add_argument('--n_steps_per_epoch', default=10000)
parser.add_argument('--loss_type',default='l2')
parser.add_argument('--n_train_steps', default=2e6)
parser.add_argument('--batch_size', default=32)
parser.add_argument('--learning_rate', default=2e-4)
parser.add_argument('--gradient_accumulate_every', default=2)
parser.add_argument('--ema_decay', default=0.995)
parser.add_argument('--save_freq', default=1000)
parser.add_argument('--sample_freq', default=1000)
parser.add_argument('--n_saves', default=50)
parser.add_argument('--save_parallel', default=False)
parser.add_argument('--n_reference', default=50)
parser.add_argument('--n_samples', default=10)
parser.add_argument('--bucket', default=None)
parser.add_argument('--save_path', default='../logs')

args = parser.parse_args()

# Environment 
env = gym.make('maze2d-umaze-v1')
                                            # maze2d-umaze-v1
observation_dim = env.observation_space  # (-inf,inf,(4,),float64)
action_dim = env.action_space            # (-1.0,1.0,(2,),float32)
transition_dim = observation_dim + action_dim
cond_dim = observation_dim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


# Dataset
dataset = GoalDataset(
    env,
    args.horizon,
    LimitsNormalizer,
    args.preprocess_fns,
    args.max_path_length,
    args.max_n_episodes,
    termination_penalty=args.termination_penalty,
    use_padding=args.use_padding
)


# Model & Trainer
model = TemporalUnet(
    args.horizon,
    transition_dim,
    cond_dim,
    args.dim_mults,
    device,
)

diffusion = GaussianDiffusion(
    model,
    args.horizon,
    observation_dim,
    action_dim,
    args.n_diffusion_steps,
    args.loss_type,
    args.clip_noised,
    args.predict_epsilon,
)

# dataset =
# renderer = 

trainer = Trainer(
    diffusion,
    dataset=dataset,
    # renderer= ,
    ema_decay=args.ema_decay,
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps//args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.save_path,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)


# Test forward & backward pass
report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')

##############      Main Loop       ###############
n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.save_path}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)