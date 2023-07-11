import os
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import itertools
import argparse

import scorefield
from scorefield.models.trainer import Trainer
from scorefield.utils.rl_utils import load_config
from scorefield.utils.rendering import Maze2dRenderer


# Args
sac_config_dir = "/home/junwoo/scorefield/scorefield/configs/sac_args.yaml"
args = load_config(sac_config_dir)

# Env & Renderer
renderer = Maze2dRenderer(args['env_name'])
env = renderer.env
env.seed(args['seed'])
env.action_space.seed(args['seed'])

torch.manual_seed(args['seed'])
np.random.seed(args['seed'])


# Trainer
trainer = Trainer(env, args)


# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    env.reset()
    
    obs = renderer.renders()
    
    while not done:
        action = trainer(obs, total_numsteps)
        
        if len(trainer.memory) > args['batch_size']:
            trainer.update_params(updates)
            updates += 1
        
        _, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        
        next_obs = renderer.renders()
        
        trainer.memory.push(obs, reward, next_obs, mask)
        
        obs = next_obs
        
    if total_numsteps > args['num_steps']:
        break
    
    trainer.writer.add_scalar('reward/train', episode_reward, i_episode)
    print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps},\
        reward: {round(episode_reward, 2)}")
    
    if i_episode % 10 == 0 and args['eval'] is True:
        avg_reward = 0.
        episodes = 10
        episode_steps = 0
        
        for _ in range(episodes):
            env.reset()
            episode_reward = 0
            done = False
            
            obs = renderer.renders()
            
            while not done:
                action = trainer(obs, episode_steps, evaluate=True)
                _, reward, done, _ = env.step(action)
                episode_reward += reward
                
                obs = renderer.renders()
                
            avg_reward += episode_reward
        avg_reward /= episodes
        
        trainer.writer.add_scalar('avg_reward/test', avg_reward, i_episode)
        
        print("----------------------------------------")
        print(f"Test Episodes: {episodes}, Avg. Reward: {round(avg_reward,2)}")
        print("----------------------------------------")
        
env.close()
