import os
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from scorefield.models.trainer import Trainer
from scorefield.utils.rl_utils import load_config, set_seed_everywhere
from scorefield.utils.rendering import Maze2dRenderer
from scorefield.utils.utils import log_num_check, visualize


def main():    
    # Args
    sac_config_dir = "/home/junwoo/scorefield/scorefield/configs/sac_args.yaml"
    args = load_config(sac_config_dir)

    set_seed_everywhere(args['seed'])

    # Env & Renderer
    renderer = Maze2dRenderer(args['env_name'], args['target_point'])
    env = renderer.env

    # Writer
    log_path = args['log_path'] + args['env_name']
    log_path = log_num_check(log_path)
    writer = SummaryWriter(log_path)

    # Trainer
    trainer = Trainer(env, renderer, writer, args)


    # Training Loop
    episode, episode_reward, done = 0, 0, True

    for step in range(args['num_train_steps']):
        
        # evaluate agent periodically
        if step % args['eval_freq'] == 0 and step > 0:
            trainer.evaluate(args['num_eval_episodes'], step)
            
        if done:
            env.reset(seed=args['seed'])
            obs = renderer.renders(args)
            # visualize(obs)
            
            if step % args['log_interval'] == 0:
                writer.add_scalar('train/reward', episode_reward, step)
            
            if step > 0:
                print(f'Episode: {episode}, total numsteps: {step}, episode steps: {episode_step},\
                    reward: {episode_reward}')
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            
        # sample action for data collection
        action = trainer(obs, episode_step, step)
        
        if step >= args['init_steps']:
            trainer.update_params(step)
            
        _, reward, done, _ = env.step(action)
        next_obs = renderer.renders(args)
        
        # allow infinite bootstrap
        done_bool = 0 if episode_step +1 == env.max_episode_steps else float(done)
        episode_reward += reward
        trainer.buffer.add(obs, action, reward, next_obs, done_bool, episode_step)
        
        obs = next_obs
        episode_step += 1
        

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()