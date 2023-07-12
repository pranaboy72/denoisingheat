import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

from .sac import SAC
from ..utils.replay_buffer import ReplayBuffer
from ..utils.rl_utils import eval_mode


class Trainer(nn.Module):
    def __init__(self, env, renderer, writer, args):
        super().__init__()
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.env = env
        self.renderer = renderer
        self.writer = writer
        self.args = args
        
        # Save Models
        self.model_path = args['log_path'] + args['model_path']
        self.best_reward = 0
        
        self.agent = SAC(env, device, args)
        self.buffer = ReplayBuffer(env.action_space.shape, device, args)
        
        
    def forward(self, obs, timestep, step):       
        if step < self.args['init_steps']:
            return self.env.action_space.sample()
        else:
            with eval_mode(self.agent):
                return self.agent.sample_action(obs, timestep)

            
    def update_params(self, step):
        num_updates = 1
        for _ in range(num_updates):
            self.agent.update(self.buffer, self.writer, step)
        
            
    def evaluate(self, num_episodes, step):
        all_ep_rewards = []
        
        def run_eval_loop(sample_stochastically=True):
            for _ in range(num_episodes):
                self.env.reset()
                obs = self.renderer.renders(self.args)
                
                done = False
                episode_reward = 0
                episode_step = 0
                while not done:
                    with eval_mode(self.agent):
                        if sample_stochastically:
                            action = self.agent.sample_action(obs, episode_step)
                        else:
                            action = self.agent.select_action(obs, episode_step)
                            
                    _, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    episode_step += 1
                    
                all_ep_rewards.append(episode_reward)
                
            mean_ep_reward = np.mean(all_ep_rewards)
            best_ep_reward = np.max(all_ep_rewards)
            
            self.writer.add_scalar('eval/mean reward', mean_ep_reward, step)
            self.writer.add_scalar('eval/best reward', best_ep_reward, step)
            
            if mean_ep_reward > self.best_reward:
                torch.save(self.agent.actor.state_dict(), self.model_path)
            
        run_eval_loop(sample_stochastically=False)    
                                    
            
        