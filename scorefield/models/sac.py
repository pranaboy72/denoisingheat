import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from scorefield.utils.rl_utils import soft_update_params
from .sac_models import Actor, Critic
import numpy as np


class SAC:
    def __init__(self, env, device,args):        
        self.device = device
        
        self.discount = args['discount']
        self.critic_tau = args['critic_tau']
        self.encoder_tau = args['encoder_tau']
        self.actor_update_freq = args['actor_update_freq']
        self.critic_target_update_freq = args['critic_target_update_freq']
        self.log_interval = args['log_interval']
        self.image_size = args['image_size']
        self.feature_dim = args['feature_dim']
        self.detach_encoder = args['detach_encoder']
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        obs_shape = (3, self.image_size, self.image_size)


        self.actor = Actor(
            obs_shape, env.action_space.shape[0], args['hidden_dim'],
            self.feature_dim, args['actor_log_std_min'], args['actor_log_std_max']
        ).to(device)
        
        self.critic = Critic(
            obs_shape, env.action_space.shape[0], args['hidden_dim'], self.feature_dim,
        ).to(device)
        
        self.critic_target = Critic(
            obs_shape, env.action_space.shape[0], args['hidden_dim'], self.feature_dim, 
        ).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        
        self.log_alpha = torch.tensor(np.log(args['init_temperature'])).to(device)
        self.log_alpha.requires_grad = True
        # set entropy to -|A|
        self.target_entropy = -np.prod(env.action_space.shape[0])
        
        # optimizers
        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=args['actor_lr'], betas=(args['actor_beta'], 0.999)
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(),lr=args['critic_lr'], betas=(args['critic_beta'], 0.999)
        )
        
        self.log_alpha_optimizer = Adam(
            [self.log_alpha], lr=args['alpha_lr'], betas=(args['alpha_beta'], 0.999)
        )
        
        self.encoder_optimizer = Adam(self.critic.encoder.parameters(), lr=args['encoder_lr'])
        
        self.train()        
            
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, obs, timestep):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, timestep, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()
        
    def sample_action(self, obs, timestep):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)            
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, timestep, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()
        
    def update_critic(self, obs, action, reward, next_obs, not_done, timestep):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs, timestep)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)
            
        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss
        
    def update_actor_and_alpha(self, obs, timestep):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, timestep, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)
        
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
        
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        return actor_loss, alpha_loss
        
    def update(self, replay_buffer, writer, step):
        obs, action, reward, next_obs, not_done, timestep = replay_buffer.sample()
        
        critic_loss = self.update_critic(obs, action, reward, next_obs, not_done, timestep)
        writer.add_scalar('loss/critic', critic_loss.item(), step)
        
        if step % self.actor_update_freq == 0:
            actor_loss, alpha_loss = self.update_actor_and_alpha(obs, timestep)
            
            writer.add_scalar('loss/actor', actor_loss.item(), step)
            writer.add_scalar('loss/entropy', alpha_loss.item(), step)
            
        if step % self.critic_target_update_freq == 0:
            soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)
            