import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime

from .encoder import ResNet18
from .sac import SAC
from scorefield.utils.replay_memory import ReplayMemory


class Trainer(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        
        self.env = env
        self.start_steps = args['start_steps']
        
        self.encoder = ResNet18()
        self.agent = SAC(env, args)
        self.memory = ReplayMemory(args['replay_size'], args['seed'])
        
        self.writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args['env_name'],
                                                             args['policy'], "autotune" if args['automatic_entropy_tuning'] else ""))

        
    def forward(self, img, t, evaluate=False):
        _, feature_map = self.encoder(img)
        temporal_feature = torch.cat([feature_map, t], dim=1)
        
        if t > self.start_steps:
            return self.agent.select_action(temporal_feature, evaluate)
        else:
            return self.env.action_space.sample()

            
    def update_params(self, updates, args):
        for i in range(args['updates_per_step']):
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(self.memory, args['batch_size'], updates)
            
            self.writer.add_scalar('loss/critic_1', critic_1_loss, updates)
            self.writer.add_scalar('loss/critic_2', critic_2_loss, updates)
            self.writer.add_scalar('loss/policy', policy_loss, updates)
            self.writer.add_scalar('loss/entropy_loss', ent_loss, updates)
            self.writer.add_scalar('entropy_temperature/alpha', alpha, updates)
            
        