import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, action_shape, device,args,transform=None):
        self.capacity = args['replay_buffer_capacity']
        self.batch_size = args['batch_size']
        self.device = device
        self.obs_shape = (3, args['image_size'], args['image_size'])  #(C,H,W)
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.uint8
        
        self.obses = np.empty((self.capacity, *self.obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((self.capacity, *self.obs_shape), dtype=obs_dtype)
        self.actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.timesteps = np.empty((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
    

    def add(self, obs, action, reward, next_obs, done, timestep):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.timesteps[self.idx], timestep)
        
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        timesteps = torch.as_tensor(self.timesteps[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones, timesteps

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]
        timestep = self.timesteps[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done, timestep

    def __len__(self):
        return self.capacity 