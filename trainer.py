import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam



from tqdm.auto import tqdm

from scorefield.models.ddpm.version import __version__
from scorefield.utils.utils import random_batch


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

class DiffusionTrainer(object):
    def __init__(
        self,
        diffusion_model,
        renderer,
        map_img,
        batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './logs/results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
    ):
        super().__init__()
        
        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        
        # sampling and training hyperparamters
        
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        
        self.train_num_steps = train_num_steps
        
        # make a batch of random goal states in the map
        self.batch = torch.tensor(random_batch(renderer, map_img, self.batch_size))
        
        # optimizer
        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        
        # for logging results in a folder periodically
        
        self.results_folder = Path(results_folder)   
        self.results_folder.mkdir(exist_ok=True)
        
        # step counter state
        self.step = 0
        
        
    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            
            while self.step < self.train_num_steps:
                
                total_loss = 0.
                
                for _ in range(self.gradient_accumulate_every):
                    data = self.batch
                    
                    loss = self.model(data)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()
                    
                    loss.backward()
                    
                
