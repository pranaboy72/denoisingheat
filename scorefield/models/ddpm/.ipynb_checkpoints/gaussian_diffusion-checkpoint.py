import os
import torch
from tqdm import tqdm


class Diffusion(object):
    def __init__(self, input_size, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.device = device
        self.input_size = input_size
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def forward_diffusion(self, x0, t):
        """
            x_t = sqrt(alpha_hat) * x_0 + sqrt(1-alpha_hat) * epsilon
        """
        noise = torch.randn_like(x0)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).unsqueeze(-1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t]).unsqueeze(-1)
        
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).long()
    
    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 2)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t]
                alpha_hat = self.alpha_hat[t]
                beta = self.beta[t]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) \
                    * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        
        return x

    def sample_onestep(self, model, obs, x, t):
        model.eval()
        with torch.no_grad():
            predicted_noise = model(obs, x, t)
            alpha = self.alpha[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.beta[t]
            if t > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) \
                    * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        
        return x