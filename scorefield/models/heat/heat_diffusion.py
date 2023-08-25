import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


class HeatDiffusion(object):
    def __init__(self, image_size, u0=1, noise_steps=500, heat_steps=200000, alpha=2.0, precision='single', device='cuda'):
        self.image_size = image_size
        self.u0 = u0
        self.noise_steps = noise_steps
        self.heat_steps = heat_steps
        self.alpha = alpha
        self.dt = 1 / (4 * alpha)   # For stability, CFL condition: dt <= (dx^2 * dy*2) / (2*alpha* (dx^2 + dy^2))
        self.precision = torch.float64 if precision == 'double' else torch.float32
        self.device = device


    def convert_space(self, previous):
        """
            Convert from [-1,1] space to [0,127] space
        """
        return ((previous + 1) * 0.5 * (self.image_size - 1)).long()
    

    def compute_K(self, u):
        obstacle_with_edges = self.obstacle_masks.clone()
        
        K = self.alpha * torch.ones_like(u)
        
        K[obstacle_with_edges] = 0
        
        return K
    

    def gaussian_filter(self, input_tensor, kernel_size=5, std=1.0):
        if kernel_size %2 == 0:
            kernel_size += 1
            
        x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size, dtype=input_tensor.dtype, device=input_tensor.device)
        kernel = torch.exp(-x**2 / (2*std**2))
        kernel /= kernel.sum()
        
        gaussian_filter = torch.outer(kernel, kernel).unsqueeze(0).unsqueeze(0)
        gaussian_filter = gaussian_filter.expand((2, 1, kernel_size, kernel_size))
        
        padding = kernel_size // 2
        input_tensor_padded = F.pad(input_tensor.permute(0, 3, 1, 2), (padding, padding, padding, padding), mode='reflect')

        smoothed_tensor = F.conv2d(input_tensor_padded, gaussian_filter, stride=1, padding=0, groups=2)
        
        del gaussian_filter
        del input_tensor_padded
        torch.cuda.empty_cache()
        
        return smoothed_tensor.permute(0,2,3,1)
        
    
    def revise_obstacle_masks(self, batch_size, obstacle_masks):
        if obstacle_masks is None:
            obstacle_masks = torch.zeros((batch_size, self.image_size, self.image_size), dtype=torch.bool, device=heat_sources.device)
        
        obstacle_masks[:, :, 0] = 1
        obstacle_masks[:, :, -1] = 1
        obstacle_masks[:, 0, :] = 1
        obstacle_masks[:, -1, :] = 1
        
        self.obstacle_masks = obstacle_masks
        
    def compute_ut(self, time_steps, heat_sources):
        heat_sources = self.convert_space(heat_sources) # When [-0.7, 0.] => [19,63] (128x128)
        
        B = heat_sources.shape[0]
        u = torch.zeros((B, self.image_size, self.image_size), dtype=self.precision, device=heat_sources.device)

        K = self.compute_K(u)

        for b in range(B):
            u[b, heat_sources[b,0], heat_sources[b, 1]] = self.u0

        max_time_steps = torch.max(time_steps).item()
        
        for t in tqdm(range(1, max_time_steps+1)):
            current_mask = (t < time_steps)
            laplacian = torch.zeros_like(u).to(self.precision)

            top = u[:, :-2, 1:-1]
            bottom = u[:, 2:, 1:-1]
            left = u[:, 1:-1, :-2]
            right = u[:, 1:-1, 2:]
            
            top[self.obstacle_masks[:, :-2, 1:-1]] = 0
            bottom[self.obstacle_masks[:, 2:, 1:-1]] = 0
            left[self.obstacle_masks[:, 1:-1, :-2]] = 0
            right[self.obstacle_masks[:, 1:-1, 2:]] = 0
            
            top_mask = 1 - self.obstacle_masks[:, :-2, 1:-1].float()
            bottom_mask = 1 - self.obstacle_masks[:, 2:, 1:-1].float()
            left_mask = 1 - self.obstacle_masks[:, 1:-1, :-2].float()
            right_mask = 1 - self.obstacle_masks[:, 1:-1, 2:].float()
            
            valid_neighbors = top_mask + bottom_mask + left_mask + right_mask
            laplacian[:, 1:-1, 1:-1] = top + bottom + left + right - valid_neighbors * u[:, 1:-1, 1:-1]
            
            u[current_mask] += self.dt * (K[current_mask] * laplacian[current_mask])

        sums = u.sum(dim=(1,2), keepdim=True)
        norm_u = u / sums

        return norm_u


    def gradient_field(self, u):
        eps = 1e-30
        u_log = torch.log(torch.clamp(u, min=eps)).unsqueeze(1)

        kernel_w = torch.tensor([[-0.5, 0, 0.5]], dtype=self.precision, device=self.device).unsqueeze(0).unsqueeze(0)
        kernel_h = torch.tensor([[-0.5],[0],[0.5]], dtype=self.precision, device=self.device).unsqueeze(0).unsqueeze(0)

        grad_w = F.conv2d(u_log, kernel_w, padding=(0,1)).squeeze(1)
        grad_h = F.conv2d(u_log, kernel_h, padding=(1,0)).squeeze(1)

        grad_w[self.obstacle_masks] = 0
        grad_h[self.obstacle_masks] = 0
        
        score = torch.stack([grad_h, grad_w], dim=-1) 
        filtered_score = self.gaussian_filter(score)

        return filtered_score
    
    def forward_diffusion(self, time_steps, heat_sources, obstacle_masks=None):
        self.revise_obstacle_masks(heat_sources.shape[0], obstacle_masks)
        time_steps = time_steps * int((self.heat_steps / self.noise_steps))
        ut_batch = self.compute_ut(time_steps, heat_sources)
        return self.gradient_field(ut_batch), ut_batch
        return ut_batch
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).long()