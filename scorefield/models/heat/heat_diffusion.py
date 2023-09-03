import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from scorefield.utils.utils import clip_batch_vectors
from scipy.ndimage import distance_transform_edt
import math


class HeatDiffusion(object):
    def __init__(self, image_size, u0=1., noise_steps=500, heat_steps=1000, alpha=1.0, beta=10.0, time_type='linear',device='cuda'):
        self.image_size = image_size
        self.u0 = u0
        self.noise_steps = noise_steps
        self.heat_steps = heat_steps
        self.alpha = alpha
        self.beta = beta
        self.time_type = time_type
        self.dt = 1 / (4 * alpha)   # For stability, CFL condition: dt <= (dx^2 * dy*2) / (2*alpha* (dx^2 + dy^2))
        
        self.device = device

        self.diffusion_steps = torch.arange(1, self.noise_steps+1, device=device)
        self.heat_steps = self.convert_timespace(self.diffusion_steps)

        self.std = self.heat_steps / 2      

    def convert_space(self, previous, converted):
        """
            Convert from [-1,1] space to [0,127] space
        """
        if converted == 'pixel':
            return ((previous + 1) * 0.5 * (self.image_size - 1)).long()
        elif converted == 'norm':
            return (previous / (self.image_size - 1) * 2 - 1)
        
    def convert_timespace(self, time_steps):
        if self.time_type == 'linear':
            return time_steps * int((self.heat_steps / self.noise_steps))
        elif self.time_type == 'exp':
            norm_steps = (torch.exp(self.diffusion_steps) - torch.exp(self.diffusion_steps[0])) / (torch.exp(self.diffusion_steps[-1]) - torch.exp(self.diffusion_steps[0]))
            return (norm_steps * (self.heat_steps - 1) + 1).to(torch.int64)
        else:
            raise "Wrong time type"


    def compute_K(self, u):
        obstacle_with_edges = self.obstacle_masks.clone()
        
        dk = torch.stack([torch.tensor(distance_transform_edt(mask.cpu().numpy())) for mask in ~obstacle_with_edges])
        dk = dk.to(u.device).float()
        decay = torch.exp(-dk/self.beta)
        K = self.alpha * torch.ones_like(u)      
        
        # K = K * (1 - decay)
        K[obstacle_with_edges] = 0
        return K
    

    def gaussian_filter(self, ut, t, kernel_size=3):
        input_tensor = ut.clone()
        if kernel_size % 2 == 0:
            kernel_size += 1

        B = input_tensor.shape[0]
        result = []

        for i in range(B):
            ti = torch.tensor(1) #t[i]

            x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size, dtype=input_tensor.dtype, device=input_tensor.device)
            kernel = torch.exp(-x**2 / (2*(torch.sqrt(ti))**2))
            kernel /= kernel.sum()

            gaussian_filter = torch.outer(kernel, kernel).unsqueeze(0).unsqueeze(0)

            padding = kernel_size // 2
            input_tensor_padded = F.pad(input_tensor[i].unsqueeze(0).unsqueeze(1), (padding, padding, padding, padding), mode='reflect')
            smoothed_tensor = F.conv2d(input_tensor_padded, gaussian_filter, stride=1, padding=0)

            result.append(smoothed_tensor.squeeze(1))
            
        return torch.stack(result, dim=0).squeeze(1)
    
    def create_obstacle_masks(self, batch_size, obstacle_masks):
        if obstacle_masks is None:
            obstacle_masks = torch.zeros((batch_size, self.image_size, self.image_size), dtype=torch.bool, device=self.device)

        kernel = torch.ones((1,1,3,3), dtype=torch.bool, device=self.device)
        obstacle_masks = F.conv2d(obstacle_masks.float().unsqueeze(1), kernel.float(), padding=1).squeeze(1) > 0

        obstacle_masks[:, :, :2] = 1
        obstacle_masks[:, :, -2:] = 1
        obstacle_masks[:, :2, :] = 1
        obstacle_masks[:, -2:, :] = 1

        self.obstacle_masks = obstacle_masks
    
    def exclude_insulators(self, u):
        masked = u * (1 - self.obstacle_masks.to(u.dtype))
        return masked
        
    def compute_ut(self, time_steps, sample_num, heat_sources):
        heat_sources = self.convert_space(heat_sources, 'pixel')

        u = torch.zeros((sample_num * len(time_steps), self.image_size, self.image_size), dtype=torch.float32, device=heat_sources.device)

        K = self.compute_K(u)

        for b in range(u.shape[0]):
            u[b, heat_sources[b,...,0], heat_sources[b,...,1]] = self.u0
            
        max_time_steps = torch.max(time_steps).item()
        time_steps = time_steps.repeat_interleave(sample_num)

        for t in range(1, max_time_steps+1):
            current_mask = (t < time_steps)
            laplacian = torch.zeros_like(u).to(torch.float32)

            top = u[:, :-2, 1:-1]
            bottom = u[:, 2:, 1:-1]
            left = u[:, 1:-1, :-2]
            right = u[:, 1:-1, 2:]

            laplacian[:, 1:-1, 1:-1] = top + bottom + left + right - 4 * u[:, 1:-1, 1:-1]

            u[current_mask] += self.dt * (K[current_mask] * laplacian[current_mask])

        u = self.gaussian_filter(u, time_steps)
        sums = u.sum(dim=(1,2), keepdim=True)
        norm_u = u / sums
        return norm_u
    
    def sample_from_heat(self, u):
        ut = u.clone()

        # ut = self.exclude_insulators(ut)
        ut_flat = ut.view(ut.size(0),-1)
        sampled_indices = torch.multinomial(ut_flat, 1).squeeze()

        h = sampled_indices // u.size(2)
        w = sampled_indices % u.size(2)

        return torch.stack((h, w), dim=1)
    
    def score(self, ut, x=None):
        u = ut.clone()
        eps = 1e-9
        u_log = torch.log(torch.clamp(u, min=eps)).unsqueeze(1)

        kernel_w = torch.tensor([[-0.5, 0, 0.5]], dtype=ut.dtype, device=ut.device).unsqueeze(0).unsqueeze(0)
        kernel_h = torch.tensor([[-0.5], [0], [0.5]], dtype=ut.dtype, device=ut.device).unsqueeze(0).unsqueeze(0)

        grad_w = F.conv2d(u_log, kernel_w, padding=(0, 1)).squeeze(1)
        grad_h = F.conv2d(u_log, kernel_h, padding=(1, 0)).squeeze(1)

        score_field = torch.stack([grad_h, grad_w], dim=-1)
        score_field[self.obstacle_masks] = 0

        # non-dimensionalize
        magnitudes = torch.norm(score_field, dim=-1, keepdim=True)
        max_mag = magnitudes.view(score_field.shape[0],-1).max(dim=1,keepdim=True)[0].view(score_field.shape[0], 1, 1, 1)
        nondim_scorefield = score_field / max_mag

        if x is not None:
            B, _ = x.shape
            batch_indices = torch.arange(B, device=x.device)
            h_indices = x[:, 0].long()
            w_indices = x[:, 1].long()

            scores = nondim_scorefield[batch_indices, h_indices, w_indices, :]

            return scores
#         score_field = clip_batch_vectors(score_field, 0.01)   # for visualization
        return nondim_scorefield


    def forward_diffusion(self, time_steps, heat_sources, sample_num, obstacle_masks=None):
        self.create_obstacle_masks(len(time_steps) * sample_num, obstacle_masks)
        time_steps = self.heat_steps[time_steps-1]
        ut_batch = self.compute_ut(time_steps, sample_num, heat_sources)
        x_t = self.sample_from_heat(ut_batch)
        return ut_batch, self.score(ut_batch, x_t), self.score(ut_batch), self.convert_space(x_t, 'norm')
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps+1, size=(n,)).long()
    