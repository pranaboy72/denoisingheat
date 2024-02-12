import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from denoisingheat.utils.utils import clip_batch_vectors
from scipy.ndimage import distance_transform_edt
import math


class GaussHeatDiffusion(object):
    def __init__(self, image_size, u0=1., noise_steps=500, min_heat_step=1, max_heat_step=1000, alpha=1.0, beta=10.0, time_type='linear',device='cuda'):
        self.image_size = image_size
        self.u0 = u0
        self.noise_steps = noise_steps
        
        lower_limit = int(math.sqrt(min_heat_step / 2))
        upper_limit = int(math.sqrt(max_heat_step / 2))
        assert lower_limit**2 * 2 == min_heat_step, "The value of min step is not valid. It should be 2*n**2" 
        assert upper_limit**2 * 2 == max_heat_step, "The value of max step is not valid. It should be 2*n**2" 
        
        self.min_heat_step = min_heat_step
        self.max_heat_step = max_heat_step
        self.alpha = alpha
        self.beta = beta
        self.time_type = time_type
        self.dt = 1 / (4 * alpha)   # For stability, CFL condition: dt <= (dx^2 * dy*2) / (2*alpha* (dx^2 + dy^2))
        
        self.device = device

        diffusion_steps = torch.arange(1, noise_steps+1, device=device)
        self.heat_steps = self.convert_timespace(diffusion_steps).to(device)
        print(f'dt:{self.heat_steps}')
        self.std = torch.sqrt(self.heat_steps / 2)
        # self.std[4:] = self.std[3]
        self.std = torch.clamp(self.std, max=image_size//2)
        print(f'heat kernel std:{self.std}')

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
            return time_steps * int((self.max_heat_step / self.noise_steps))
        elif self.time_type == 'exp':
            factor = torch.tensor(2.)
            log_space = torch.linspace(0, 1, steps=self.noise_steps)
            range_adjusted = (int(math.sqrt(self.max_heat_step/2)) - int(math.sqrt(self.min_heat_step/2)))
        
            exp_sequence = int(math.sqrt(self.min_heat_step/2)) + range_adjusted * (torch.exp(factor * log_space) -1) / (torch.exp(factor) - 1)
            exp_sequence = torch.round(exp_sequence)
            exp_sequence[0] = int(math.sqrt(self.min_heat_step/2))
            exp_sequence[-1] = int(math.sqrt(self.max_heat_step/2))
            return (2 * exp_sequence**2).to(torch.int64)
        else:
            raise "Wrong time type"
    
    def compute_K(self, u):
        obstacle_with_edges = self.obstacle_masks.clone()

        K = self.alpha * torch.ones_like(u)      
        
        K[obstacle_with_edges] = 0
        return K

    def gaussian_filter(self, ut, diffusion_steps, kernel_size=3):
        input_tensor = ut.clone()
        if kernel_size % 2 == 0:
            kernel_size += 1

        B = input_tensor.shape[0]
        result = []

        for i in range(B):
            std = self.std[diffusion_steps[i]-1] #t[i]

            x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size, dtype=input_tensor.dtype, device=input_tensor.device)
            kernel = torch.exp(-x**2 / (2*std**2))
            kernel /= kernel.sum()

            gaussian_filter = torch.outer(kernel, kernel).unsqueeze(0).unsqueeze(0)

            padding = kernel_size // 2
            input_tensor_padded = F.pad(input_tensor[i].unsqueeze(0).unsqueeze(1), (padding, padding, padding, padding), mode='reflect')
            smoothed_tensor = F.conv2d(input_tensor_padded, gaussian_filter, stride=1, padding=0)

            result.append(smoothed_tensor.squeeze(1))
            
        return torch.stack(result, dim=0).squeeze(1)
    
    
    def create_obstacle_masks(self, batch_size):
        obstacle_masks = torch.zeros((batch_size, self.image_size, self.image_size), dtype=torch.bool, device=self.device)
        
        obstacle_masks[:, :, :1] = 1
        obstacle_masks[:, :, -1:] = 1
        obstacle_masks[:, :1, :] = 1
        obstacle_masks[:, -1:, :] = 1

        self.obstacle_masks = obstacle_masks
    
    def exclude_insulators(self, u):
        masked = u * (1 - self.obstacle_masks.to(u.dtype))
        return masked
        
    def compute_ut(self, heat_dts, diffusion_steps, heat_sources):
        heat_sources = self.convert_space(heat_sources, 'pixel')

        u = torch.zeros((len(heat_dts), self.image_size, self.image_size), dtype=torch.float32, device=heat_sources.device)
        
        K = self.compute_K(u)
        
        for b in range(heat_sources.shape[0]):
            for g in range(heat_sources.shape[1]):
                u[b, heat_sources[b,g,0], heat_sources[b,g,1]] = self.u0
            
        max_time_steps = torch.max(heat_dts).item()

        for t in range(1, max_time_steps+1):
            current_mask = (t < heat_dts)
            laplacian = torch.zeros_like(u).to(torch.float32)

            top = u[:, :-2, 1:-1]
            bottom = u[:, 2:, 1:-1]
            left = u[:, 1:-1, :-2]
            right = u[:, 1:-1, 2:]

            top_mask = self.obstacle_masks[:, :-2, 1:-1]
            bottom_mask = self.obstacle_masks[:, 2:, 1:-1]
            left_mask = self.obstacle_masks[:, 1:-1, :-2]
            right_mask = self.obstacle_masks[:, 1:-1, 2:]

            count_non_obstacle = 4 - (top_mask.float() + bottom_mask.float() + left_mask.float() + right_mask.float())

            laplacian[:, 1:-1, 1:-1] = top + bottom + left + right - count_non_obstacle * u[:, 1:-1, 1:-1]

            u[current_mask] += self.dt * (K[current_mask] * laplacian[current_mask])

        u_filtered = self.gaussian_filter(u, diffusion_steps)
        return u_filtered
    
    
    def sample_from_heat(self, u, n):
        ut = u.clone()

        ut = self.exclude_insulators(ut)    # exclude samples in insulators by numerical error
        ut_flat = ut.view(ut.size(0),-1)
        sampled_indices = torch.multinomial(ut_flat, n, replacement=True)

        h = sampled_indices // u.size(2)
        w = sampled_indices % u.size(2)

        return torch.stack((h, w), dim=-1)


    def forward_diffusion(self, diffusion_steps, heat_sources, sample_num):
        self.create_obstacle_masks(len(diffusion_steps))
        heat_dts = self.heat_steps[diffusion_steps-1]
        ut_batch = self.compute_ut(heat_dts, diffusion_steps, heat_sources)
        x_t = self.sample_from_heat(ut_batch, sample_num)
        return ut_batch, self.convert_space(x_t, 'norm')
    
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps+1, size=(n,), device=self.device).long()
    
    
class HeatDiffusion_Revised(object):
    def __init__(self, image_size, u0=1., noise_steps=500, min_heat_step=1, max_heat_step=1000, alpha=1.0, beta=10.0, time_type='linear',device='cuda'):
        self.image_size = image_size
        self.u0 = u0
        self.noise_steps = noise_steps
        
        lower_limit = int(math.sqrt(min_heat_step / 2))
        upper_limit = int(math.sqrt(max_heat_step / 2))
        assert lower_limit**2 * 2 == min_heat_step, "The value of min step is not valid. It should be 2*n**2" 
        assert upper_limit**2 * 2 == max_heat_step, "The value of max step is not valid. It should be 2*n**2" 
        
        self.min_heat_step = min_heat_step
        self.max_heat_step = max_heat_step
        self.alpha = alpha
        self.beta = beta
        self.time_type = time_type
        self.dt = 1 / (4 * alpha)   # For stability, CFL condition: dt <= (dx^2 * dy*2) / (2*alpha* (dx^2 + dy^2))
        
        self.device = device

        diffusion_steps = torch.arange(1, noise_steps+1, device=device)
        self.heat_steps = self.convert_timespace(diffusion_steps).to(device)
        print(f'dt:{self.heat_steps}')
        self.std = torch.sqrt(self.heat_steps / 2)
        # self.std[4:] = self.std[3]
        self.std = torch.clamp(self.std, max=image_size//2)
        print(f'heat kernel std:{self.std}')

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
            return time_steps * int((self.max_heat_step / self.noise_steps))
        elif self.time_type == 'exp':
            factor = torch.tensor(2.)
            log_space = torch.linspace(0, 1, steps=self.noise_steps)
            range_adjusted = (int(math.sqrt(self.max_heat_step/2)) - int(math.sqrt(self.min_heat_step/2)))
        
            exp_sequence = int(math.sqrt(self.min_heat_step/2)) + range_adjusted * (torch.exp(factor * log_space) -1) / (torch.exp(factor) - 1)
            exp_sequence = torch.round(exp_sequence)
            exp_sequence[0] = int(math.sqrt(self.min_heat_step/2))
            exp_sequence[-1] = int(math.sqrt(self.max_heat_step/2))
            return (2 * exp_sequence**2).to(torch.int64)
        else:
            raise "Wrong time type"


    def compute_K(self, u):
        obstacle_with_edges = self.obstacle_masks.clone()

        K = self.alpha * torch.ones_like(u)      
        
        K[obstacle_with_edges] = 0
        return K
    

    def gaussian_filter(self, ut, diffusion_steps, kernel_size=3):
        input_tensor = ut.clone()
        if kernel_size % 2 == 0:
            kernel_size += 1

        B = input_tensor.shape[0]
        result = []

        for i in range(B):
            std = self.std[diffusion_steps[i]-1] #t[i]

            x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size, dtype=input_tensor.dtype, device=input_tensor.device)
            kernel = torch.exp(-x**2 / (2*std**2))
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
    
        
    def compute_ut(self, heat_dts, diffusion_steps, heat_sources):
        heat_sources = self.convert_space(heat_sources, 'pixel')

        u = torch.zeros((len(heat_dts), self.image_size, self.image_size), dtype=torch.float32, device=heat_sources.device)

        K = self.compute_K(u)

        for b in range(heat_sources.shape[0]):
            for g in range(heat_sources.shape[1]):
                u[b, heat_sources[b,g,0], heat_sources[b,g,1]] = self.u0
            
        max_time_steps = torch.max(heat_dts).item()

        for t in range(1, max_time_steps+1):
            current_mask = (t < heat_dts)
            laplacian = torch.zeros_like(u).to(torch.float32)

            top = u[:, :-2, 1:-1]
            bottom = u[:, 2:, 1:-1]
            left = u[:, 1:-1, :-2]
            right = u[:, 1:-1, 2:]
            
            top_mask = self.obstacle_masks[:, :-2, 1:-1]
            bottom_mask = self.obstacle_masks[:, 2:, 1:-1]
            left_mask = self.obstacle_masks[:, 1:-1, :-2]
            right_mask = self.obstacle_masks[:, 1:-1, 2:]

            count_non_obstacle = 4 - (top_mask.float() + bottom_mask.float() + left_mask.float() + right_mask.float())

            laplacian[:, 1:-1, 1:-1] = top + bottom + left + right - count_non_obstacle * u[:, 1:-1, 1:-1]

            u[current_mask] += self.dt * (K[current_mask] * laplacian[current_mask])

        u_filtered = self.gaussian_filter(u, diffusion_steps)
        return u_filtered
    
    
    def sample_from_heat(self, u, n):
        ut = u.clone()

        ut = self.exclude_insulators(ut)    # exclude samples in insulators by numerical error
        ut_flat = ut.view(ut.size(0),-1)
        sampled_indices = torch.multinomial(ut_flat, n, replacement=True)

        h = sampled_indices // u.size(2)
        w = sampled_indices % u.size(2)

        return torch.stack((h, w), dim=-1)

    
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

        if x is not None:
            B, N, _ = x.shape
            batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(B, N)
            h_indices = x[:, :, 0].long()
            w_indices = x[:, :, 1].long()

            scores = score_field[batch_indices, h_indices, w_indices, :]

            return scores
        return score_field


    def forward_diffusion(self, diffusion_steps, heat_sources, sample_num, obstacle_masks=None):
        self.create_obstacle_masks(len(diffusion_steps) * sample_num, obstacle_masks)
        heat_dts = self.heat_steps[diffusion_steps-1]
        ut_batch = self.compute_ut(heat_dts, diffusion_steps, heat_sources)
        x_t = self.sample_from_heat(ut_batch, sample_num)
#         return ut_batch, self.convert_space(x_t, 'norm')
        return ut_batch, self.score(ut_batch, x_t), self.score(ut_batch), self.convert_space(x_t, 'norm')
    
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps+1, size=(n,), device=self.device).long()