import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from scorefield.utils.utils import clip_batch_vectors
from scipy.ndimage import distance_transform_edt


class HeatDiffusion(object):
    def __init__(self, image_size, u0=1., noise_steps=500, heat_steps=1000, alpha=1.0, beta=10.0,  precision='single', device='cuda'):
        self.image_size = image_size
        self.u0 = u0
        self.noise_steps = noise_steps
        self.heat_steps = heat_steps
        self.alpha = alpha
        self.beta = beta
        self.dt = 1 / (4 * alpha)   # For stability, CFL condition: dt <= (dx^2 * dy*2) / (2*alpha* (dx^2 + dy^2))
        self.precision = torch.float64 if precision == 'double' else torch.float32
        
        self.device = device
        

    def convert_space(self, previous, converted):
        """
            Convert from [-1,1] space to [0,127] space
        """
        if converted == 'pixel':
            return ((previous + 1) * 0.5 * (self.image_size - 1)).long()
        elif converted == 'norm':
            return (previous / (self.image_size - 1) * 2 - 1)


    def compute_K(self, u):
        obstacle_with_edges = self.obstacle_masks.clone()
        
        dk = torch.stack([torch.tensor(distance_transform_edt(mask.cpu().numpy())) for mask in ~obstacle_with_edges])
        dk = dk.to(u.device).float()
        decay = torch.exp(-dk/self.beta)
        K = self.alpha * torch.ones_like(u)        
        
#         K = K * (1 - decay)
        K[obstacle_with_edges] = 0
        return K
    

    def gaussian_filter(self, input_tensor, kernel_size=3, std=1.0):
#         if kernel_size %2 == 0:
#             kernel_size += 1
            
#         x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size, dtype=input_tensor.dtype, device=input_tensor.device)
#         kernel = torch.exp(-x**2 / (2*std**2))
#         kernel /= kernel.sum()
        
#         gaussian_filter = torch.outer(kernel, kernel).unsqueeze(0).unsqueeze(0)
#         gaussian_filter = gaussian_filter.expand((2, 1, kernel_size, kernel_size))

#         B, H, W = input_tensor.shape
#         gaussian_filter = gaussian_filter.repeat(B, 1, 1, 1)
        
#         padding = kernel_size // 2
#         input_tensor_padded = F.pad(input_tensor.permute(0, 3, 1, 2), (padding, padding, padding, padding), mode='reflect')
        
#         smoothed_tensor = F.conv2d(input_tensor_padded, gaussian_filter, stride=1, padding=0, groups=2)
        
#         return smoothed_tensor.permute(0,2,3,1)
        
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size, dtype=input_tensor.dtype, device=input_tensor.device)
        kernel = torch.exp(-x**2 / (2*std**2))
        kernel /= kernel.sum()

        gaussian_filter = torch.outer(kernel, kernel).unsqueeze(0).unsqueeze(0)

        padding = kernel_size // 2
        input_tensor_padded = F.pad(input_tensor.unsqueeze(1), (padding, padding, padding, padding), mode='reflect')

        smoothed_tensor = F.conv2d(input_tensor_padded, gaussian_filter, stride=1, padding=0)

        return smoothed_tensor.squeeze(1)
    
    def revise_obstacle_masks(self, batch_size, obstacle_masks):
        if obstacle_masks is None:
            obstacle_masks = torch.zeros((batch_size, self.image_size, self.image_size), dtype=torch.bool, device=self.device)

        obstacle_masks[:, :, 0] = 1
        obstacle_masks[:, :, -1] = 1
        obstacle_masks[:, 0, :] = 1
        obstacle_masks[:, -1, :] = 1
   
        self.obstacle_masks = obstacle_masks
        
    def compute_ut(self, time_steps, heat_sources):
        heat_sources = self.convert_space(heat_sources, 'pixel') # When [-0.7, 0.] => [19,63] (128x128)
        
        B = heat_sources.shape[0]
        u = torch.zeros((B, self.image_size, self.image_size), dtype=self.precision, device=heat_sources.device)

        K = self.compute_K(u)

        for b in range(B):
            u[b, heat_sources[b,0], heat_sources[b, 1]] = self.u0

        f = u.clone()
            
        max_time_steps = torch.max(time_steps).item()
        
        for t in range(1, max_time_steps+1):
            current_mask = (t < time_steps)
            laplacian = torch.zeros_like(u).to(self.precision)

            top = u[:, :-2, 1:-1]
            bottom = u[:, 2:, 1:-1]
            left = u[:, 1:-1, :-2]
            right = u[:, 1:-1, 2:]

            laplacian[:, 1:-1, 1:-1] = top + bottom + left + right - 4 * u[:, 1:-1, 1:-1]
            
            u[current_mask] += self.dt * (K[current_mask] * laplacian[current_mask])
             
        u = self.gaussian_filter(u)
        sums = u.sum(dim=(1,2), keepdim=True)
        norm_u = u / sums

        return norm_u
    
    def sample_from_heat(self, ut):
        B = ut.size(0)
        
        valid_mask = (ut > 0).flatten().view(B, -1)
        valid_indices = torch.nonzero(valid_mask)
        
        random_indices = []
        for b in range(B):
            choices = valid_indices[valid_indices[:, 0] == b][:, 1]
            if len(choices) > 0:
                random_index = choices[torch.randint(0, len(choices), (1,))].item()
                random_indices.append(random_index)
            else:
                random_indices.append(torch.randint(0, ut.shape[1] * ut.shape[2], (1,)).item())
        random_indices = torch.tensor(random_indices, dtype=torch.int64)

        h = random_indices // ut.size(2)
        w = random_indices % ut.size(2)

        return torch.stack((h, w), dim=1)

    def score(self, u, x=None):
        eps = 1e-9
        u_log = torch.log(torch.clamp(u, min=eps)).unsqueeze(1)

        kernel_w = torch.tensor([[-0.5, 0, 0.5]], dtype=self.precision, device=self.device).unsqueeze(0).unsqueeze(0)
        kernel_h = torch.tensor([[-0.5],[0],[0.5]], dtype=self.precision, device=self.device).unsqueeze(0).unsqueeze(0)

        grad_w = F.conv2d(u_log, kernel_w, padding=(0,1)).squeeze(1)
        grad_h = F.conv2d(u_log, kernel_h, padding=(1,0)).squeeze(1)

        grad_w[self.obstacle_masks] = 0
        grad_h[self.obstacle_masks] = 0
        
        score_field = torch.stack([grad_h, grad_w], dim=-1) 
        
        score_field[self.obstacle_masks] = 0
        if x is not None:
            batch_indices = torch.arange(u.shape[0])
            x_indices = x[:, 0]
            y_indices = x[:, 1]
            score = score_field[batch_indices, x_indices, y_indices, :]
        
            return score
#         score_field = clip_batch_vectors(score_field, 0.01)
        return score_field

    def forward_diffusion(self, time_steps, heat_sources, obstacle_masks=None):
        self.revise_obstacle_masks(heat_sources.shape[0], obstacle_masks)
        time_steps = time_steps * int((self.heat_steps / self.noise_steps))
        ut_batch = self.compute_ut(time_steps, heat_sources)
        x_t = self.sample_from_heat(ut_batch)
        
        return self.score(ut_batch, x_t), self.score(ut_batch), self.convert_space(x_t, 'norm')
#         return ut_batch
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).long()
    
    
    