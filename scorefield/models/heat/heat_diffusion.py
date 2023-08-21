import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class HeatDiffusion(object):
    def __init__(self, image_size, Lx=2, Ly=2, alpha=.5, u0=1e4, noise_steps=500, device='cuda'):
        self.device = device
        self.Lx = Lx
        self.Ly = Ly
        self.image_size = image_size
        self.Nx = image_size
        self.Ny = image_size
        self.alpha = alpha
        self.u0 = u0
        self.dx = Lx / (image_size - 1)
        self.dy = Ly / (image_size - 1)
        self.dt = 0.25 * min(self.dx, self.dy)**2 / alpha
        self.noise_steps = noise_steps
    
    def initialize(self, x0):
        self.batchsize = x0.shape[0]
        self.u = torch.full((self.batchsize, self.Nx, self.Ny), 0.0, dtype=torch.float32, device=x0.device)
        
        for idx, coord in enumerate(x0):
            x = 0.5 * (1 - coord[0][0]) * self.Lx
            y = 0.5 * (coord[0][1] + 1) * self.Ly

            i = int(torch.round(x / self.dx).item())
            j = int(torch.round(y / self.dy).item())

            self.u[idx, i, j] = self.u0
            
        
    def get_ut(self, timesteps, obstacle_mask=None):  
        if obstacle_mask is None:
            obstacle_mask = torch.zeros((self.batchsize, self.Nx, self.Ny), dtype=torch.bool, device=self.device)

        # Detect obstacle boundaries
        obstacle_boundary = F.conv2d(obstacle_mask.float().unsqueeze(1), torch.ones((1,1,3,3), device=self.device), padding=1).squeeze(1) > 0
        obstacle_boundary = obstacle_boundary & (~obstacle_mask)

        results = []
        for idx in range(self.batchsize):
            u_temp = self.u[idx].clone().unsqueeze(0)  # Extract current batch
            for t in range(timesteps[idx].item()):
                u_new = u_temp.clone()

                # Finite difference scheme for inner cells
                u_new[:, 1:-1, 1:-1] = u_temp[:, 1:-1, 1:-1] + self.alpha * self.dt * (
                    (u_temp[:, 2:, 1:-1] - 2*u_temp[:, 1:-1, 1:-1] + u_temp[:, :-2, 1:-1]) / self.dx**2 +
                    (u_temp[:, 1:-1, 2:] - 2*u_temp[:, 1:-1, 1:-1] + u_temp[:, 1:-1, :-2]) / self.dy**2
                )

                # Neumann BC for obstacles
                u_new[0, obstacle_boundary[idx]] = u_temp[0, obstacle_boundary[idx]]

                # Neumann BC for edges of the map
                u_new[:, 0, 1:-1] = u_temp[:, 1, 1:-1]
                u_new[:, 1:-1, 0] = u_temp[:, 1:-1, 1]
                u_new[:, -1, 1:-1] = u_temp[:, -2, 1:-1]
                u_new[:, 1:-1, -1] = u_temp[:, 1:-1, -2]
                u_new[:, 0, 0] = (u_temp[:, 1, 0] + u_temp[:, 0, 1]) / 2.0
                u_new[:, -1, 0] = (u_temp[:, -2, 0] + u_temp[:, -1, 1]) / 2.0
                u_new[:, 0, -1] = (u_temp[:, 1, -1] + u_temp[:, 0, -2]) / 2.0
                u_new[:, -1, -1] = (u_temp[:, -2, -1] + u_temp[:, -1, -2]) / 2.0

                u_temp = u_new

            results.append(u_temp.squeeze(0))

        return torch.stack(results)
    
    def forward_diffusion(self, x0, timesteps, obstacle_mask=None):
        self.initialize(x0)
        timesteps = timesteps * int((70000 / self.noise_steps))
        ut = self.get_ut(timesteps, obstacle_mask)
#         return ut
        return self.gradient_field(ut)
    
    def gradient_field(self, u):
        if not isinstance(u, torch.Tensor):
            u_dis = torch.tensor(u, device=self.device)
        else:
            u_dis = u.clone()
            
        u_dis = u_dis.unsqueeze(1)
        
        kernel_x = torch.tensor([[-0.5, 0, 0.5]], device=self.device).float().unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-0.5], [0], [0.5]], device=self.device).float().unsqueeze(0).unsqueeze(0)
        
        scale_factor = (self.image_size - 1) / 2.0
        
        grad_x = F.conv2d(u_dis, kernel_x, padding=(0,1)).squeeze(1) / scale_factor
        grad_y = F.conv2d(u_dis, kernel_y, padding=(1,0)).squeeze(1) / scale_factor
        
        return torch.stack([grad_x, grad_y], dim=-1)
    
    def visualize_distribution(self, u, threshold=1e-4):
        for idx, batch_u in enumerate(u):
            plt.figure(figsize=(8,6))
            mask = np.where(batch_u > threshold)
            batch_u[mask] = 0
            
            plt.imshow(batch_u, cmap='hot', interpolation='nearest', origin='lower',alpha=0.5)
            plt.show()
        
    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps, (n,), device=self.device)


class HeatDiffusion2(object):
    def __init__(self, image_size, u0=1000, noise_steps=500, heat_steps=200000, alpha=0.5, device='cuda'):
        self.image_size = image_size
        self.u0 = u0
        self.noise_steps = noise_steps
        self.heat_steps = heat_steps
        self.alpha = alpha
        self.dt = 0.5 #int(1 / (4 * alpha))   # For stability, CFL condition: dt <= (dx^2 * dy*2) / (2*alpha* (dx^2 + dy^2))
        self.device = device
        self.laplacian_kernel = torch.tensor([[[[0, 1, 0], 
                                       [1, -4, 1], 
                                       [0, 1, 0]]]], device=device).float()


    def convert_space(self, previous):
        """
            Convert from [-1,1] space to [0,127] space
        """
        return ((previous + 1) * 0.5 * (self.image_size - 1)).long()
        
        
    def compute_ut(self, time_steps, heat_sources, obstacle_masks=None):
        heat_sources = self.convert_space(heat_sources)
        
        B = heat_sources.shape[0]
        u = torch.zeros((B, self.image_size, self.image_size), device=heat_sources.device)

        if obstacle_masks is None:
            obstacle_masks = torch.zeros((B, self.image_size, self.image_size), dtype=torch.bool, device=device)

        K = self.alpha * torch.ones_like(u)
        K[obstacle_masks] = 0
        K[:, 0, :] = 0
        K[:, -1,:] = 0
        K[:, :, 0] = 0
        K[:, :,-1] = 0
        
        
        for b, t in enumerate(time_steps):
            u[b, heat_sources[b, 0], heat_sources[b,1]] = self.u0
            
            for _ in range(int(t.item())):   
                u_with_channel = u[b].unsqueeze(0).unsqueeze(0)
                laplacian = F.conv2d(u_with_channel, self.laplacian_kernel, padding=1).squeeze(0).squeeze(0)

                u[b] += self.dt * (K[b] * laplacian)
        
        # max_t = int(torch.max(time_steps).item())
        
        # for t in range(max_t):
        #     u[torch.arange(B), heat_sources[:, 0], heat_sources[:, 1]] = self.u0
        #     laplacian = F.conv2d(u.unsqueeze(1), self.laplacian_kernel, padding=1).squeeze(1)
            
        #     not_exceeded_mask = (t < time_steps).unsqueeze(-1).unsqueeze(-1)
        #     u[not_exceeded_mask] = u[not_exceeded_mask] + self.dt * (K[not_exceeded_mask] * laplacian[not_exceeded_mask])

        return u

    def gradient_field(self, u):
        u_log = torch.log(u)
        u_dis = u_log.clone().unsqueeze(1)
        
        kernel_x = torch.tensor([[-0.5, 0, 0.5]], device=u.device).float().unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-0.5], [0], [0.5]], device=u.device).float().unsqueeze(0).unsqueeze(0)
        
        scale_factor = (self.image_size - 1) / 2.0
        
        grad_x = F.conv2d(u_dis, kernel_x, padding=(0,1)).squeeze(1) / scale_factor
        grad_y = F.conv2d(u_dis, kernel_y, padding=(1,0)).squeeze(1) / scale_factor
        
        return torch.stack([grad_x, grad_y], dim=-1)
    
    def forward_diffusion(self, time_steps, heat_sources, obstacle_masks):
        time_steps = time_steps * int((self.heat_steps / self.noise_steps))
        ut_batch = self.compute_ut(time_steps, heat_sources, obstacle_masks)
        return self.gradient_field(ut_batch)
        # return ut_batch
    
    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps, (n,), device=self.device)