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
        self.obstacles = obstacle_mask
        if obstacle_mask is None:
            obstacle_mask = torch.zeros((self.batchsize, self.Nx, self.Ny), dtype=torch.bool, device=self.dcvice)
        
        # Neumann BC
        obstacle_boundary = F.conv2d(obstacle_mask.float().unsqueeze(1), torch.ones((1,1,3,3), device=self.device), padding=1).squeeze(1) > 0
        obstacle_boundary = obstacle_boundary & (~obstacle_mask)
        
        results = []
        for idx in tqdm(range(self.batchsize)):
            u_temp = self.u[idx].clone().unsqueeze(0)  # Extract current batch
            for t in range(timesteps[idx].item()):
                u_new = u_temp.clone()

                # Apply the finite difference scheme
                u_new[:, 1:-1, 1:-1] = u_temp[:, 1:-1, 1:-1] + self.alpha * self.dt * (
                    (u_temp[:, 2:, 1:-1] - 2*u_temp[:, 1:-1, 1:-1] + u_temp[:, :-2, 1:-1]) / self.dx**2 +
                    (u_temp[:, 1:-1, 2:] - 2*u_temp[:, 1:-1, 1:-1] + u_temp[:, 1:-1, :-2]) / self.dy**2
                )

                # Dirichlet BC
                # u_new[:, obstacle_mask[idx]] = u_temp[:, obstacle_mask[idx]]

                # Neumann BC
                u_new[:, 0, :] = u_new[:, 1, :]
                u_new[:, -1, :] = u_new[:, -2, :]
                u_new[:, :, 0] = u_new[:, :, 1]
                u_new[:, :, -1] = u_new[:, :, -2]
                
                u_new[:, obstacle_boundary[idx]] = u_temp[:, obstacle_boundary[idx]]
                
                u_temp = u_new

            results.append(u_temp.squeeze(0))

        return torch.stack(results)
    
    def forward_diffusion(self, x0, timesteps, obstacle_mask=None):
        self.initialize(x0)
        timesteps = timesteps * int((80000 / self.noise_steps))
        ut = self.get_ut(timesteps, obstacle_mask)
        return ut
#         return self.gradient_field(ut)
    
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

