import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

class HeatDiffusion(object):
    def __init__(self, image_size, Lx=2, Ly=2, alpha=.5, u0=1e4, noise_steps=500, device='cuda'):
        self.device = device
        self.Lx = Lx
        self.Ly = Ly
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
        self.u = torch.full((self.batchsize, self.Nx, self.Ny), self.u0, dtype=torch.float32, device=self.device)
        self.u.fill_(0.0)  # reset all to u0

        for idx, x0 in enumerate(x0):
            x = 0.5 * (1 - x0[0][0]) * self.Lx
            y = 0.5 * (x0[0][1] + 1) * self.Ly 

            i, j = x / self.dx, y / self.dy

            i0, j0 = int(i), int(j)
            i1, j1 = i0 + 1, j0 + 1

            fx, fy = i - i0, j - j0

            self.u[idx, i0, j0] += (1 - fx) * (1 - fy) * self.u0
            self.u[idx, i1, j0] += fx * (1 - fy) * self.u0
            self.u[idx, i0, j1] += (1 - fx) * fy * self.u0
            self.u[idx, i1, j1] += fx * fy * self.u0

    def get_ut(self, timesteps, obstacle_mask=None):  
        self.obstacles = obstacle_mask
        if obstacle_mask is None:
            obstacle_mask = torch.zeros((self.batchsize, self.Nx, self.Ny), dtype=torch.bool, device=self.device)

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

                # Ensuring that heat doesn't flow in/out of the obstacle
                u_new[:, obstacle_mask[idx]] = u_temp[:, obstacle_mask[idx]]

                u_temp = u_new

            results.append(u_temp.squeeze(0))

        return torch.stack(results)

    def forward_diffusion(self, x0, timesteps, obstacle_mask=None):
        self.initialize(x0)
        timesteps = timesteps * int(70000/self.noise_steps)
        ut = self.get_ut(timesteps, obstacle_mask)
        return ut
        # return self.gradient_field(ut)

    def gradient_field(self, u):
        if not isinstance(u, torch.Tensor):
            u_dis = torch.tensor(u, device=self.device)
        else:
            u_dis = u.clone()

        u_dis = u_dis.unsqueeze(1)

        kernel_x = torch.tensor([[-0.5, 0, 0.5]], device=self.device).float().unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-0.5], [0], [0.5]], device=self.device).float().unsqueeze(0).unsqueeze(0)

        grad_x = F.conv2d(u_dis, kernel_x, padding=(0,1)).squeeze(1)
        grad_y = F.conv2d(u_dis, kernel_y, padding=(1,0)).squeeze(1)

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
    def __init__(self, image_size, u0=1, noise_steps=500, heat_steps=200000, alpha=2.0, device='cuda'):
        self.image_size = image_size
        self.u0 = u0
        self.noise_steps = noise_steps
        self.heat_steps = heat_steps
        self.alpha = alpha
        self.dt = 0.1 #1 / (4 * alpha)   # For stability, CFL condition: dt <= (dx^2 * dy*2) / (2*alpha* (dx^2 + dy^2))
        self.device = device
        self.laplacian_kernel = torch.tensor([[[[0, 1, 0], 
                                       [1, -4, 1], 
                                       [0, 1, 0]]]], device=device).float()


    def convert_space(self, previous):
        """
            Convert from [-1,1] space to [0,127] space
        """
        return ((previous + 1) * 0.5 * (self.image_size - 1)).long()
    
    
    def compute_alpha(self, dist, max_alpha, decay_rate=0.1):
        return max_alpha * torch.exp(-decay_rate * dist)


    def compute_K(self, u, obstacle_masks, width=10, corner_radius=8):
        obstacle_with_edges = obstacle_masks.clone()
        obstacle_with_edges[:, 0, :] = 1
        obstacle_with_edges[:, -1, :] = 1
        obstacle_with_edges[:, :, 0] = 1
        obstacle_with_edges[:, :, -1] = 1
        # obstacle_with_edges = ~obstacle_with_edges
        
#         obstacle_np = obstacle_with_edges.cpu().numpy().astype(np.bool)
#         distance_transfroms = [distance_transform_edt(batch) for batch in obstacle_np]
#         distances = torch.tensor(np.stack(distance_transfroms), device=u.device, dtype=torch.float32)
        
#         variable_alpha = self.compute_alpha(distances, self.alpha)
#         K = variable_alpha * torch.ones_like(u)
        K = self.alpha * torch.ones_like(u)
        
        K[obstacle_with_edges] = 0
        
        # y, x = torch.meshgrid(torch.arange(self.image_size), torch.arange(self.image_size))
        # mask1 = (x - width)**2 + (y - (self.image_size - width))**2 <= corner_radius**2
        # mask2 = (x - (self.image_size - width))**2 + (y - (self.image_size - width))**2 <= corner_radius**2
        
        # K[:, mask1] = 0
        # K[:, mask2] = 0
        
        return K
        
    def compute_ut(self, time_steps, heat_sources, obstacle_masks):
#         torch.set_default_dtype(torch.float64)
        
        heat_sources = self.convert_space(heat_sources)
        
        B = heat_sources.shape[0]
        u = torch.zeros((B, self.image_size, self.image_size), device=heat_sources.device)

        if obstacle_masks is None:
            obstacle_masks = torch.zeros((B, self.image_size, self.image_size), dtype=torch.bool, device=heat_sources.device)

        K = self.compute_K(u, obstacle_masks)

        max_t = time_steps.max().item()
#         for b in range(B):
#             u[b, heat_sources[b, 0], heat_sources[b, 1]] = self.u0
        
        for t in range(1, max_t + 1):
            current_mask = (t <= time_steps)
            u[:, heat_sources[:,0], heat_sources[:,1]] = self.u0
            laplacian = F.conv2d(u.unsqueeze(1), self.laplacian_kernel,padding=1).squeeze(1)
            u[current_mask] += self.dt * (K[current_mask] * laplacian[current_mask])
            u[current_mask][obstacle_masks[current_mask]] = 0
            
        return u

    def gradient_field(self, u):
        if not isinstance(u, torch.Tensor):
            u_dis = torch.tensor(u, device=self.device)
        else:
            u_dis = u.clone()
        u_dis = u_dis.to(torch.float64)
        eps = 1e-10
        u_dis = torch.log(u_dis + eps).unsqueeze(1)

        kernel_x = torch.tensor([[-0.5, 0, 0.5]], dtype=torch.float64, device=self.device).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-0.5], [0], [0.5]], dtype=torch.float64, device=self.device).unsqueeze(0).unsqueeze(0)

        grad_x = F.conv2d(u_dis, kernel_x, padding=(0,1)).squeeze(1)
        grad_y = F.conv2d(u_dis, kernel_y, padding=(1,0)).squeeze(1)

        return torch.stack([grad_x, grad_y], dim=-1).to(torch.float32)
    
    def forward_diffusion(self, time_steps, heat_sources, obstacle_masks=None):
        time_steps = time_steps * int((self.heat_steps / self.noise_steps))
        ut_batch = self.compute_ut(time_steps, heat_sources, obstacle_masks)
        return self.gradient_field(ut_batch)
#         return ut_batch
    
    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps, (n,), device=self.device)