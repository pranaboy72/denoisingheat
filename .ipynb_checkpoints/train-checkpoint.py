import os
import math

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

from scorefield.models.ddpm.denoising_diffusion import Unet
from scorefield.models.ddpm.denoising_diffusion_1d import Unet1D, GaussianDiffusion1D, Dataset1D, Trainer1D
from scorefield.models.ddpm.gaussian_diffusion import Diffusion
# from scorefield.utils.rendering import Maze2dRenderer
from scorefield.utils.rl_utils import load_config
from scorefield.utils.utils import log_num_check, imshow, gen_goals, random_batch, eval_batch, prepare_input
from scorefield.utils.diffusion_utils import bilinear_interpolate

import matplotlib.pyplot as plt


# Args
config_dir = "./scorefield/configs/diffusion.yaml"
args = load_config(config_dir)
device = args['device']

model_path = os.path.join(args['log_path'], args['model_path'])

map_img = Image.open("map.png")
bounds = args['bounds']


class Unet2D(Unet):
    def __init__(
        self, 
        dim, 
        out_dim, 
        dim_mults=(1, 2, 4, 8)
    ):
        super().__init__(dim=dim, out_dim=out_dim, dim_mults=dim_mults)

    def forward(self, obs, x_t, t):
        score_map = super().forward(obs, t)
        score = bilinear_interpolate(score_map, x_t)    # output: (B,2)
        return score

img_size = args['image_size']
noise_steps = args['noise_steps']
train_lr = args['train_lr']
    
model = Unet2D(
    dim=img_size,
    out_dim = 2,
    dim_mults = (1, 2, 4, 8),
).to(device)

diffusion = Diffusion(
    bounds,
    input_size = (2,), 
    noise_steps= noise_steps,
    device=device,
    beta_start=1e-5,
    beta_end=1e-3,
)

optim = torch.optim.Adam(params=model.parameters(), lr=train_lr)


epochs = args['epochs']
batch_size = args['batch_size']
goal_num = args['goal_num']

goals = gen_goals(bounds, goal_num)

assert batch_size % goal_num == 0, 'batch size has to be divided by the goal number'
n = batch_size // goal_num
expanded_goals = goals.unsqueeze(1).expand(-1, n, -1)

for iters in tqdm(range(epochs)):
    optim.zero_grad()
    
    # x0 = (torch.rand(goal_num, 2, device=device, dtype=torch.float32)*2 -1.) * 0.01 + goals
    random_offsets = (torch.rand(*expanded_goals.shape, device=goals.device, dtype=goals.dtype) * 2 - 1.) * 0.01
    x0 = expanded_goals + random_offsets
    obs = prepare_input(map_img, img_size, goal_pos=x0, circle_rad=2)
    t = diffusion.sample_timesteps(batch_size).to(device)
    
    x_noisy, noise = diffusion.forward_diffusion(x0, t)
    noise_pred = model(obs, x_noisy, t)
    loss =  F.l1_loss(noise, noise_pred)
    loss.backward()
    optim.step()
    
    if iters % 500 == 0:
        print(f"iter {iters}: {loss.item()}")

torch.save(model.state_dict(), "./logs/pretrained/multi_goals.pt")