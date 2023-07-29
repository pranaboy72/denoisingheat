import torch
from scorefield.models.denoising_diffusion import Unet, GaussianDiffusion

model = Unet(
    dim=64,
    dim_mults=(1,2,4,8),
    flash_attn=True,
)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,
)

training_images = 
loss = diffusion(training_images)
