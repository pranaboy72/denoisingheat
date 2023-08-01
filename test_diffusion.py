import torch
from scorefield.models.diffusion_encoder.denoising_diffusion import Unet, GaussianDiffusion

model = Unet(
    dim=64,
    dim_mults=(1,2,4,8),
    flash_attn=True,
)

diffusion = GaussianDiffusion(
    model,
    image_size=64,
    timesteps=1000,
)

training_images = 
loss = diffusion(training_images)
