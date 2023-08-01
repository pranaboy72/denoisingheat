import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from ..cnn_encoder.cnn_encoder import make_encoder
from..diffusion_encoder.denoising_diffusion import Unet, GaussianDiffusion

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, 
        feature_dim, log_std_min, log_std_max,
    ):
        super().__init__()

        self.encoder = make_encoder(obs_shape, feature_dim)
    

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim), nn.ReLU(), # + 1: time 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape)
        )

        self.apply(weight_init)

    def forward(
        self, obs, timestep, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.Tensor([timestep]).view(1,-1).to(obs.device)
        
        obs = torch.cat([obs, timestep], dim=1)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, feature_dim, 
    ):
        super().__init__()


        self.encoder = make_encoder(obs_shape, feature_dim)

        self.Q1 = QFunction(
            feature_dim+1, action_shape, hidden_dim  # 1: time
        )
        self.Q2 = QFunction(
            feature_dim+1, action_shape, hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, timestep, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.Tensor([timestep]).view(1,-1).to(obs.device)
        obs = torch.cat([obs, timestep], dim=1)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        return q1, q2


    
        
    