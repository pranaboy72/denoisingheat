import torch
import torch.nn as nn
import torch.nn.functional as F


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias
    
class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, filters):
        super().__init__()
        
        assert len(obs_shape) == 3  # obs shape: (3x64x64)
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = len(filters)
        
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], filters[0], kernel_size=5, stride=2, padding=2)]
        )
        for i in range(self.num_layers-1):
            self.convs.append(nn.Conv2d(filters[i], filters[i+1], kernel_size=5, stride=2, padding=2))
        
        self.fc1 = nn.Linear(32 * 32, 2 * self.feature_dim)
        self.fc2 = nn.Linear(2 * self.feature_dim, feature_dim)

    
    def forward(self, obs):
        conv = torch.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        x = conv.view(conv.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
        
    
    def copy_conv_weights_from(self, source):
        """ Tie layers """
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])
            
    
def make_encoder(obs_shape, feature_dim):
    return Encoder(obs_shape, feature_dim, filters=[32,32,64,128,256])

    