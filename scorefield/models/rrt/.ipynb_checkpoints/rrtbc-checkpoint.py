import torch
import torch.nn as nn
import torch.nn.functional as F

class RRTBC(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.img_fc = nn.Linear(128*20*20, 512)
        
        self.coord_embed = nn.Linear(2, 128)
        
        self.fc1 = nn.Linear(512 + 128, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        
    def forward(self, img, coord):
        x1 = F.relu(self.conv1(img))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = x1.view(x1.size(0), -1)
        x1 = F.relu(self.img_fc(x1))
        
        x2 = F.relu(self.coord_embed(coord))
        
        torch.cat((x1, x2), dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)   
    