import torch
import torch.nn as nn

class USPolicy3(nn.Module):
    def __init__(self, image_size_hw=(150, 200)):
        super().__init__()
        H, W = image_size_hw

        # match skrl config: conv stride 2/2/2/1, relu activations
        self.features_extractor_container = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=2, padding=0),  # .0
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), # .2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0), # .4
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # .6
            nn.ReLU(),
        )

        # compute flatten dim (should become 17920 for 150x200)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, H, W)
            flat_dim = int(self.features_extractor_container(dummy).flatten(1).shape[1])

        # match skrl config: [512,256,128] with ELU
        self.net_container = nn.Sequential(
            nn.Linear(flat_dim, 512),  # .0
            nn.ELU(),
            nn.Linear(512, 256),       # .2
            nn.ELU(),
            nn.Linear(256, 128),       # .4
            nn.ELU(),
        )

        self.policy_layer = nn.Linear(128, 3)
        self.value_layer = nn.Linear(128, 1)
        self.log_std_parameter = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        # x: (N,1,H,W) float
        z = self.features_extractor_container(x).flatten(1)
        z = self.net_container(z)
        return self.policy_layer(z)   # mean actions