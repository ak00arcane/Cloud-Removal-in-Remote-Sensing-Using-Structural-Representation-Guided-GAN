import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for LSGAN
    5-layer pure convolutional network
    """
    def __init__(self, in_channels=3):  # Fixed: removed config parameter
        super().__init__()
        
        def conv_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            # Layer 1: (3, 256, 256) -> (64, 128, 128)
            *conv_block(in_channels, 64, normalize=False),
            
            # Layer 2: (64, 128, 128) -> (128, 64, 64)
            *conv_block(64, 128),
            
            # Layer 3: (128, 64, 64) -> (256, 32, 32)
            *conv_block(128, 256),
            
            # Layer 4: (256, 32, 32) -> (512, 16, 16)
            *conv_block(256, 512),
            
            # Layer 5: (512, 16, 16) -> (1, 8, 8)
            nn.Conv2d(512, 1, 4, 2, 1)
        )
    
    def forward(self, x):
        return self.model(x)