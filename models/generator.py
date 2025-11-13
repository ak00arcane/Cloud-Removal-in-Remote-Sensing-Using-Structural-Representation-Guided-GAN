import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# Basic Building Blocks
# ==============================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ==============================
# Generator (U-Net Style)
# ==============================

class CloudRemovalGenerator(nn.Module):
    def __init__(self, config):
        super(CloudRemovalGenerator, self).__init__()

        # Define architecture parameters
        in_channels = getattr(config, "INPUT_CHANNELS", 6)   # cloudy + temporal
        out_channels = getattr(config, "OUTPUT_CHANNELS", 3)
        encoder_channels = getattr(config, "ENCODER_CHANNELS", [64, 128, 256, 512])
        decoder_channels = getattr(config, "DECODER_CHANNELS", [512, 256, 128, 64])

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        prev_ch = in_channels
        for ch in encoder_channels:
            self.encoder_blocks.append(ConvBlock(prev_ch, ch))
            prev_ch = ch
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(encoder_channels[-1], encoder_channels[-1] * 2)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        reversed_enc = list(reversed(encoder_channels))
        prev_ch = encoder_channels[-1] * 2
        for ch in decoder_channels:
            self.decoder_blocks.append(UpBlock(prev_ch, ch))
            prev_ch = ch

        # Final output
        self.output_layer = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, cloudy_img, cloud_mask=None, temporal_img=None):
        # Combine inputs (cloudy + temporal)
        if temporal_img is not None:
            x = torch.cat([cloudy_img, temporal_img], dim=1)
        else:
            x = cloudy_img

        # Encoder path
        skips = []
        for enc in self.encoder_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        skips = list(reversed(skips))
        for i, dec in enumerate(self.decoder_blocks):
            x = dec(x, skips[i])

        # Final predicted clean image
        pred_img = self.output_layer(x)

        # For compatibility with train.py expecting 3 outputs
        pred_grad = torch.zeros_like(pred_img)   # placeholder
        pred_stru = torch.zeros_like(pred_img)   # placeholder

        return pred_img, pred_grad, pred_stru
