import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 convs with BN+ReLU, keeps spatial size (padding=1)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """MaxPool then a conv block."""
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2),
        _conv_block(in_channels, out_channels),
    )

class _UpBlock(nn.Module):
    """Upsample with transposed conv then a conv block; concatenates skip."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concat, channels double (skip + upsampled)
        self.conv = _conv_block(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Shapes align for 256x256 inputs (power of two), but pad if off by one
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along channel dimension (aka skip connection)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """A lightweight UNet for 256x256 inputs/outputs."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64):
        super().__init__()
        self.enc1 = _conv_block(in_channels, base_channels)
        self.enc2 = _down_block(base_channels, base_channels * 2)
        self.enc3 = _down_block(base_channels * 2, base_channels * 4)
        self.enc4 = _down_block(base_channels * 4, base_channels * 8)
        self.enc5 = _down_block(base_channels * 8, base_channels * 16)

        self.bottleneck = _conv_block(base_channels * 16, base_channels * 32)

        self.up5 = _UpBlock(base_channels * 32, base_channels * 16)
        self.up4 = _UpBlock(base_channels * 16, base_channels * 8)
        self.up3 = _UpBlock(base_channels * 8, base_channels * 4)
        self.up2 = _UpBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)
        c5 = self.enc5(c4)

        bottleneck = self.bottleneck(c5)

        # Decoder: upsample and concatenate with corresponding encoder skip
        u5 = self.up5(bottleneck, c4)  # upsample to c4 size (32x32), concat with c4
        u4 = self.up4(u5, c3)          # upsample to c3 size (64x64), concat with c3
        u3 = self.up3(u4, c2)          # upsample to c2 size (128x128), concat with c2
        u2 = self.up2(u3, c1)          # upsample to c1 size (256x256), concat with c1
        u1 = self.up1(u2)
        return self.head(u1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: returns class predictions (not logits).
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Class predictions (B, H, W) with values in [0, num_classes-1]
        """
        logits = self.forward(x)  # (B, num_classes, H, W)
        probs = torch.softmax(logits, dim=1)  # Apply softmax
        predictions = torch.argmax(probs, dim=1)  # Get class indices
        return predictions
