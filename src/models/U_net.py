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
    """A lightweight UNet for 256x256 inputs/outputs.
    
    Architecture (for 256x256 input):
        Encoder:
            enc1: 256×256, base_ch        (no downsampling)
            enc2: 128×128, base_ch×2      (pool + conv)
            enc3: 64×64,   base_ch×4      (pool + conv)
            enc4: 32×32,   base_ch×8      (pool + conv)
            bottleneck: 16×16, base_ch×16 (pool + conv) - deepest level
        
        Decoder:
            up4: 32×32,   base_ch×8  (upsample + concat with enc4)
            up3: 64×64,   base_ch×4  (upsample + concat with enc3)
            up2: 128×128, base_ch×2  (upsample + concat with enc2)
            up1: 256×256, base_ch    (upsample + concat with enc1)
            head: 256×256, num_classes
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64):
        super().__init__()
        # Encoder
        self.enc1 = _conv_block(in_channels, base_channels)           # 256×256
        self.enc2 = _down_block(base_channels, base_channels * 2)     # 128×128
        self.enc3 = _down_block(base_channels * 2, base_channels * 4) # 64×64
        self.enc4 = _down_block(base_channels * 4, base_channels * 8) # 32×32
        self.bottleneck = _down_block(base_channels * 8, base_channels * 16)  # 16×16 (includes pooling!)

        # Decoder
        self.up4 = _UpBlock(base_channels * 16, base_channels * 8)  # 32×32, skip from enc4
        self.up3 = _UpBlock(base_channels * 8, base_channels * 4)   # 64×64, skip from enc3
        self.up2 = _UpBlock(base_channels * 4, base_channels * 2)   # 128×128, skip from enc2
        self.up1 = _UpBlock(base_channels * 2, base_channels)       # 256×256, skip from enc1

        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Kaiming for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Ensure classification head has uniform logits at init
        # This prevents bias toward certain classes before training
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.enc1(x)           # 256×256, base_ch
        c2 = self.enc2(c1)          # 128×128, base_ch×2
        c3 = self.enc3(c2)          # 64×64, base_ch×4
        c4 = self.enc4(c3)          # 32×32, base_ch×8
        bottleneck = self.bottleneck(c4)  # 16×16, base_ch×16

        # Decoder: upsample and concatenate with corresponding encoder skip
        u4 = self.up4(bottleneck, c4)  # 32×32, skip from enc4
        u3 = self.up3(u4, c3)          # 64×64, skip from enc3
        u2 = self.up2(u3, c2)          # 128×128, skip from enc2
        u1 = self.up1(u2, c1)          # 256×256, skip from enc1
        
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
