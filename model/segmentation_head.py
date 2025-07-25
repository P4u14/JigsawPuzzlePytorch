import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=1, upsample_factor=32):
        super().__init__()
        # map features -> logits
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.upsample_factor = upsample_factor

    def forward(self, x):
        logits = self.classifier(x)
        return F.interpolate(
            logits,
            scale_factor=self.upsample_factor,
            mode='bilinear',
            align_corners=False
        )