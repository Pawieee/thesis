import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =============================================================================
# CBAM ATTENTION MODULES
# =============================================================================

class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention Module.

    Computes channel-wise attention weights by aggregating spatial information
    via both average pooling and max pooling, then passing through a shared MLP.
    The two outputs are summed and passed through a sigmoid to produce
    per-channel attention weights in [0, 1].

    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

    Args:
        channels (int): Number of input channels.
        ratio (int): Channel reduction ratio for the MLP bottleneck. Default 8.
    """
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP implemented as 1x1 convolutions
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    CBAM Spatial Attention Module.

    Computes spatial attention weights by aggregating channel information
    via average pooling and max pooling along the channel dimension,
    concatenating the results, and passing through a convolutional layer.

    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

    Args:
        kernel_size (int): Convolution kernel size. Default 7 (as in paper).
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_concat))


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Sequentially applies channel attention followed by spatial attention.
    Each attention map is applied multiplicatively as a feature recalibration.

    Args:
        channels (int): Number of input channels.
        ratio (int): Channel reduction ratio for ChannelAttention MLP. Default 8.
        kernel_size (int): Kernel size for SpatialAttention conv. Default 7.
    """
    def __init__(self, channels, ratio=8, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# =============================================================================
# DENSENET-121 FEATURE EXTRACTOR
# =============================================================================

# DenseNet-121 channel counts at each CBAM insertion point.
# These are fixed architectural constants — do NOT infer at runtime.
# Source: DenseNet-121 architecture (Huang et al., CVPR 2017)
#   After initial conv (stem output, before Dense Block 1):  64
#   After Transition 1 (before Dense Block 2):              128
#   After Transition 2 (before Dense Block 3):              256
#   After Transition 3 (before Dense Block 4):              512
_DENSENET121_CBAM_CHANNELS = {
    'cbam1': 64,
    'cbam2': 128,
    'cbam3': 256,
    'cbam4': 512,
}


class DenseNetFeatureExtractor(nn.Module):
    """
    DenseNet-121 feature extractor with optional CBAM attention integration.

    Two modes are supported via the `baseline` flag:

    Baseline mode (baseline=True):
        Standard DenseNet-121 backbone with a Regularized Dense Block head.
        No CBAM modules. Produces a feature vector of size `output_dim`.
        Backbone freeze/unfreeze is handled EXTERNALLY by the training loop —
        NOT in __init__. This allows the two-phase frozen/unfrozen strategy
        to work correctly.

    Proposed mode (baseline=False):
        DenseNet-121 backbone with 4 CBAM blocks inserted before each
        Dense Block. CBAM channels are hardcoded to known DenseNet-121
        architectural constants. Produces a feature vector of size `output_dim`.

    Forward pass (proposed):
        Input → Stem → CBAM1 → Block1 → Trans1
               → CBAM2 → Block2 → Trans2
               → CBAM3 → Block3 → Trans3
               → CBAM4 → Block4 → Norm5 → ReLU
               → GlobalAvgPool → Flatten → RegularizedDenseBlock → Output

    Args:
        backbone_name (str): Backbone identifier. Only 'densenet121' supported.
        output_dim (int): Output embedding dimension. Default 1024.
        pretrained (bool): Load ImageNet pretrained weights. Default True.
        baseline (bool): Use baseline mode (no CBAM). Default False.
    """

    def __init__(self, backbone_name='densenet121', output_dim=1024,
                 pretrained=True, baseline=False):
        super().__init__()
        self.baseline = baseline

        if backbone_name != 'densenet121':
            raise ValueError(
                f"Unsupported backbone_name '{backbone_name}'. "
                "Only 'densenet121' is currently supported."
            )

        # Load DenseNet-121 with or without ImageNet weights
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        original_model = models.densenet121(weights=weights)
        features = original_model.features

        if self.baseline:
            # ── Baseline Mode ──────────────────────────────────────────────
            # Full DenseNet-121 backbone — no CBAM.
            # NOTE: Backbone is NOT frozen here. Freezing and unfreezing
            # must be handled by the training loop (freeze_backbone /
            # unfreeze_backbone), consistent with the proposed model.
            self.backbone = features

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            # Regularized Dense Block — matches proposed model naming.
            # Named consistently so optimizer param groups can reference
            # 'regularized_dense_block' regardless of mode.
            self.regularized_dense_block = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(p=0.5),
                nn.Linear(1024, output_dim)
            )

        else:
            # ── Proposed Mode ──────────────────────────────────────────────
            # DenseNet-121 with CBAM blocks inserted before each Dense Block.

            # Stem: Conv7x7 + BN + ReLU + MaxPool (first 4 children of features)
            self.initial_layers = nn.Sequential(*list(features.children())[:4])

            # CBAM channels are hardcoded to DenseNet-121 architectural constants.
            # Do NOT infer these from the loaded model at runtime — that approach
            # is brittle across torchvision versions.
            self.cbam1 = CBAMBlock(channels=_DENSENET121_CBAM_CHANNELS['cbam1'])  # 64
            self.block1 = features.denseblock1
            self.trans1 = features.transition1

            self.cbam2 = CBAMBlock(channels=_DENSENET121_CBAM_CHANNELS['cbam2'])  # 128
            self.block2 = features.denseblock2
            self.trans2 = features.transition2

            self.cbam3 = CBAMBlock(channels=_DENSENET121_CBAM_CHANNELS['cbam3'])  # 256
            self.block3 = features.denseblock3
            self.trans3 = features.transition3

            self.cbam4 = CBAMBlock(channels=_DENSENET121_CBAM_CHANNELS['cbam4'])  # 512
            self.block4 = features.denseblock4

            # Final BatchNorm from DenseNet-121 (norm5)
            self.norm5 = features.norm5

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            # Regularized Dense Block: BN → Dropout → Linear projection
            self.regularized_dense_block = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(p=0.5),
                nn.Linear(1024, output_dim)
            )

    def forward(self, x):
        """
        Forward pass through the feature extractor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Feature embedding of shape [B, output_dim].
        """
        if self.baseline:
            x = self.backbone(x)
            x = F.relu(x, inplace=True)     # Final activation after DenseNet norm5
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.regularized_dense_block(x)
            return x

        else:
            # Stem
            x = self.initial_layers(x)

            # CBAM1 → Dense Block 1 → Transition 1
            x = self.cbam1(x)
            x = self.block1(x)
            x = self.trans1(x)

            # CBAM2 → Dense Block 2 → Transition 2
            x = self.cbam2(x)
            x = self.block2(x)
            x = self.trans2(x)

            # CBAM3 → Dense Block 3 → Transition 3
            x = self.cbam3(x)
            x = self.block3(x)
            x = self.trans3(x)

            # CBAM4 → Dense Block 4 → Final BN → ReLU
            x = self.cbam4(x)
            x = self.block4(x)
            x = self.norm5(x)
            x = F.relu(x, inplace=True)     # Required activation after norm5

            # Global Average Pool → Flatten → Regularized Dense Block
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.regularized_dense_block(x)
            return x

    def get_backbone_params(self):
        """
        Returns backbone parameter groups for use in optimizer construction.

        Baseline:  all parameters in self.backbone
        Proposed:  initial_layers + all dense blocks + transitions + norm5

        Returns:
            list: List of parameter tensors belonging to the backbone.
        """
        if self.baseline:
            return list(self.backbone.parameters())
        else:
            backbone_modules = [
                self.initial_layers,
                self.block1, self.trans1,
                self.block2, self.trans2,
                self.block3, self.trans3,
                self.block4, self.norm5,
            ]
            params = []
            for module in backbone_modules:
                params.extend(module.parameters())
            return params

    def get_head_params(self):
        """
        Returns non-backbone (head) parameter groups for optimizer construction.

        Baseline:  regularized_dense_block
        Proposed:  cbam1-4 + regularized_dense_block

        Returns:
            list: List of parameter tensors belonging to the head.
        """
        if self.baseline:
            return list(self.regularized_dense_block.parameters())
        else:
            head_modules = [
                self.cbam1, self.cbam2, self.cbam3, self.cbam4,
                self.regularized_dense_block,
            ]
            params = []
            for module in head_modules:
                params.extend(module.parameters())
            return params