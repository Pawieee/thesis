import torch
import torch.nn as nn
import torchvision.models as models


class ChannelAttention(nn.Module):
    """CBAM Channel Attention Module"""
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_concat)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channels, ratio=8, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class DenseNetFeatureExtractor(nn.Module):
    """
    A feature extractor module based on DenseNet-121 with CBAM block integration.

    This module loads a pre-trained DenseNet-121 backbone, integrates CBAM 
    attention modules, and outputs the feature vector from the penultimate layer.
    """
    def __init__(self, backbone_name='densenet121', output_dim=1024, weights='imagenet'):
        """
        Initializes the DenseNetFeatureExtractor.
        Args:
            backbone_name (str): The name of the backbone to use. Currently supports 'densenet121'.
                                 Defaults to 'densenet121'.
            output_dim (int): The desired dimension of the output feature vector.
                              DenseNet-121 outputs 1024 features by default. Defaults to 1024.
            weights (str): Pre-trained weights to load. Options: 'imagenet' or None. Defaults to 'imagenet'.
        """
        super().__init__()

        # Load pre-trained DenseNet-121 model
        weights_obj = models.DenseNet121_Weights.IMAGENET1K_V1 if weights == 'imagenet' else None
        densenet = models.densenet121(weights=weights_obj)
        
        # DenseNet-121 outputs 1024 features
        original_dim = 1024

        # Extract the features module (everything except the classifier)
        self.backbone = densenet.features
        self.output_dim = output_dim

        # Add CBAM attention module after feature extraction
        self.cbam = CBAMBlock(original_dim, ratio=8)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Add an optional linear layer if the desired output dimension
        # differs from DenseNet-121's original feature dimension (1024).
        if self.output_dim != original_dim:
            self.fc = nn.Linear(original_dim, self.output_dim)
        else:
            # If dimensions match, use Identity to avoid unnecessary layer
            self.fc = nn.Identity()


    def forward(self, x):
        """
        Performs the forward pass to extract features.

        Args:
            x (torch.Tensor): Input image tensor. Shape: (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Output feature vector. Shape: (batch_size, output_dim).
        """
        # Pass input through the DenseNet backbone
        features = self.backbone(x) # Shape: (batch_size, 1024, H, W)

        # Apply CBAM attention module
        features = self.cbam(features) # Shape: (batch_size, 1024, H, W)

        # Global average pooling
        features = self.avgpool(features) # Shape: (batch_size, 1024, 1, 1)
        
        # Flatten
        features = torch.flatten(features, 1) # Shape: (batch_size, 1024)

        # Apply the final linear layer (or Identity if dimensions match)
        output_features = self.fc(features) # Shape: (batch_size, output_dim)

        return output_features