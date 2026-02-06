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
    """
    def __init__(self, backbone_name='densenet121', output_dim=1024, pretrained=True):
        """
        Initializes the DenseNetFeatureExtractor.

        Args:
            backbone_name (str): The name of the backbone to use. Defaults to 'densenet121'.
            output_dim (int): The desired dimension of the output feature vector. Defaults to 1024.
            pretrained (bool): Whether to load pre-trained ImageNet weights. Defaults to True.
        """
        super().__init__()

        # 1. Load the specified DenseNet model
        if backbone_name == 'densenet121':
            # Logic matches your ResNet reference: map bool -> weights object
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            original_model = models.densenet121(weights=weights)
        else:
            raise ValueError("Unsupported backbone_name. Currently only 'densenet121' is supported.")
        features = original_model.features
        
        # DenseNet-121 outputs 1024 features
        self.initial_layers = nn.Sequential(*list(features.children())[:4])

        
        # Block 1 + Transition 1
        self.cbam1 = CBAMBlock(channels=64)
        self.block1 = features.denseblock1
        self.trans1 = features.transition1

        
        # Block 2 + Transition 2
        self.cbam2 = CBAMBlock(channels=features.transition1.conv.out_channels)
        self.block2 = features.denseblock2
        self.trans2 = features.transition2

        
        # Block 3 + Transition 3
        self.cbam3 = CBAMBlock(channels=features.transition2.conv.out_channels)        
        self.block3 = features.denseblock3
        self.trans3 = features.transition3
        
        # Block 4 + Final Norm
        self.cbam4 = CBAMBlock(channels=features.transition3.conv.out_channels)
        self.block4 = features.denseblock4
        self.norm5 = features.norm5
        
        # 3. Global Pooling & Embedding
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, output_dim) if output_dim != 1024 else nn.Identity()

    def forward(self, x):
        """
        Performs the forward pass to extract features.
        """
        # Pass input through the DenseNet backbone
        x = self.initial_layers(x)
        
        x = self.cbam1(x)
        x = self.block1(x)
        x = self.trans1(x)
        
        x = self.cbam2(x)  
        x = self.block2(x)
        x = self.trans2(x)
        
        x = self.cbam3(x)
        x = self.block3(x)
        x = self.trans3(x)
        
        x = self.cbam4(x)
        x = self.block4(x)
        x = self.norm5(x)
 
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x