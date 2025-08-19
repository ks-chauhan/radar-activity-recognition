"""
Radar Activity Recognition Model Architecture
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


class PowerfulRadarNet128(nn.Module):
    """
    Powerful RadarNet model with ResNet18 backbone for 128x128 RD Map images
    
    Args:
        num_classes (int): Number of activity classes
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, num_classes=11, dropout_rate=0.3):
        super().__init__()
        
        # Pretrained ResNet18 as backbone
        resnet = resnet18(weights='IMAGENET1K_V1')
        
        # Keep all layers except final FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Direct classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/4),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights with Kaiming normal"""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, 128, 128)
        
        Returns:
            torch.Tensor: Output logits of shape (B, num_classes)
        """
        features = self.backbone(x)  # (B, 512, 1, 1) for 128x128 input
        return self.classifier(features)
    
    def get_feature_maps(self, x):
        """
        Extract feature maps for visualization
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            dict: Dictionary containing feature maps from different layers
        """
        features = {}
        
        # Extract features from different ResNet layers
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 5, 6, 7]:  # Conv layers
                features[f'layer_{i}'] = x
        
        return features