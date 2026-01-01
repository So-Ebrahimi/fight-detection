import torch
import torch.nn as nn
import cv2
import os

class ResidualBlock3D(nn.Module):
    """3D Residual block for better gradient flow"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection - needed if stride changes dimensions or channels differ
        self.use_skip = (stride != 1) or (in_channels != out_channels)
        if self.use_skip:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.skip = None
    
    def forward(self, x):
        residual = self.skip(x) if self.use_skip else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class FightModel(nn.Module):
    def __init__(self, num_frames=16, num_classes=2):
        super(FightModel, self).__init__()
        self.num_frames = num_frames
        
        # Initial 3D CNN layer
        self.conv3d_1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn3d_1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool3d_1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Residual blocks for deeper feature extraction
        self.res_block1 = ResidualBlock3D(64, 64)
        self.res_block2 = ResidualBlock3D(64, 128, stride=(2, 2, 2))
        self.res_block3 = ResidualBlock3D(128, 128)
        self.res_block4 = ResidualBlock3D(128, 256, stride=(2, 2, 2))
        self.res_block5 = ResidualBlock3D(256, 256)
        
        # Additional conv layers for temporal modeling
        self.conv3d_temporal = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_temporal = nn.BatchNorm3d(512)
        self.pool3d_temporal = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classifier with multiple layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Input shape: [batch, frames, channels, height, width]
        # Convert to: [batch, channels, frames, height, width] for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        
        # Initial convolution
        x = self.relu(self.bn3d_1(self.conv3d_1(x)))
        x = self.pool3d_1(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        
        # Temporal modeling
        x = self.relu(self.bn3d_temporal(self.conv3d_temporal(x)))
        x = self.pool3d_temporal(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

