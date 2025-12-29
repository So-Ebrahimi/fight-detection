import torch
import torch.nn as nn
import cv2
import os

class FightModel(nn.Module):
    def __init__(self, num_frames=16, num_classes=2):
        super(FightModel, self).__init__()
        self.num_frames = num_frames
        
        # 3D CNN for spatiotemporal features
        self.conv3d_1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(32)
        self.pool3d_1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(64)
        self.pool3d_2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d_3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_3 = nn.BatchNorm3d(128)
        self.pool3d_3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: [batch, frames, channels, height, width]
        # Need to convert to: [batch, channels, frames, height, width] for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.relu(self.bn3d_1(self.conv3d_1(x)))
        x = self.pool3d_1(x)
        
        x = self.relu(self.bn3d_2(self.conv3d_2(x)))
        x = self.pool3d_2(x)
        
        x = self.relu(self.bn3d_3(self.conv3d_3(x)))
        x = self.pool3d_3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

