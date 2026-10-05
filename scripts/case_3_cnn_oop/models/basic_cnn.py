# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:17:42 2026

@author: saeid
"""

import torch.nn as nn

class simpleCNN(nn.Module):
    def __init__(self, in_channels=1, dropout=0.2):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fcl = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fcl(x)
        return x