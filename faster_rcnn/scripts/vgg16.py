import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    # Just feature extractor layer and how it can be used generate feature maps from convolutional layers
    def __init__(self):
        super().__init__()
        # Block 1
        self._block1_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding='same')
        self._block1_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self._block1_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self._block2_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        self._block2_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self._block2_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self._block3_conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same')
        self._block3_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same')
        self._block3_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same')
        self._block3_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self._block4_conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same')
        self._block4_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same')
        self._block4_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same')
        self._block4_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5
        self._block5_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same')
        self._block5_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same')
        self._block5_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same')

        # Freeze first 2 blocks
        self._block1_conv1.weight.requires_grad = False
        self._block1_conv1.bias.requires_grad = False
        self._block1_conv2.weight.requires_grad = False
        self._block1_conv2.bias.requires_grad = False

        self._block2_conv1.weight.requires_grad = False
        self._block2_conv1.bias.requires_grad = False
        self._block2_conv2.weight.requires_grad = False
        self._block2_conv2.bias.requires_grad = False

    def forward(self, x):
        x = F.relu(self._block1_conv1(x))
        x = F.relu(self._block1_conv2(x))
        x = self._block1_pool1(x)

        x = F.relu(self._block2_conv1(x))
        x = F.relu(self._block2_conv2(x))
        x = self._block2_pool1(x)

        x = F.relu(self._block3_conv1(x))
        x = F.relu(self._block3_conv2(x))
        x = F.relu(self._block3_conv3(x))
        x = self._block3_pool1(x)

        x = F.relu(self._block4_conv1(x))
        x = F.relu(self._block4_conv2(x))
        x = F.relu(self._block4_conv3(x))
        x = self._block4_pool1(x)

        x = F.relu(self._block5_conv1(x))
        x = F.relu(self._block5_conv2(x))
        x = F.relu(self._block5_conv3(x))

        return x
    
class PoolToFeatureVector(nn.Module):
    def __init__(self, dropout_probability):
        super().__init__()
        self._fc1 = nn.Linear(in_features=512*7*7, out_features=4096)
        self._fc2 = nn.Linear(in_features=4096, out_features=4096)

        self.dropout1 = nn.Dropout(p = dropout_probability)
        self.dropout2 = nn.Dropout(p = dropout_probability)
    
    def forawrd(self, rois):
        rois = rois.reshape((rois.shape[0], 512*7*7))
        rois = F.relu(self._fc1(rois))
        rois = self.dropout1(rois)
        rois = F.relu(self._fc2(rois))
        rois = self.dropout2(rois)
        return rois