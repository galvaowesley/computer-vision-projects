"""
Module for dog detection utilities using activation maps.
"""

import torch
from torch import nn
from torchvision import models
import torchvision.transforms.v2 as transf

class ImageTransforms:
    """
    Utility class for image transformations for dog detection.

    Methods:
        __call__(img):
            Apply transformations to the input image.
    """
    def __init__(self):
        self.transforms = transf.Compose([
            transf.PILToTensor(),
            transf.Resize(size=(2*224, 2*224), antialias=True),
            transf.ToDtype(torch.float32),
            transf.Normalize(mean=(122.7, 114.6, 100.9), std=(59.2, 58.4, 59.0))
        ])
    def __call__(self, img):
        """
        Apply transformations to the input image.

        Args:
            img (PIL.Image): Input image.
        Returns:
            torch.Tensor: Transformed image tensor.
        """
        return self.transforms(img)

class ActivationMapResNet18(nn.Module):
    """
    Model to extract activation maps before global pooling from ResNet18.

    Methods:
        forward(x):
            Forward pass to get activation map.
    """
    def __init__(self):
        super().__init__()
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.eval()
        model.avgpool = nn.Identity()
        model.fc = nn.Identity()
        self.model = model
    def forward(self, x):
        """
        Forward pass to get activation map.

        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Activation map (14x14).
        """
        x = self.model(x)
        x = x.reshape(512, 14, 14)
        return x.mean(dim=0)

class ActivationMapResNet50(nn.Module):
    """
    Model to extract activation maps before global pooling from ResNet50.

    Methods:
        forward(x):
            Forward pass to get activation map.
    """
    def __init__(self):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.eval()
        model.avgpool = nn.Identity()
        model.fc = nn.Identity()
        self.model = model
    def forward(self, x):
        """
        Forward pass to get activation map.

        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Activation map (14x14).
        """
        x = self.model(x)
        x = x.reshape(2048, 14, 14)
        return x.mean(dim=0)

def detect_max_activation(activation_map):
    """
    Detect the position of the maximum value in the activation map.

    Args:
        activation_map (torch.Tensor): Activation map (2D).
    Returns:
        tuple: (row, col) position of the maximum value.
    """
    max_idx = torch.argmax(activation_map)
    max_pos = (max_idx // activation_map.shape[1], max_idx % activation_map.shape[1])
    return max_pos

def map_activation_to_original(activation_pos, original_size, resized_size=(448, 448), map_size=(14, 14)):
    """
    Map the activation position to the original image coordinates.

    Args:
        activation_pos (tuple): (row, col) in activation map.
        original_size (tuple): (width, height) of the original image.
        resized_size (tuple): Size to which the image was resized.
        map_size (tuple): Size of the activation map.
    Returns:
        tuple: (x, y) position in the original image.
    """
    factor_x = original_size[0] / resized_size[0]
    factor_y = original_size[1] / resized_size[1]
    x = int(activation_pos[1] * (resized_size[0] / map_size[0]) * factor_x)
    y = int(activation_pos[0] * (resized_size[1] / map_size[1]) * factor_y)
    return (x, y)
