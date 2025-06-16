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
    Allows passing a pre-loaded model (from checkpoint) in the constructor.
    """
    def __init__(self, pretrained_model=None):
        super().__init__()
        if pretrained_model is not None:
            model = pretrained_model
        else:
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
    Allows passing a pre-loaded model (from checkpoint) in the constructor.
    """
    def __init__(self, pretrained_model=None):
        super().__init__()
        if pretrained_model is not None:
            model = pretrained_model
        else:
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

import numpy as np
from scipy.ndimage import label, find_objects

def estimate_bounding_box_from_activation_(activation_map, threshold_ratio=0.5):
    """
    Estimate a bounding box around the max activation using a threshold and connected components.

    Args:
        activation_map (torch.Tensor): 2D activation map.
        threshold_ratio (float): Ratio of the max value to use as threshold (e.g., 0.5).

    Returns:
        (min_col, min_row, max_col, max_row): Bounding box coordinates in activation map space.
    """
    act_np = activation_map.cpu().numpy()
    max_val = act_np.max()
    threshold = max_val * threshold_ratio
    mask = act_np >= threshold

    # Label connected components
    labeled, num_features = label(mask)
    max_pos = np.unravel_index(np.argmax(act_np), act_np.shape)
    label_of_max = labeled[max_pos]

    if label_of_max == 0:
        # fallback: just return the max pixel as a 1x1 box
        return (max_pos[1], max_pos[0], max_pos[1], max_pos[0])

    # Find the bounding box of the component containing the max
    slices = find_objects(labeled == label_of_max)[0]
    min_row, max_row = slices[0].start, slices[0].stop - 1
    min_col, max_col = slices[1].start, slices[1].stop - 1
    return (min_col, min_row, max_col, max_row)


import numpy as np
from scipy.ndimage import label, find_objects

def estimate_bounding_box_from_activation(activation_map, threshold_ratio=0.5):
    """
    Estimate a bounding box around the max activation using a threshold and connected components.
    The bounding box is adjusted so that the max activation is the centroid of the box (ou o mais próximo possível).

    Args:
        activation_map (torch.Tensor): 2D activation map.
        threshold_ratio (float): Ratio of the max value to use as threshold (e.g., 0.5).

    Returns:
        (min_col, min_row, max_col, max_row): Bounding box coordinates in activation map space.
    """
    act_np = activation_map.cpu().numpy()
    max_val = act_np.max()
    threshold = max_val * threshold_ratio
    mask = act_np >= threshold

    # Label connected components
    labeled, num_features = label(mask)
    max_pos = np.unravel_index(np.argmax(act_np), act_np.shape)
    label_of_max = labeled[max_pos]

    if label_of_max == 0:
        # fallback: just return the max pixel as a 1x1 box
        return (max_pos[1], max_pos[0], max_pos[1], max_pos[0])

    # Find the bounding box of the component containing the max
    slices = find_objects(labeled == label_of_max)[0]
    min_row, max_row = slices[0].start, slices[0].stop - 1
    min_col, max_col = slices[1].start, slices[1].stop - 1

    # Ajustar a bounding box para que o máximo fique no centroide
    box_height = max_row - min_row + 1
    box_width = max_col - min_col + 1
    cy, cx = max_pos
    half_h = box_height // 2
    half_w = box_width // 2
    new_min_row = max(cy - half_h, 0)
    new_max_row = min(cy + half_h, act_np.shape[0] - 1)
    new_min_col = max(cx - half_w, 0)
    new_max_col = min(cx + half_w, act_np.shape[1] - 1)
    # Ajustar se a caixa ficou menor por causa das bordas
    if (new_max_row - new_min_row + 1) < box_height:
        if new_min_row == 0:
            new_max_row = min(new_min_row + box_height - 1, act_np.shape[0] - 1)
        else:
            new_min_row = max(new_max_row - box_height + 1, 0)
    if (new_max_col - new_min_col + 1) < box_width:
        if new_min_col == 0:
            new_max_col = min(new_min_col + box_width - 1, act_np.shape[1] - 1)
        else:
            new_min_col = max(new_max_col - box_width + 1, 0)
    return (new_min_col, new_min_row, new_max_col, new_max_row)