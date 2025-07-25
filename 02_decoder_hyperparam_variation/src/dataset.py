import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as transf
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors


class Subset(Dataset):
    def __init__(self, ds, indices, transform=None):
        """
        Subset of a dataset with optional transform.
        Args:
            ds (Dataset): Original dataset.
            indices (list): Indices to include in the subset.
            transform (callable, optional): Transform to apply to each sample.
        """
        self.ds = ds
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        """
        Get item from subset by index.
        Args:
            idx (int): Index in the subset.
        Returns:
            tuple: (image, target) after optional transform.
        """
        img, target = self.ds[self.indices[idx]]
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        """
        Returns the number of items in the subset.
        Returns:
            int: Number of items.
        """
        return len(self.indices)


class OxfordIIITPetSeg(Dataset):
    """
    Oxford-IIIT Pet Segmentation Dataset.
    """

    def __init__(self, root, transforms=None, ignore_val=2):
        """
        Initialize Oxford-IIIT Pet Segmentation dataset.
        Args:
            root (str or Path): Root directory of the dataset.
            transforms (callable, optional): Transformations to apply to images and labels.
            ignore_val (int, optional): Value to assign to ignored pixels (border).
        """
        root = Path(root)
        images_folder = root / "images"
        segs_folder = root / "annotations/trimaps"
        anns_file = root / "annotations/list.txt"

        images = []
        segs = []
        for line in open(anns_file).read().splitlines():
            if line[0] != "#":  # Remove comentários do arquivo
                name, class_id, species_id, breed_id = line.strip().split()
                images.append(images_folder / f"{name}.jpg")
                segs.append(segs_folder / f"{name}.png")

        self.images = images
        self.segs = segs
        self.transforms = transforms
        self.ignore_val = ignore_val

    def __getitem__(self, idx, apply_transform=True):
        """
        Get image and segmentation mask by index.
        Args:
            idx (int): Index of the sample.
            apply_transform (bool, optional): Whether to apply transforms.
        Returns:
            tuple: (image, target) where image is RGB and target is mask.
        """
        image = Image.open(self.images[idx]).convert("RGB")
        target_or = Image.open(self.segs[idx])
        target_np = np.array(target_or)
        target_np[target_np == 2] = 0
        if self.ignore_val != 3:
            target_np[target_np == 3] = self.ignore_val
        target = Image.fromarray(target_np, mode="L")
        if self.transforms and apply_transform:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Returns:
            int: Number of samples.
        """
        return len(self.images)


class TransformsTrain:
    def __init__(self, resize_size=384):
        """
        Transformations for training images and masks.
        Args:
            resize_size (int, optional): Size to resize images and masks.
        """
        transforms = transf.Compose(
            [
                transf.PILToTensor(),
                transf.RandomResizedCrop(
                    size=(resize_size, resize_size),
                    scale=(0.5, 1.0),
                    ratio=(0.9, 1.1),
                    antialias=True,
                ),
                # transf.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.01),
                transf.RandomHorizontalFlip(),
                transf.ToDtype(
                    {tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}
                ),
                transf.Normalize(mean=(122.7, 114.6, 100.9), std=(59.2, 58.4, 59.0)),
            ]
        )
        self.transforms = transforms

    def __call__(self, img, target):
        """
        Apply training transformations to image and mask.
        Args:
            img (PIL.Image): Input image.
            target (PIL.Image): Segmentation mask.
        Returns:
            tuple: (transformed image, transformed mask)
        """
        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)
        img, target = self.transforms(img, target)
        img = img.data
        target = target.data
        target = target.squeeze()
        return img, target


class TransformsEval:
    def __init__(self, resize_size=384):
        """
        Transformations for evaluation images and masks.
        Args:
            resize_size (int, optional): Size to resize images and masks.
        """
        transforms = transf.Compose(
            [
                transf.PILToTensor(),
                transf.Resize(size=resize_size, antialias=True),
                transf.ToDtype(
                    {tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}
                ),
                transf.Normalize(mean=(122.7, 114.6, 100.9), std=(59.2, 58.4, 59.0)),
            ]
        )
        self.transforms = transforms

    def __call__(self, img, target):
        """
        Apply evaluation transformations to image and mask.
        Args:
            img (PIL.Image): Input image.
            target (PIL.Image): Segmentation mask.
        Returns:
            tuple: (transformed image, transformed mask)
        """
        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)
        img, target = self.transforms(img, target)
        img = img.data
        target = target.data
        target = target.squeeze()
        return img, target


def cat_list(images, fill_value=0):
    """
    Concatenate a list of images or masks into a batch tensor, padding as needed.
    Args:
        images (list of torch.Tensor): List of images or masks.
        fill_value (int, optional): Value to use for padding.
    Returns:
        torch.Tensor: Batched tensor of images or masks.
    """
    is_target = images[0].ndim == 2
    num_rows, num_cols = zip(*[img.shape[-2:] for img in images])
    r_max, c_max = max(num_rows), max(num_cols)
    if is_target:
        batch_shape = (len(images), r_max, c_max)
    else:
        batch_shape = (len(images), 3, r_max, c_max)
    batched_imgs = torch.full(batch_shape, fill_value, dtype=images[0].dtype)
    for idx in range(len(images)):
        img = images[idx]
        if is_target:
            batched_imgs[idx, : img.shape[0], : img.shape[1]] = img
        else:
            batched_imgs[idx, :, : img.shape[1], : img.shape[2]] = img
    return batched_imgs


def collate_fn(batch, img_fill=0, target_fill=2):
    """
    Collate function for DataLoader to batch images and masks with padding.
    Args:
        batch (list): List of (image, target) tuples.
        img_fill (int, optional): Fill value for images.
        target_fill (int, optional): Fill value for targets.
    Returns:
        tuple: (batched_imgs, batched_targets)
    """
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=img_fill)
    batched_targets = cat_list(targets, fill_value=target_fill)
    return batched_imgs, batched_targets


def unormalize(img):
    """
    Unnormalize an image tensor to original pixel values.
    Args:
        img (torch.Tensor): Normalized image tensor (C, H, W).
    Returns:
        torch.Tensor: Unnormalized image tensor (H, W, C) as uint8.
    """
    img = img.permute(1, 2, 0)
    mean = torch.tensor([122.7, 114.6, 100.9])
    std = torch.tensor([59.2, 58.4, 59.0])
    img = img * std + mean
    img = img.to(torch.uint8)
    return img


def get_dataset(root, split=0.2, resize_size=384):
    """
    Create train and validation datasets and class weights for Oxford-IIIT Pet Segmentation.
    Args:
        root (str or Path): Root directory of the dataset.
        split (float, optional): Fraction of data for validation set.
        resize_size (int, optional): Size to resize images and masks.
    Returns:
        tuple: (ds_train, ds_valid, class_weights)
            ds_train (Subset): Training dataset subset.
            ds_valid (Subset): Validation dataset subset.
            class_weights (tuple): Class weights for loss function.
    """
    class_weights = (0.33, 0.67)
    ds = OxfordIIITPetSeg(root)
    n = len(ds)
    n_valid = int(n * split)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    ds_train = Subset(ds, indices[n_valid:], TransformsTrain(resize_size))
    ds_valid = Subset(ds, indices[:n_valid], TransformsEval(resize_size))
    return ds_train, ds_valid, class_weights
