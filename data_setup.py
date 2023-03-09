"""
Contains functionality for creating PyTorch DataLoaders for 
v data.
"""
import os
import cv2
import math
import torch
import skimage.io
import numpy as np
import albumentations as A

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from enum import Enum

NUM_WORKERS = 0
# NUM_WORKERS = os.cpu_count()

#Create Patches from the original images
def create_patches(image_dir, patch_size, target_dir):
    # Make target directory
    target_dir_path = Path(target_dir)
    target_imgs_path, target_masks_path = Path(target_dir+'/'+'imgs/'), Path(target_dir+'/'+'masks/')
    target_dir_path.mkdir(parents=True, exist_ok=True)
    target_imgs_path.mkdir(parents=True, exist_ok=True)
    target_masks_path.mkdir(parents=True, exist_ok=True)

    images_index, masks_index = 0, 0

    for path, _, _ in sorted(os.walk(image_dir)):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':
            images = sorted(os.listdir(path))
            for _, image_name in enumerate(images):
                if image_name.endswith(".jpg"):
                    image = cv2.imread(path+"/"+image_name)
                    size_X, size_Y = math.ceil(image.shape[1]/patch_size), math.ceil(image.shape[0]/patch_size)
                    pad_X, pad_Y = (patch_size * size_X - image.shape[1]) / (size_X - 1), (patch_size * size_Y - image.shape[0]) / (size_Y - 1)
                    image = Image.fromarray(image)
                    top = 0
                    for _ in range(size_Y):
                        left = 0
                        for _ in range(size_X):
                            crop_image = transforms.functional.crop(image, top, left, patch_size, patch_size)
                            crop_image = np.array(crop_image)
                            cv2.imwrite(f"{target_imgs_path}/image"+str(images_index).zfill(4)+".jpg", crop_image)
                            images_index += 1
                            left = left + patch_size - pad_X
                        top = top + patch_size - pad_Y
        
        if dirname == 'masks':
            images = sorted(os.listdir(path))
            for _, image_name in enumerate(images):
                if image_name.endswith(".png"):
                    image = cv2.imread(path+"/"+image_name)
                    size_X, size_Y = math.ceil(image.shape[1]/patch_size), math.ceil(image.shape[0]/patch_size)
                    pad_X, pad_Y = (patch_size * size_X - image.shape[1]) / (size_X - 1), (patch_size * size_Y - image.shape[0]) / (size_Y - 1)
                    image = Image.fromarray(image)
                    top = 0
                    for _ in range(size_Y):
                        left = 0
                        for _ in range(size_X):
                            crop_image = transforms.functional.crop(image, top, left, patch_size, patch_size)
                            crop_image = np.array(crop_image)
                            cv2.imwrite(f"{target_masks_path}/image"+str(masks_index).zfill(4)+".png", crop_image)
                            masks_index += 1
                            left = left + patch_size - pad_X
                        top = top + patch_size - pad_Y
    
    return None

# Mask color codes
class MaskColorMap(Enum):
    Unlabelled = (155, 155, 155)
    Building = (60, 16, 152)
    Land = (132, 41, 246)
    Road = (110, 193, 228)
    Vegetation = (254, 221, 58)
    Water = (226, 169, 41)

# One-hot encode masks
def one_hot_encode_masks(masks, num_classes):

    img_height, img_width, img_channels = masks.shape

        # create new mask of zeros
    encoded_image = np.zeros((img_height, img_width, 1)).astype(int)

    for j, cls in enumerate(MaskColorMap):
        encoded_image[np.all(masks == cls.value, axis=-1)] = j

    # return one-hot encoded labels
    encoded_image = np.reshape(np.eye(num_classes, dtype=int)[encoded_image],(img_height,img_width,num_classes))

    return encoded_image

# Calculate the means and stds of the dataset
def get_means_stds(output_dir):

    train_data = datasets.ImageFolder(root = output_dir+'/train', transform = transforms.ToTensor())

    means = torch.zeros(3)
    stds = torch.zeros(3)

    for img, label in train_data:
        means += torch.mean(img, dim = (1,2))
        stds += torch.std(img, dim = (1,2))

        means /= len(train_data)
        stds /= len(train_data)
    
    return means, stds

# Create data augmentation
def get_transforms(means=None, stds=None):

    data_augmentation = {
    'train': A.Compose([
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=means, std=stds),
    ]),
    'val': A.Compose([
        A.Normalize(mean=means, std=stds),
    ]),
    'test': A.Compose([
        A.Normalize(mean=means, std=stds),
    ]),}

    return data_augmentation

class SemanticSegmentationDataset(torch.utils.data.Dataset):
    """Semantic Segmentation Dataset"""

    def __init__(self, image_dir, mask_dir, image_names, mask_names, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            image_names (list): List of image names.
            mask_names (list): List of mask names.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.mask_names = mask_names
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_names[idx])

        image = skimage.io.imread(img_name)
        mask = skimage.io.imread(mask_name)

        # One-hot encoding
        mask = one_hot_encode_masks(mask, 6)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        tenn = transforms.ToTensor()
        image = tenn(image)
        mask = tenn(mask)
        
        return image, mask

def create_dataloaders(
    output_dir: str, 
    data_augmentation: dict, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
    image_datasets = {x: SemanticSegmentationDataset(image_dir=os.path.join(output_dir, x, 'imgs'),
                                                     mask_dir=os.path.join(output_dir, x, 'masks'), 
                                                     image_names=sorted(os.listdir(os.path.join(output_dir, x, 'imgs'))),
                                                     mask_names=sorted(os.listdir(os.path.join(output_dir, x, 'masks'))),
                                                     transform=data_augmentation[x]) for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], 
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=True, 
                                 drop_last=True) for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    return dataloaders, dataset_sizes