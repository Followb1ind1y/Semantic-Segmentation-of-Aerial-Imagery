"""
Trains a Semantic Segmentation of Aerial Imagery Model.
"""

import torch
import splitfolders
import torch.optim as optim
import data_setup
import segmentation_models_pytorch as smp
import utils,engine

from torch.optim import lr_scheduler

# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 16

# Create Patches from the original images
# patch_size = 224
# original_dir = 'Semantic_segmentation_dataset'
# data_setup.create_patches(image_dir=original_dir, patch_size=patch_size, target_dir='Patches')

data_dir = 'Patches'
splitfolders.ratio(data_dir, output="train_split", ratio=(0.8, 0.1, 0.1))
output_dir = 'train_split'

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Calculate the means and stds of the dataset
# means, stds = data_setup.get_means_stds(output_dir=output_dir)

# Create data augmentation
data_augmentation = data_setup.get_transforms()

# Create DataLoaders with help from data_setup.py
dataloaders, dataset_sizes = data_setup.create_dataloaders(
    output_dir=output_dir, 
    data_augmentation=data_augmentation, 
    batch_size=BATCH_SIZE
)

#  Create segmentation model
ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d'

# Create segmentation model with pretrained encoder
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=6, 
    activation=ACTIVATION,
)

## Model inItialization
model = model.to(device)
metric_DeepLab_V3 = utils.IoU()
criterion_DeepLab_V3 = utils.CategoricalCrossEntropyLoss()
optimizer_DeepLab_V3 = optim.Adam(model.parameters(), lr=0.01)
exp_lr_scheduler_DeepLab_V3 = lr_scheduler.StepLR(optimizer_DeepLab_V3, step_size=7, gamma=0.1)

# Trainer
trainer = engine.Trainer(model=model,
                  dataloaders=dataloaders,
                  epochs=5,
                  metric=metric_DeepLab_V3,
                  criterion=criterion_DeepLab_V3, 
                  optimizer=optimizer_DeepLab_V3,
                  scheduler=exp_lr_scheduler_DeepLab_V3,
                  save_dir="Model_Output",
                  device=device)

## Training process
model_results = trainer.train_model()