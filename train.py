"""
Trains a Semantic Segmentation of Aerial Imagery Model.
"""

import torch
import splitfolders
import numpy as np
import torch.optim as optim
import segmentation_models_pytorch as smp
import utils,engine,data_setup,predictions

from torch.optim import lr_scheduler

# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup hyperparameters
NUM_EPOCHS = 30
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
means, stds = data_setup.get_means_stds(output_dir=output_dir)

# Create data augmentation
data_augmentation = data_setup.get_transforms(means=means, stds=stds)

# Create DataLoaders with help from data_setup.py
dataloaders, dataset_sizes = data_setup.create_dataloaders(
    output_dir=output_dir, 
    data_augmentation=data_augmentation, 
    batch_size=BATCH_SIZE
)

#  Create segmentation model
ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d'

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=6, 
    activation=ACTIVATION,
)

## Model inItialization
model = model.to(device)
metric_UNet = utils.MeanIoU()
criterion_UNet = utils.MultiDiceLoss()
optimizer_UNet = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler_UNet = lr_scheduler.StepLR(optimizer_UNet, step_size=7, gamma=0.1)

# Trainer
trainer = engine.Trainer(model=model,
                         dataloaders=dataloaders,
                         epochs=30,
                         metric=metric_UNet,
                         criterion=criterion_UNet, 
                         optimizer=optimizer_UNet,
                         scheduler=exp_lr_scheduler_UNet,
                         save_dir="UNet_Model_Output",
                         device=device)

## Training process
model_results = trainer.train_model()

## Evaluate the model
outputs = predictions.evaluate_model(model=model, dataloaders=dataloaders['val'], 
                                     metric=metric_UNet,criterion=criterion_UNet,
                                     device=device)

## Display the predictions
images, masks = next(iter(dataloaders['test']))

idx = 0
original_image = np.transpose(images[idx])
ground_truth_mask = np.transpose(np.argmax(masks[idx], axis=0, keepdims=True))
res = predictions.predict_mask(img=images, model=model, device=device)
predicted_mask = np.transpose(np.argmax(res[idx].to('cpu'), axis=0, keepdims=True))
utils.display(original_image=original_image, ground_truth_mask=ground_truth_mask, predicted_mask=predicted_mask)