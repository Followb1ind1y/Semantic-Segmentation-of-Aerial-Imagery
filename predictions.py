"""
Utility functions to make predictions.
"""
import torch
import torch.nn.functional as F
import numpy as np

# Find the test iou of a target model
def evaluate_model(model: torch.nn.Module, 
                   dataloaders: torch.utils.data.DataLoader,
                   metric: torch.nn.Module, 
                   criterion: torch.nn.Module, 
                   device: torch.device):
    """
    Evaluate model performance on testset
    """
    model.eval()
    model.to(device)

    running_ious, running_losses = [], []

    for x, y in dataloaders:
    # Send to device (GPU or CPU)
        inputs = x.to(device)
        targets = y.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            # Calculate the loss
            loss = criterion(outputs, targets)
            loss_value = loss.item()
            running_losses.append(loss_value)

            # Calculate the iou
            iou_value = metric(outputs, targets)
            running_ious.append(iou_value.detach().cpu().numpy())
        
    mean_loss = np.mean(running_losses)
    mean_metric = np.mean(running_ious)
        
    return mean_loss, mean_metric

# Predict the masks
def predict_mask(img: torch.Tensor, 
            model: torch.nn.Module, 
            device: str):

    model.eval()
    model.to(device)

    x = img.to(device)
    with torch.no_grad():
        out = model(x)

    result = torch.softmax(out, dim=1)

    return result