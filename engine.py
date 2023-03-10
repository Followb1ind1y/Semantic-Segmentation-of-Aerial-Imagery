"""
Contains functions for training and testing a PyTorch model.
"""

import torch
import time
import utils
import numpy as np

from tqdm import trange

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 dataloaders: torch.utils.data.DataLoader,
                 epochs: int, 
                 metric: torch.nn.Module, 
                 criterion: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 save_dir: str,
                 device: torch.device):
        
        self.model = model
        self.train_dataloader = dataloaders['train']
        self.val_dataloader = dataloaders['val']
        self.epoch = 0
        self.epochs = epochs
        self.metric = metric
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.device = device

        # Create empty results dictionary
        self.results = {"train_loss": [],
                        "train_iou": [],
                        "val_loss": [],
                        "val_iou": []
                        }
        
    def train_model(self):
        """
        Train the Model.
        """
        start_time = time.time()

        progressbar = trange(self.epochs, desc="Progress")
        for _ in progressbar:
            # Epoch counter
            self.epoch += 1

            # Training block
            self.train_epoch()

            # Validation block
            self.val_epoch()
            print(f'\nEpoch {self.epoch}: Train loss: {self.results["train_loss"][-1]} Train iou: {self.results["train_iou"][-1]} Val loss: {self.results["val_loss"][-1]} Val iou: {self.results["val_iou"][-1]}')

            # Save checkpoints every epoch
            utils.save_model(self.model, self.save_dir, self.epoch)

        time_elapsed = time.time() - start_time
        print('\n')
        print('-' * 20)
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # plot training curve
        utils.plot_curve(results=self.results, epochs=self.epochs)

        return self.results

    def train_epoch(self):
        """
        Training Mode
        """
        self.model.train() # training mode
        running_ious, running_losses = [], []

        for x, y in self.train_dataloader:
            # Send to device (GPU or CPU)
            inputs = x.to(self.device)
            targets = y.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward - track history if only in train
            outputs = self.model(inputs)
            # Calculate the loss
            loss = self.criterion(outputs, targets)
            loss_value = loss.item()
            running_losses.append(loss_value)

            # Calculate the iou
            iou_value = self.metric(outputs, targets)
            running_ious.append(iou_value)

            # Backward pass
            loss.backward()
            # Update the parameters
            self.optimizer.step()

        self.scheduler.step()
        self.results["train_loss"].append(np.mean(running_losses))
        self.results["train_iou"].append(np.mean(running_ious))

    def val_epoch(self):
        """
        Validation Mode
        """
        self.model.eval() # Validation mode
        running_ious, running_losses = [], []

        for x, y in self.val_dataloader:
            # Send to device (GPU or CPU)
            inputs = x.to(self.device)
            targets = y.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                # Calculate the loss
                loss = self.criterion(outputs, targets)
                loss_value = loss.item()
                running_losses.append(loss_value)

                # Calculate the iou
                iou_value = self.metric(outputs, targets)
                running_ious.append(iou_value)

        self.results["val_loss"].append(np.mean(running_losses))
        self.results["val_iou"].append(np.mean(running_ious))