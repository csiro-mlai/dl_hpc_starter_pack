from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from typing import Dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class MLPNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(LightningModule):
    def __init__(
        self,
        mbatch_size: int,
        lr: float,
        prefetch_factor: int = 5,
        num_workers: int = 1,
        **kwargs,
    ):
        """
        Argument/s:
            mbatch_size - mini-batch size.
            lr - initial learning rate.
            prefetch_factor - no. of samples pre-loaded by each worker, i.e.
                prefetch_factor multiplied by num_workers samples are prefetched
                over all workers.
            num_workers - number of subprocesses to use for DataLoader. 0 means
                that the data will be loaded in the main process.
            kwargs - keyword arguments.
        """
        super(MLP, self).__init__()
        self.save_hyperparameters()

        self.mbatch_size = mbatch_size
        self.lr = lr
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # Model
        self.model = MLPNet()

        # Loss
        self.loss = nn.MSELoss()

    def setup(self, stage=None):
        """
        Dataset preparation.

        Argument/s:
            stage - either 'fit' (training & validation sets) or 'test'
                (test set).
        """

        # Assign training set
        if stage == 'fit' or stage is None:
            train_x = torch.rand(1000) * 2 * torch.pi
            train_y = torch.cos(train_x)
            self.train_set = [
                (torch.unsqueeze(x, 0), torch.unsqueeze(y, 0))
                 for x, y in zip(train_x, train_y)]

            val_x = torch.rand(100) * 2 * torch.pi
            val_y = torch.cos(val_x)
            self.val_set = list(zip(val_x, val_y))
            self.val_set = [
                (torch.unsqueeze(x, 0), torch.unsqueeze(y, 0))
                 for x, y in zip(val_x, val_y)]

        if stage == 'test' or stage is None:
            test_x = torch.rand(1000) * 2 * torch.pi
            test_y = torch.cos(test_x)
            self.test_set = [
                (torch.unsqueeze(x, 0), torch.unsqueeze(y, 0))
                 for x, y in zip(test_x, test_y)]

    def train_dataloader(self, shuffle=True):
        """
        Training set DataLoader.

        Argument/s:
            shuffle - shuffle the order of the examples.

        Returns:
            DataLoader.
        """
        return DataLoader(
            self.train_set,
            batch_size=self.mbatch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            
        )

    def val_dataloader(self):
        """
        Validation set DataLoader.

        Argument/s:
            shuffle - shuffle the order of the examples.

        Returns:
            DataLoader.
        """
        return DataLoader(
            self.val_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        """
        Test set DataLoader.

        Argument/s:
            shuffle - shuffle the order of the examples.

        Returns:
            DataLoader.
        """
        return DataLoader(
            self.test_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
        )

    def configure_optimizers(self):
        """
        Define optimiser/s and learning rate scheduler/s. A good source for understanding
        how the pytorch learning rate schedulers work can be found at:
            https://www.programmersought.com/article/12164650026/

        Returns:
            The optimiser/s.
        """
        optimiser = {'optimizer': torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)}
        return optimiser

    def forward(self, x):
        """
        Forward propagation over the torch.mm.Module attributes of self.

        Argument/s:
            images - a mini-batch of images from the dataset.

        Returns:
            A dictionary containing the outputs from the network/s.
        """
        return {'cosine': self.model(x)}

    def training_step(self, batch, batch_idx):
        """
        Training step (the training loss needs to be returned).

        Argument/s:
            batch - mini-batch from the training set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.

        Returns:
            loss - training loss for the mini-batch.
        """

        # Mini-batch of examples
        x, y = batch

        # Inference
        y_hat = self(x)

        # Loss
        loss = self.loss(y_hat['cosine'], y)

        # Logging
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=x.size()[0])

        # Update and log scores for each validation metric.
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Argument/s:
            batch - mini-batch from the validation set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.
        """

        # Mini-batch of examples
        x, y = batch

        # Inference
        y_hat = self(x)

        # Loss
        loss = self.loss(y_hat['cosine'], y)

        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=x.size()[0])

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Argument/s:
            batch - mini-batch from the test set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.
        """

        # Mini-batch of examples
        x, y = batch

        # Inference
        y_hat = self(x)

        # Compute metric scores
        loss = self.loss(torch.argmax(y_hat['cosine'], dim=1), y)

        # Log the test accuracy
        self.log('test_acc', loss, on_step=False, on_epoch=True, batch_size=x.size()[0])

