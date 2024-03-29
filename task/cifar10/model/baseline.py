from lightning import LightningModule
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from typing import Dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Baseline(LightningModule):
    """
    Baseline CIFAR-10 model.
    """

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(
        self,
        dataset_dir: str,
        mbatch_size: int,
        lr: float,
        prefetch_factor: int = 5,
        num_workers: int = 1,
        **kwargs,
    ):
        """
        Argument/s:
            dataset_dir - dataset directory.
            mbatch_size - mini-batch size.
            lr - initial learning rate.
            prefetch_factor - no. of samples pre-loaded by each worker, i.e.
                prefetch_factor multiplied by num_workers samples are prefetched
                over all workers.
            num_workers - number of subprocesses to use for DataLoader. 0 means
                that the data will be loaded in the main process.
            kwargs - keyword arguments.
        """
        super(Baseline, self).__init__()
        self.save_hyperparameters()

        self.dataset_dir = dataset_dir
        self.mbatch_size = mbatch_size
        self.lr = lr
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # Model
        self.model = CNN()

        # Loss
        self.loss = nn.CrossEntropyLoss()

        # Dataset paths
        self.dataset_dir = os.path.join(self.dataset_dir, 'cifar10')

        # Image transformations
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomRotation(5),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        )

        self.test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        )

        # Evaluation metrics
        self.val_accuracy = Accuracy('multiclass', num_classes=10)
        self.test_accuracy = Accuracy('multiclass', num_classes=10)

    def setup(self, stage=None):
        """
        Dataset preparation.

        Argument/s:
            stage - either 'fit' (training & validation sets) or 'test'
                (test set).
        """

        # Assign training set
        if stage == 'fit' or stage is None:
            train_set = torchvision.datasets.CIFAR10(
                root=self.dataset_dir, train=True, download=True, transform=self.train_transforms,
            )
            self.train_set, self.val_set = random_split(train_set, [45000, 5000])

        if stage == 'test' or stage is None:
            self.test_set = torchvision.datasets.CIFAR10(
                root=self.dataset_dir, train=False, download=True, transform=self.test_transforms,
            )

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

    def forward(self, images):
        """
        Forward propagation over the torch.mm.Module attributes of self.

        Argument/s:
            images - a mini-batch of images from the dataset.

        Returns:
            A dictionary containing the outputs from the network/s.
        """
        return {'logits': self.model(images)}

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
        images, labels = batch

        # Inference
        y_hat = self(images)

        # Loss
        loss = self.loss(y_hat['logits'], labels)

        # Logging
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=images.size()[0])

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
        images, labels = batch

        # Inference
        y_hat = self(images)

        # Loss
        loss = self.loss(y_hat['logits'], labels)

        # Compute metric scores
        self.val_accuracy(torch.argmax(y_hat['logits'], dim=1), labels)

        # Logging
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, batch_size=images.size()[0])
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=images.size()[0])

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Argument/s:
            batch - mini-batch from the test set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.
        """

        # Mini-batch of examples
        images, labels = batch

        # Inference
        y_hat = self(images)

        # Compute metric scores
        self.test_accuracy(torch.argmax(y_hat['logits'], dim=1), labels)

        # Log the test accuracy
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, batch_size=images.size()[0])
