from pytorch_lightning import LightningModule
from task.cifar10.model.baseline import Baseline
from task.cifar10.model.resnet18 import ResNet18

class Composite(LightningModule):
    def __init__(self, **kwargs):
        self.baseline = Baseline(self, **kwargs)

    def setup(self, stage=None):
        self.baseline.setup(stage)

    def train_dataloader(self, shuffle=True):
        return self.baseline.train_dataloader(shuffle)

    def val_dataloader(self):
        return self.baseline.val_dataloader()

    def test_dataloader(self):
        return self.baseline.test_dataloader()

    def configure_optimizers(self):     
        return ResNet18.configure_optimizers(self)  # Use configure_optimizers() from ResNet18.

    def forward(self, images):
        return self.baseline.forward(images)

    def training_step(self, batch, batch_idx):
        return self.baseline.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.baseline.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.baseline.test_step(batch, batch_idx)
