import torch
import torchvision

# Inherit from baseline model
from task.cifar10.model.baseline import Baseline

class ResNet18(Baseline):
    """
    ResNet-18 CIFAR-10 model.
    """

    def __init__(self, **kwargs):
        """
        Argument/s:
            kwargs - keyword arguments.
        """
        super(ResNet18, self).__init__(**kwargs)

        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(
            in_features=512, out_features=len(self.classes), bias=True,
        )

    def configure_optimizers(self):
        """
        Define optimiser/s and learning rate scheduler/s. A good source for understanding
        how the pytorch learning rate schedulers work can be found at:
            https://www.programmersought.com/article/12164650026/

        Returns:
            The optimiser/s.
        """
        optimiser = {'optimizer': torch.optim.AdamW(self.parameters(), lr=self.lr)}
        return optimiser
