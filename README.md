[//]: # (# In developement...)

# Transmodal

---

**Transmodal meaning**: Crossing, occurring in, or using more than one mode or modality. ***The name could be changed to something that fits the current state of the repo more.***

***Aims of this library:***

 - To be simple and easy to understand so that the focus is on the data science.
 - To reduce the time taken from implementation to results.
 - To promote rapid innovation of models via configuration files, class composition and/or class inheritance.
 - Reduce boilerplate code (sections of code that are repeated in multiple places with little to no variation).
 - To simplify cluster management and distributed computing for the user.
 - Be able to easily accommodate multiple research avenues simultaneously.

Most of this is accomplished by leveraging [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), familiarity with it is recommended.

# Repository map

---


Overview of the repository. ***The most important parts are: `task`, `config`, `models`, and `stages`.***
```
├──  task  
│    │
│    └── TASK_NAME     - name of the task, e.g., cifar10.
│        └── config    - .yaml configuration files for a model.
│        └── models    - .py modules that contain pytorch_lightning.LightningModule definitions that represent models.
│        └── stages.py - training and testing stages for a task.
│
│
├──  transmodal
│    │
│    └── tools                     - for all other modules; tools that are repeadetly used.
│    └── cluster.py                - contains the cluster management object.
│    └── command_line_arguments.py - argparse for reading command line arguments.
│    └── trainer.py                - contains a wrapper for pytorch_lightning.Trainer.
│    └── utils.py                  - small utility definitions.
│
│
├──  main.py - main.py does the following:
│               1. Reads command line arguments using argparse.
│               2. Imports the 'stages' function for the task from task/TASK_NAME/stages.py.
│               3. Loads the specified configuration .yaml for the job from task/TASK_NAME/config.
│               4. Submits the job (the configuration + 'stages') to the cluster manager (or runs it locally if 'submit' is false).
│
│
├──  requirements.txt - Packages required by the library (pip install -r requirements.txt).
```


# Tasks

---


***Tasks are named based on the data and the type of prediction or inference being made***. For example:
 - Two tasks have the same data but require different names due to differing predictions, e.g., **MS-COCO Detection** and **MS-COCO Caption**.
 - Two tasks may have similar predictions but require different names due to differing data, e.g., **MNIST** and **Chinese MNIST**.

***Some publicly available tasks include***:
- Image classification tasks, e.g., [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html), [ImageNet](https://www.image-net.org/). 
- Object detection tasks, e.g., [MS-COCO Detection](https://cocodataset.org/#detection-2020).
- Image captioning detection tasks, e.g., [MS-COCO Caption](https://cocodataset.org/#captions-2015).
- Speech recognition tasks, e.g., [LibriSpeech](https://www.openslr.org/12).
- Chest X-Ray report generation, e.g., [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/).

***How to add a task:***

Adding a task is as simple as creating a directory with the name of the task in `task`. For example, if we choose CIFAR10 as the task, with the task name `cifar10`, then we would create the directory `task/cifar10`. The task directory will then house everything necessary for that task, for example, the models, the configurations for the models, the data pipeline, and the stages of development (training and testing).

# Models

---


***Please familiarise yourself with the [`pytorch_lightning.LightningModule`](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#) in order to correctly implement a model:*** https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

Once we have created our task directory (e.g., `task/cifar10`), we now want to create a model using a [`pytorch_lightning.LightningModule`](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#). Everything we need for the model can be placed in the `LightningModule`, in including commonly used libraries and objects, for example:

 - [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module): base class for all neural networks in `PyTorch`. 
 - [transformers](https://huggingface.co/docs/transformers/index): a library containing pre-trained Transformer models.
 - [torchvision](https://pytorch.org/vision/stable/index.html): a library for image pre-processing and pre-trained computer vision models.
 - [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset): an object that processes each instance of a dataset.
 - [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader): an object that samples mini-batches from a [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).

***Note:*** 

- The data pipeline could be implemented within the `LightningModule` or seperately from a model using a [pytorch_lightning.LightningDataModule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html). The `LightningModule` instance would then have to be given separately to the `pytorch_lightning.Trainer`. 

***Example:***

 - An example model for `cifar10` is in [task/cifar10/model/baseline.py](https://github.com/aehrc/transmodal/blob/simplified_22/task/cifar10/model/baseline.py). 

# Innovate via Model Composition & Inheritance

---
To promote rapid innovation of models, we recommend using class composition and/or inheritance. ***For example, we may have a baseline that not only includes a basic model, but also the data pipeline:***

```python
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, random_split
import torchvision
import torch

class Baseline(LightningModule):
    def __init__(self, lr, ..., **kwargs):
        super(Baseline, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = torchvision.models.resnet18(...)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_set = torchvision.datasets.CIFAR10(...)
            self.train_set, self.val_set = random_split(train_set, [45000, 5000])

        if stage == 'test' or stage is None:
            self.test_set = torchvision.datasets.CIFAR10(...)

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.train_set, ...)

    def val_dataloader(self):
        return DataLoader(self.val_set, ...)

    def test_dataloader(self):
        return DataLoader(self.test_set, ...)

    def configure_optimizers(self):     
        optimiser = {'optimizer': torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)}
        return optimiser

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self(images)
        loss = self.loss(y_hat, labels)
        self.log_dict({'train_loss': loss}, ...)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self(images)
        loss = self.loss(y_hat, labels)
        self.val_accuracy(torch.argmax(y_hat['logits'], dim=1), labels)
        self.log_dict({'val_acc': self.val_accuracy, 'val_loss': loss}, ...)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self(images)
        self.test_accuracy(torch.argmax(y_hat['logits'], dim=1), labels)
        self.log_dict({'test_acc': self.test_accuracy}, ...)
```

After training and testing the baseline, we may want to improve upon its performance. For example, if we wanted to make the following modifications:
 
 - Use a DenseNet instead of a ResNet.
 - Use the `AdamW` optimiser.
 - Use a warmup learning rate scheduler. 

***All we would need to do is inherit the baseline and make our modifications:***

```python
from transformers import get_constant_schedule_with_warmup

class Inheritance(Baseline):

    def __init__(self, num_warmup_steps, **kwargs):
        super(DenseNetAdamW, self).__init__(**kwargs)
        self.save_hyperparameters()
        self.num_warmup_steps = num_warmup_steps
        self.model = torchvision.models.densenet121(...)

    def configure_optimizers(self):
        optimiser = {'optimizer': torch.optim.AdamW(self.parameters(), lr=self.lr)}
        optimiser['scheduler'] = {
                'scheduler': get_constant_schedule_with_warmup(optimiser['optimizer'], self.num_warmup_steps),
                'interval': 'step',
                'frequency': 1,
            }
        return optimiser
```
We could also construct a model that is the combination of the two via composition. For example, we may want to use evarything from `Baseline`, but the optimiser from `Inheritance`:

```python
from pytorch_lightning import LightningModule

class Composition(LightningModule):
    def __init__(self, **kwargs):
        Baseline.__init__(self, **kwargs)

    def setup(self, stage=None):
        Baseline.setup(self, stage)

    def train_dataloader(self, shuffle=True):
        return Baseline.train_dataloader(self, shuffle)

    def val_dataloader(self):
        return Baseline.val_dataloader(self)

    def test_dataloader(self):
        return Baseline.test_dataloader(self)

    def configure_optimizers(self):     
        return Inheritance.configure_optimizers(self)  # Use configure_optimizers() from Inheritance. 

    def forward(self, images):
        return Baseline.forward(self, images)

    def training_step(self, batch, batch_idx):
        return Baseline.training_step(self, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return Baseline.validation_step(self, batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return Baseline.test_step(self, batch, batch_idx)
```

# Configuration `.yaml` files and `argparse`

---


Currently, there are two methods for giving arguments:

1. **Via command line arguments using the [`argparse` module](https://docs.python.org/3/library/argparse.html)**. `argparse` mainly handles paths, development stage flags (e.g., training and testing flags), and cluster manager arguments.
2. **Via a configuration file stored in [`YAML` format](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started)**. Can handle all the arguments defined by the `argparse` plus more, including hyperparameters for the model.

***There are only four mandatory arguments***
1. `task`, the name of the task.
2. `config`, the name of the configuration (no extension).
3. `module`, the name of the module that the model definition is housed.
4. `definition`, the name of the class representing the model.
5. `exp_dir`, the experiment directory, i.e., where all outputs, including model checkpoints will be saved.

***`task` and `config` must be given as command line arguments for `argparse`:***

```shell
python3 main.py --config baseline --task cifar10
```

***`module`, `definition`, and `exp_dir` can be given either as command line arguments, or be placed in the configuration file.***




For each model of a task, we define a configuration. Hyperparameters, paths, as well as the device configuration can be stored in a configuration file. Configuration files have the following strict requirements:

1. They are stored in the `config` directory of  a task, e.g., `task/cifar10/config`.
2. They are stored in [`YAML` format](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started), e.g., `task/cifar10/config/baseline.yaml`.

# Innovate via Configuration Files

---

If we have the following configuration file for the aforementioned CIFAR10  `Baseline` model, `task.cifar10.config.baseline.yaml`:

```yaml
train: True
test: True
module: baseline
definition: Baseline
monitor: 'val_acc'
monitor_mode: 'max'
lr: 1e-3
max_epochs: 32
mbatch_size: 32
num_workers: 5
exp_dir: /my/experiment/directory
dataset_dir: /my/datasets/directory
```

Another way we can improve upon the baseline model, i.e., the baseline configuration, is by modifying its hyperparameters. For example, we can still use `Baseline`, but alter the learning rate in `task.cifar10.config.baseline_rev_a.yaml`:

```yaml
train: True
test: True
module: baseline
definition: Baseline
monitor: 'val_acc'
monitor_mode: 'max'
lr: 1e-4  # modify this.
max_epochs: 32
mbatch_size: 32
num_workers: 5
exp_dir: /my/experiment/directory
dataset_dir: /my/datasets/directory
```

```shell
python3 main.py --config baseline_rev_a --task cifar10
```

# Stages & `pytorch_lightning.Trainer`

---


In each task directory is a Python module called `stages.py`, which contains the `stages` definition. This definition takes an object as input that houses the configuration for a job.  

Typically, the following things happen in `stages()`:

 - The `LightningModule` model is imported via the `model` argument, e.g.,
    ```python
    from lib.utils import importer
   
    Model = importer(definition=args.definition, module='.'.join(['task', args.task, 'model', args.module])
    model = Model(**vars(args))
   ```
    See `transmodal.utils.importer` for a handy function that imports based on strings.
 - A `pytorch_lightning.Trainer` instance is created, e.g., `trainer = pytorch_lightning.Trainer(...)`.
 - The model is trained using trainer: `trainer.fit(model)`.
 - The model is tested using trainer: `trainer.test(model)`.

It handles the training and testing of a model for a task by using a [`pytorch_lightning.Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).

***A helpful wrapper at `transmodal/trainer.py` exists that passes frequently used and useful `callbacks`, `loggers`, and `plugins` to a `pytorch_lightning.Trainer` instance:***

```python
from lib.trainer import trainer_instance

trainer = trainer_instance(**vars(args))
```
Place any of the parameters for the trainer detailed at 
https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api in your configuration file, and they will be passed to the `pytorch_lightning.Trainer` instance.

# Tying it all together: `main.py`

---

***This is an overview of what occurs in `main.py`, you do not need to modify it.***

The main function does the following:

 - Gets the command line arguments using `argparse`, e.g., arguments like this:
    ```shell
    python3 main.py --config baseline --task cifar10
    ```
 - Imports the `stages` definition for the task using `transmodal.utils.importer`.
 - Reads the configuration `.yaml` and combines it with the command line arguments.
 - Submits `stages` to the cluster manager if `args.submit = True` or runs `stages` locally. The command line arguments and the configuration arguments are passed to `stages` in both cases.

# Cluster manager arguments & distributed computing

---

The following arguments are used for distributed computing:

| Argument      | Description                                                | Default      |
|---------------|------------------------------------------------------------|--------------|
| `num_workers` | No. of workers per DataLoader & GPU                        | `1`          |
| `num_gpus`    | Number of GPUs per node                                    | `1`          |
| `num_nodes`   | Number of nodes (should only be used with `submit = True`) | `1`          |

The following arguments are used to configure a job for a cluster manager (the default cluster manager is SLURM):

| Argument     | Description                                                    | Default      |
|--------------|----------------------------------------------------------------|--------------|
| `memory`     | Amount of memory per node                                      | `'16GB'`     |
| `time_limit` | Job time limit                                                 | `'02:00:00'` |
| `submit`     | Submit job to the cluster manager                              | `False`      |
| `resumable`  | Resumable training; Automatic resubmission to cluster manager  | `False`      |
| `qos`        | Quality of service                                             | `None`       |
| `begin`      | When to begin the Slurm job, e.g. now+1hour                    | `'now'`      |
| `email`      | Email for cluster manager notifications                        | `None`       |
| `venv_path`  | Path to ''bin/activate'' of a venv.                            | `None`       |

***These can be given as command line arguments:***

 ```shell
 python3 main.py --config baseline --task cifar10 --submit 1 --num-gpus 4 --num-workers 5 --memory 32GB
 ```

***Or they can be placed in the configuration `.yaml` file:***

```yaml
submit: True  # Added.
num_gpus: 4  # Added.
num-workers: 5  # Added.
memory: '32GB'  # Added.

train: True
test: True
module: baseline
definition: Baseline
monitor: 'val_acc'
monitor_mode: 'max'
lr: 1e-3
max_epochs: 32
mbatch_size: 32
num_workers: 5
exp_dir: /my/experiment/directory
dataset_dir: /my/datasets/directory
```
And executed with:
```shell
 python3 main.py --config baseline --task cifar10
 ```

# Feature Wish List

---


 - Ray Tune for hyperparameter optimisation.
 - A more foolproof way to combine command line arguments and the configuration arguments.
 - Usable with notebooks.