![](/assets/image.png)

***Aims of this library:***

- To be simple and easy to understand so that the focus is on the data science.
- To reduce the time taken from implementation to results.
- To promote rapid development of models via configuration files, class composition and/or class inheritance.
- Reduce boilerplate code (sections of code that are repeated in multiple places with little to no variation).
- To simplify cluster management and distributed computing with High Performance Computing (HPC).
- Be able to easily accommodate multiple research avenues simultaneously.
- To cooperatively improve the functionality and documentation of this repository to make it better!

***Features:***
- The [Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) `LightningModule` and `Trainer` are used to implement, train, and test models. It allows for many of the above aims to be accomplished, such as simplified distributed computing and a reduction of boilerplate code. It also allows us to simply use class inheritance and composition, allowing for rapid development.
- The [Compose API](https://hydra.cc/docs/advanced/compose_api/) of [Hydra](https://hydra.cc/) is used to create a hierarchical configuration, allowing for rapid development.
- [Neptune.ai](https://neptune.ai/) is used to track experiments; metric scores are automatically uploaded to [Neptune.ai](https://neptune.ai/), allowing you to easily track your experiments from your browser.
- Scripts for submission to a cluster manager, such as [SLURM](https://slurm.schedmd.com/documentation.html) are written for you. Also, cluster manager jobs are automatically resubmitted and resumed if they haven't finished before the time-limit.

# Installation

The Deep Learning and HPC starter pack is available on PyPI:
```shell
pip install dlhpcstarter
```

# Table of Contents

[//]: # (- [How to structure your project]&#40;#how-to-structure-your-project&#41;)
- [Package map](#package-map)
- [Tasks](#tasks)
- [Models](#models)
- [Development via Model Composition and Inheritance](#development-via-model-composition-and-inheritance)
- [Configuration YAML files and argparse](#configuration-yaml-files-and-argparse)
- [Development via Configuration Files](#development-via-configuration-files)
- [Next level: Configuration composition via Hydra](#next-level-configuration-composition-via-hydra)
- [Stages and Trainer](#stages-and-trainer)
- [Tying it all together: `main.py`](#tying-it-all-together-mainpy)
- [Cluster manager and distributed computing](#cluster-manager-and-distributed-computing)
- [Monitoring using Neptune.ai](#monitoring-using-neptuneai)
- [Where all the outputs go: `exp_dir`](#where-all-the-outputs-go-exp_dir)
- [Repository Wish List](#repository-wish-list)

[//]: # (# How to structure your project)

[//]: # ()
[//]: # (---)

[//]: # (There will be a `task` directory containing each of your tasks, e.g., `cifar10`. For each task, you will have a set of configurations and models, which are stored in the `config` and `models` directories, respectively. Each task will also have a `stages` module for each stage of model development.)

[//]: # (```)

[//]: # (├──  task  )

[//]: # (│    │)

[//]: # (│    └── TASK_NAME     - name of the task, e.g., cifar10.)

[//]: # (│        └── config    - .yaml configuration files for a model.)

[//]: # (│        └── models    - .py modules that contain Lightning LightningModule definitions that represent models.)

[//]: # (│        └── stages.py - training and testing stages for a task.)

[//]: # (```)

# Package map

---

The package is structured as follows:

```
├──  dlhpcstarter
│    │
│    ├── tools                     - for all other modules; tools that are repeadetly used.
│    ├── __main__.py               - __main__.py does the following:
│    │                                    1. Reads command line arguments using argparse.
│    │                                    2. Imports the 'stages' function for the task from task/
│    │                                    3. Loads the specified configuration .yaml for the job from 
│    │                                    4. Submits the job (the configuration + 'stages') to the 
│    │                                       cluster manager (or runs it locally if 'submit' is false).
│    └── cluster.py                - contains the cluster management object.
│    └── command_line_arguments.py - argparse for reading command line arguments.
│    └── trainer.py                - contains an optional wrapper for the Lightning Trainer.
│    └── utils.py                  - small utility definitions.

```

# Tasks

---


***Tasks can have any name. The name could be based on the data or the type of inference being made***. For example:
- Two tasks have the same data but require different names due to differing predictions, e.g., **MS-COCO Detection** and **MS-COCO Caption**.
- Two tasks may have similar predictions but require different names due to differing data, e.g., **MNIST** and **Chinese MNIST**.

***Some publicly available tasks include***:
- Image classification tasks, e.g., [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html), [ImageNet](https://www.image-net.org/).
- Object detection tasks, e.g., [MS-COCO Detection](https://cocodataset.org/#detection-2020).
- Image captioning detection tasks, e.g., [MS-COCO Caption](https://cocodataset.org/#captions-2015).
- Speech recognition tasks, e.g., [LibriSpeech](https://www.openslr.org/12).
- Chest X-Ray report generation, e.g., [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/).

***What does the name do?***

It is used to separate the outputs of the experiment from other tasks.

# Models

---


***Please familiarise yourself with the [`Lightning LightningModule`](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#) in order to correctly implement a model:*** https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

A model is created using a [`Lightning LightningModule`](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#). Everything we need for the model can be placed in the `LightningModule`, including commonly used libraries and objects, for example:

- [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module): base class for all neural networks in `PyTorch`.
- [transformers](https://huggingface.co/docs/transformers/index): a library containing pre-trained Transformer models.
- [torchvision](https://pytorch.org/vision/stable/index.html): a library for image pre-processing and pre-trained computer vision models.
- [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset): an object that processes each instance of a dataset.
- [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader): an object that samples mini-batches from a [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).

***Note:***

- The data pipeline could be implemented within the `LightningModule` or seperately from a model using a [Lightning LightningDataModule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html). The `LightningModule` instance would then have to be given separately to the `Lightning Trainer` instance.

***Example:***

- An example model for `cifar10` is in [task/cifar10/model/baseline.py](https://github.com/csiro-mlai/dl_hpc_starter_pack/blob/main/task/cifar10/model/baseline.py).

# Development via Model Composition and Inheritance

---
To promote rapid development of models, one solution is to use class composition and/or inheritance. ***For example, we may have a baseline that not only includes a basic model, but also the data pipeline:***

```python
from lightning.pytorch import LightningModule
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
        super(Inheritance, self).__init__(**kwargs)
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
We could also construct a model that is the combination of the two via composition. For example, we may want to use everything from `Baseline`, but the optimiser from `Inheritance`:

```python
from lightning.pytorch import LightningModule

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
        return Inheritance.configure_optimizers(self)  # Use configure_optimizers() from Inheritance.

    def forward(self, images):
        return self.baseline.forward(images)

    def training_step(self, batch, batch_idx):
        return self.baseline.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.baseline.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.baseline.test_step(batch, batch_idx)
```

# Configuration YAML files and argparse

---


Currently, there are two methods for giving arguments:

1. **Via command line arguments using the [`argparse` module](https://docs.python.org/3/library/argparse.html)**. `argparse` mainly handles paths, development stage flags (e.g., training and testing flags), and cluster manager arguments.
2. **Via a configuration file stored in [`YAML` format](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started)**. Can handle all the arguments defined by the `argparse` plus more, including hyperparameters for the model.

***The mandatory arguments include:***
1. `task`, the name of the task.
2. `config`, relative or absolute path to the configuration file (with or without the extension).
3. `module`, the module that the model definition is housed.
4. `definition`, the class representing the model.
5. `exp_dir`, the experiment directory, i.e., where all outputs, including model checkpoints will be saved.
6. `monitor`, metric to monitor for [ModelCheckpoint](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html) and [EarlyStopping](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html?highlight=earlystoppin#earlystopping) (optional), as well as test checkpoint loading (e.g., 'val_loss').
7. `monitor_mode`, whether the monitored metric is to be maximised or minimised ('max' or 'min').

***`task` and `config` must be given as command line arguments for `argparse`:***

```shell
dlhpcstarter --config task/cifar10/config/baseline --task cifar10
```

***`module`, `definition`, and `exp_dir` can be given either as command line arguments, or be placed in the configuration file.***

For each model of a task, we define a configuration. Hyperparameters, paths, as well as the device configuration can be stored in a configuration file. Configurations are in [`YAML` format](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started), e.g., `task/cifar10/config/baseline.yaml`.

# Development via Configuration Files

---

If we have the following configuration file for the aforementioned CIFAR10  `Baseline` model, `task/cifar10/config/baseline.yaml`:

```yaml
train: True
test: True
module: task.cifar10.model.baseline
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

Another way we can improve upon the baseline model, i.e., the baseline configuration, is by modifying its hyperparameters. For example, we can still use `Baseline`, but alter the learning rate in `task/cifar10/config/baseline_rev_a.yaml`:

```yaml
train: True
test: True
module: task.cifar10.model.baseline
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
dlhpcstarter --config task/cifar10/config/baseline_rev_a --task cifar10
```

# Next level: Configuration composition via Hydra

---

If your new configuration only modifies a few arguments of another configuration file, you can take advantage of the composition feature of [Hydra](https://hydra.cc/). This makes creating `task/cifar10/config/baseline_rev_a.yaml` from the previous section easy. We simply add the arguments from `task/cifar10/config/baseline.yaml` by adding its name to the `defaults` list:

```yaml
defaults:
  - baseline
  - _self_

lr: 1e-4
```
***Note that other configuration files are imported with reference to the current configuration path (not the working directory).***


Please note that groups are not being used, and packages should be placed using `@_global_` if the configurations being used for composition are not in the same directory. ***For example, the following would not work with this repository as the arguments in `hpc_paths` will be grouped under `paths`:***

```yaml
defaults:
  - paths/hpc_paths
  - _self_

train: True
test: True
resumable: True
module: task.cifar10.model.baseline
definition: Baseline
monitor: 'val_acc'
monitor_mode: 'max'
lr: 1e-3
max_epochs: 3
mbatch_size: 32
num_workers: 5
```

To get around this, simply place `@_global_` to remove the grouping:

```yaml
defaults:
  - paths/hpc_paths@_global_  # changed here to remove "paths" grouping.
  - _self_

train: True
test: True
resumable: True
module: task.cifar10.model.baseline
definition: Baseline
monitor: 'val_acc'
monitor_mode: 'max'
lr: 1e-3
max_epochs: 3
mbatch_size: 32
num_workers: 5
```

This also allows us to organise configurations easily. For example, if we have the following directory structure:
```
├── task
│   └──  cifar10          
│        └── config  
│            ├── cluster
│            │    ├── 2hr.yaml
│            │    └── 24hr.yaml
│            │
│            ├── distributed
│            │    ├── 1gpu.yaml
│            │    ├── 4gpu.yaml
│            │    └── 4gpu4node.yaml
│            │
│            ├── paths
│            │    ├── local.yaml
│            │    └── hpc.yaml
│            │
│            └── baseline.yaml
```
With `task/cifar10/config/baseline.yaml` as:
```yaml
defaults:
  - cluster/2hr@_global_
  - distributed/4gpu@_global_
  - paths/hpc_paths@_global_
  - _self_

train: True
test: True
resumable: True
module: task.cifar10.model.baseline
definition: Baseline
monitor: 'val_acc'
monitor_mode: 'max'
lr: 1e-3
max_epochs: 3
mbatch_size: 32
num_workers: 5
```

Where `task/cifar10/config/baseline.yaml` will now include arguments from the following example sub-configurations:

- `task/cifar10/config/cluster/2hr.yaml`:
   ```yaml
   memory: 32GB
   time_limit: '02:00:00'
   venv_path: /path/to/my/venv/bin/activate
   ```
- `task/cifar10/config/distributed/4gpu.yaml`:
  ```yaml
  num_gpus: 2
  strategy: ddp
  ```
- `task/cifar10/config/paths/hpc.yaml`:
  ```yaml
  exp_dir: /path/to/my/experiments
  dataset_dir: /path/to/my/dataset
  ```

See the following documentation for more information:
- https://hydra.cc/docs/1.2/tutorials/basic/your_first_app/defaults/
- https://hydra.cc/docs/1.2/advanced/defaults_list/#composition-order
- https://hydra.cc/docs/1.2/advanced/overriding_packages/

# Stages and Trainer

---


In each task directory is a Python module called `stages.py`, which contains the `stages` definition. This definition takes an object as input that houses the configuration for a job.

Typically, the following things happen in `stages()`:

- The `LightningModule` model is imported via the `model` argument, e.g.,
   ```python
   from src import importer
  
   Model = importer(definition=args.definition, module=args.module)
   model = Model(**vars(args))
  ```
  See `src.utils.importer` for a handy function that imports based on strings.
- A `Lightning Trainer` instance is created, e.g., `trainer = lightning.pytorch.Trainer(...)`.
- The model is trained using trainer: `trainer.fit(model)`.
- The model is tested using trainer: `trainer.test(model)`.

It handles the training and testing of a model for a task by using a [`Lightning Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).

***A helpful wrapper at `src/trainer.py` exists that passes frequently used and useful `callbacks`, `loggers`, and `plugins` to a `Lightning Trainer` instance:***

```python
from src.dlhpcstarter.trainer import trainer_instance

trainer = trainer_instance(**vars(args))
```
Place any of the parameters for the trainer detailed at
https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api in your configuration file, and they will be passed to the `Lightning Trainer` instance.

# Tying it all together: `dlhpcstarter`

---

***This is an overview of what occurs when the entrypoint `dlhpcstarter` is executed, this is not necessary to understand to use the package.***



`dlhpcstarter` does the following:

- Gets the command line arguments using `argparse`, e.g., arguments like this:
   ```shell
   dlhpcstarter --config task.cifar10.config.baseline --task cifar10
   ```
- Imports the `stages` definition for the task using `src.utils.importer`.
- Reads the configuration `.yaml` and combines it with the command line arguments.
- Submits `stages` to the cluster manager if `args.submit = True` or runs `stages` locally. The command line arguments and the configuration arguments are passed to `stages` in both cases.

# Cluster manager and distributed computing

---

The following arguments are used for distributed computing:

| Argument      | Description                                                 | Default |
|---------------|-------------------------------------------------------------|---------|
| `num_workers` | No. of workers per DataLoader & GPU.                        | `1`     |
| `num_gpus`    | Number of GPUs per node.                                    | `None`  |
| `num_nodes`   | Number of nodes (should only be used with `submit = True`). | `1`     |

The following arguments are used to configure a job for a cluster manager (the default cluster manager is SLURM):

| Argument     | Description                                                    | Default      |
|--------------|----------------------------------------------------------------|--------------|
| `memory`     | Amount of memory per node.                                     | `'16GB'`     |
| `time_limit` | Job time limit.                                                | `'02:00:00'` |
| `submit`     | Submit job to the cluster manager.                             | `None`       |
| `resumable`  | Resumable training; Automatic resubmission to cluster manager. | `None`       |
| `qos`        | Quality of service.                                            | `None`       |
| `begin`      | When to begin the Slurm job, e.g. `now+1hour`.                 | `None`       |
| `email`      | Email for cluster manager notifications.                       | `None`       |
| `venv_path`  | Path to ''bin/activate'' of a venv.                            | `None`       |

***These can be given as command line arguments:***

 ```shell
dlhpcstarter --config task/cifar10/config/baseline --task cifar10 --submit 1 --num-gpus 4 --num-workers 5 --memory 32GB
 ```

***Or they can be placed in the configuration `.yaml` file:***

```yaml
num_gpus: 4  # Added.
num_workers: 5  # Added.
memory: '32GB'  # Added.

train: True
test: True
module: task.cifar10.model.baseline
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
dlhpcstarter --config task/cifar10/config/baseline --task cifar10 --submit True
 ```

If using a cluster manager, add the path to the `bin/activate` of your virtual environment:
```yaml
...
venv_path: /my/env/name/bin/activate
...
```

# Monitoring using Neptune.ai

Simply sign up at https://neptune.ai/ and add your username and API token to your configuration file:

```yaml
...
neptune_username: my_username
neptune_api_key: df987y94y2q9hoiusadhc9wy9tr82uq408rjw98ch987qwhtr093q4jfi9uwehc987wqhc9qw4uf9w3q4h897324th
...
```
The [PyTorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) will then automatically upload metrics using the [Neptune Logger](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.neptune.html) to [Neptune.ai](https://neptune.ai/). Once logged in to https://neptune.ai/, you will be able to monitor your task. See here for information about using the online UI: https://docs.neptune.ai/you-should-know/displaying-metadata.

# Where all the outputs go: `exp_dir`

---

The experiments directory is where all your outputs will be saved, including model checkpoints, metric scores. This is also where the cluster manager script, as well as where stderr and stdout are saved.

Note: the trial number also sets the seed number for your experiment.

***Description to be finished.


# Repository Wish List

---
- Transfer cluster management over to submitit: https://ai.facebook.com/blog/open-sourcing-submitit-a-lightweight-tool-for-slurm-cluster-computation/
- Add description about how to use https://neptune.ai/.
- Use https://hydra.cc/ instead of argparse (or have the option to use either).
- https://docs.ray.io/en/latest/tune/index.html for hyperparameter optimisation.
- Notebook examples.

