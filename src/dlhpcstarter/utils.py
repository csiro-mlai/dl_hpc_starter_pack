from argparse import Namespace
from hydra import compose, initialize_config_dir
from pathlib import Path
from typing import Optional
import glob
import GPUtil
import importlib
import numpy as np
import os
import re
import warnings


def importer(
        definition: str,
        module: str,
        work_dir: Optional[str] = None,
):
    """
    Loads an instance of the definition from the given module. Can also load relative to a specified working directory.

    Argument/s:
        definition - the definition in the module. Typically the name of a function or class.
        module - the module, e.g. tasks.mnist.datamodule
        work_dir - working directory; loads the module relative to this directory.

    Returns:
        An instance of 'definition'.
    """

    if work_dir:

        absolute_path = os.path.join(work_dir, *module.split('.')) + '.py'

        assert os.path.isfile(absolute_path), \
            f'''{absolute_path} does not exist. The target definition was: {definition}.'''

        module = importlib.machinery.SourceFileLoader(module, absolute_path).load_module()

    else:

        path = os.path.join(*module.split('.')) + '.py'

        assert os.path.isfile(path), f'{path} does not exist. The target definition was: {definition}.'

        module = importlib.import_module(module)

    return getattr(module, definition)


def load_config_and_update_args(args: Namespace, print_args: bool = False) -> None:
    """
    Loads the configuration .yaml file and updates the args object.

    Argument/s:
        args - object that houses the arguments for the job.
        print_args - print the arguments for the job.
    """
    # Add the working directory to paths
    if not args.work_dir:
        args.work_dir = os.getcwd()

    # Configuration name
    if args.config.endswith('.yaml'):
        args.config_file_name = args.config
        args.config = args.config.replace('.yaml', '')
    else:
        args.config_file_name = args.config + '.yaml'

    # Load configuration using Hydra's Compose API
    args.config_dir = Path(args.config).parent
    if not os.path.isabs(args.config_dir):
        args.config_dir = os.path.join(args.work_dir, args.config_dir)
    with initialize_config_dir(version_base=None, config_dir=args.config_dir):
        config = compose(config_name=Path(args.config).parts[-1])

    # Update args with config and check for conflicts
    args.config_full_path = os.path.join(args.work_dir, args.config_file_name)
    for k, v in config.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
        else:
            if getattr(args, k) != v:
                raise ValueError(f'There is a conflict between command line argument "--{k} {getattr(args, k)}" '
                                 f'({type(getattr(args, k))}) and configuration argument "{k}: {v}" from '
                                 f'{args.config_full_path} ({type(v)}).')

    # The model name must be defined in the configuration or in the command line arguments
    assert args.module, f'"module" must be specified as a command line argument or in {args.config_full_path}.'
    assert args.definition, f'"definition" must be specified as a command line argument or in {args.config_full_path}.'
    assert args.exp_dir, f'"exp_dir" must be specified as a command line argument or in {args.config_full_path}.'

    # Defaults: There is probably a better place to do this
    args.num_workers = args.num_workers if args.num_workers is not None else 1
    args.num_nodes = args.num_nodes if args.num_nodes is not None else 1
    if args.submit:
        args.time_limit = args.time_limit if args.time_limit is not None else '02:00:00'
        args.memory = args.memory if args.memory is not None else '16GB'

    # Add the task, configuration name, and the trial number to the experiment directory
    args.trial = args.trial if args.trial is not None else 0
    args.exp_dir_trial = os.path.join(args.exp_dir, args.task, args.config, 'trial_' + f'{args.trial}')
    Path(args.exp_dir_trial).mkdir(parents=True, exist_ok=True)

    if print_args:
        print(f'args: {args.__dict__}')

    # Print GPU usage and set GPU visibility
    if hasattr(args, 'num_gpus'):
        gpu_usage_and_visibility(args.cuda_visible_devices, args.submit)


def get_epoch_ckpt_path(exp_dir_trial: str, load_epoch: int) -> str:
    """
    Get the checkpoint path based on the epoch number.

    Argument/s:
        exp_dir_trial - Experiment directory for the trial (where the checkpoints are saved).
        load_epoch - epoch to load.

    Returns:
        Path to the epoch's checkpoint.
    """
    try:
        ckpt_path = glob.glob(os.path.join(exp_dir_trial, "*epoch=" + str(load_epoch) + "*.ckpt"))
        assert len(ckpt_path) == 1, f'Multiple checkpoints for epoch {load_epoch}: {ckpt_path}.'

    except:
        raise ValueError(
            "Epoch {} is not in the checkpoint directory.".format(str(load_epoch))
        )
    return ckpt_path[0]


def get_best_ckpt_path(exp_dir_trial: str, monitor_mode: str) -> str:
    """
    Get the best performing checkpoint from the experiment directory.

    Argument/s:
        exp_dir_trial - Experiment directory for the trial (where the checkpoints are saved).
        monitor_mode - Metric monitoring mode, either "min" or "max".

    Returns:
        Path to the epoch's checkpoint.
    """

    ckpt_list = glob.glob(os.path.join(exp_dir_trial, '*=*=*.ckpt'))

    if not ckpt_list:
        raise ValueError(f'No checkpoints exist in the checkpoint directory: {exp_dir_trial}.')

    scores = [
        re.findall(r"[-+]?\d*\.\d+|\d+", i.rsplit('=', 1)[1])[0] for i in ckpt_list
    ]

    if monitor_mode == 'max':
        ckpt_path = ckpt_list[np.argmax(scores)]
    elif monitor_mode == 'min':
        ckpt_path = ckpt_list[np.argmin(scores)]
    else:
        raise ValueError("'monitor_mode' must be max or min, not {}.".format(monitor_mode))
    return ckpt_path


def get_test_ckpt_path(
        exp_dir_trial: str, monitor_mode: str, test_epoch: Optional[int] = None, test_ckpt_path: Optional[str] = None,
) -> str:
    """
    Get the test checkpoint.

    Argument/s:
        exp_dir_trial - Experiment directory for the trial (where the checkpoints are saved).
        monitor_mode - Metric motitoring mode, either "min" or "max".
        test_epoch - epoch to test.
        test_ckpt_path - path to checkpoint to be tested.

    Returns:
        Path to the epoch's checkpoint.
    """

    set_options_bool = list(map(bool, [test_epoch is not None, test_ckpt_path]))
    assert set_options_bool.count(True) <= 1, f'Both "test_epoch" and "test_ckpt_path" cannot be set.'

    if test_ckpt_path:
        assert os.path.isfile(test_ckpt_path), f'File does not exist: {test_ckpt_path}.'
        ckpt_path = test_ckpt_path
    elif test_epoch is not None:
        ckpt_path = get_epoch_ckpt_path(exp_dir_trial, load_epoch=test_epoch)
    else:
        ckpt_path = get_best_ckpt_path(exp_dir_trial, monitor_mode)

    return ckpt_path


def write_test_ckpt_path(ckpt_path: str, exp_dir_trial: str):
    """
    Write ckpt_path used for testing to a text file.

    Argument/s:
        ckpt_path - path to the checkpoint of the epoch that scored
            highest for the given validation metric.
        exp_dir_trial - Experiment directory for the trial.
    """
    with open(os.path.join(exp_dir_trial, 'test_ckpt_path.txt'), 'a') as f:
        f.write(ckpt_path + "\n")


def resume_from_ckpt_path(
        exp_dir_trial: str,
        resumable: bool = False,
        resume_epoch: int = None,
        resume_ckpt_path: str = None,
):
    """
    Resume training from the specified checkpoint.

    Argument/s:
        exp_dir_trial - Experiment directory for the trial (where the checkpoints are saved).
        resumable - if training is automatically resumable (i.e., provide last.ckpt as the path).
        resume_epoch - get the path of the checkpoint for a given epoch.
        resume_ckpt_path - outright provide the checkpoint path.

    Returns:
          ckpt_path - path to a checkpoint.
    """

    ckpt_path = None

    options = ['resumable', 'resume_epoch', 'resume_ckpt_path']
    set_options_bool = list(map(bool, [resumable, resume_epoch is not None, resume_ckpt_path]))
    set_options = [j for i, j in enumerate(options) if set_options_bool[i]]
    assert set_options_bool.count(True) in [0, 1], \
        f'A maximum of one of these options can be set: {options}. The following are set: {set_options}.'

    if resumable:
        last_ckpt_path = os.path.join(exp_dir_trial, 'last.ckpt')

        # last.ckpt will be a directory with deepspeed:
        if os.path.isfile(last_ckpt_path) or os.path.isdir(last_ckpt_path):
            ckpt_path = os.path.join(exp_dir_trial, 'last.ckpt')
        else:
            warnings.warn('last.ckpt does not exist, starting training from epoch 0.')
    elif resume_epoch is not None:
        ckpt_path = get_epoch_ckpt_path(exp_dir_trial, load_epoch=resume_epoch)
    elif resume_ckpt_path:
        ckpt_path = resume_ckpt_path

    if ckpt_path is not None:
        print('Resuming training from {}.'.format(ckpt_path))

    return ckpt_path


def gpu_usage_and_visibility(cuda_visible_devices: Optional[str] = None, submit: bool = False):
    """
    Prints out GPU utilisation and sets the visibility of the GPUs to the job.

    Argument/s:
        cuda_visible_devices - which GPUs are visible to the job.
        submit - if the job is being submitted to a cluster manager, ignore GPU visibility requirements.
    """

    # Print GPU usage
    for gpu in GPUtil.getGPUs():
        print(f'Initial utilisation on GPU:{gpu.id} is {gpu.memoryUtil:0.3f}.')

    # Determine the visibility of devices
    if cuda_visible_devices and (not submit):
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
        print(f'CUDA_VISIBLE_DEVICES: {cuda_visible_devices}')


class suppress_stdout_stderr(object):
    """
    Suppress STDOUT and STDERR.
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]

        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
