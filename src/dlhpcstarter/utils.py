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


import torch
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
            f'''{absolute_path} does not exist. The target definition and modules: {definition} & {module}. The working directory: {work_dir}.'''

        module = importlib.machinery.SourceFileLoader(module, absolute_path).load_module()

    else:

        path = os.path.join(*module.split('.')) + '.py'

        assert os.path.isfile(path), f'{path} does not exist. The target definition and modules: {definition} & {module}.'

        module = importlib.import_module(module)

    return getattr(module, definition)


def load_config_and_update_args(args: Namespace, print_args: bool = False) -> None:
    """
    Loads the configuration .yaml file and updates the args object.

    Argument/s:
        args - object that houses the arguments for the job.
        print_args - print the arguments for the job.
    """
    # Add the working directory to paths:
    if not args.work_dir:
        args.work_dir = os.getcwd()

    # Configuration:
    if args.config.endswith('.yaml'):
        args.config_file_name = args.config
        args.config = args.config.replace('.yaml', '')
    else:
        args.config_file_name = args.config + '.yaml'
    args.config_name = Path(args.config).parts[-1]

    # Load configuration using Hydra's Compose API:
    args.config_dir = Path(args.config).parent
    if not os.path.isabs(args.config_dir):
        args.config_dir = os.path.join(args.work_dir, args.config_dir)
    with initialize_config_dir(version_base=None, config_dir=args.config_dir):
        config = compose(config_name=args.config_name)

    # Update args with config and check for conflicts:
    args.config_full_path = os.path.join(args.work_dir, args.config_file_name)
    for k, v in config.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
        else:
            if getattr(args, k) != v:
                raise ValueError(f'There is a conflict between command line argument "--{k} {getattr(args, k)}" '
                                 f'({type(getattr(args, k))}) and configuration argument "{k}: {v}" from '
                                 f'{args.config_full_path} ({type(v)}).')

    # The model name must be defined in the configuration or in the command line arguments:
    assert args.module, f'"module" must be specified as a command line argument or in {args.config_full_path}.'
    assert args.definition, f'"definition" must be specified as a command line argument or in {args.config_full_path}.'
    assert args.exp_dir, f'"exp_dir" must be specified as a command line argument or in {args.config_full_path}.'

    # Defaults: There is probably a better place to do this:
    args.num_workers = args.num_workers if args.num_workers is not None else 1
    args.num_nodes = args.num_nodes if args.num_nodes is not None else 1
    if args.submit:
        args.time_limit = args.time_limit if args.time_limit is not None else '02:00:00'
        args.memory = args.memory if args.memory is not None else '16GB'

    # Add the task, configuration name, and the trial number to the experiment directory:
    args.trial = args.trial if args.trial is not None else 0
    args.exp_dir_trial = os.path.join(args.exp_dir, args.task, args.config_name, 'trial_' + f'{args.trial}')
    Path(args.exp_dir_trial).mkdir(parents=True, exist_ok=True)

    # Do not resume from last if fast_dev_run is True:
    args.resume_last = False if args.fast_dev_run else args.resume_last

    if print_args:
        print(f'args: {args.__dict__}')

    # Print GPU usage and set GPU visibility:
    gpu_usage_and_visibility(args.cuda_visible_devices, args.submit)


def get_epoch_ckpt_path(exp_dir_trial: str, load_epoch: int, extension: str = ".ckpt") -> str:
    """
    Get the checkpoint path based on the epoch number.

    Argument/s:
        exp_dir_trial - Experiment directory for the trial (where the checkpoints are saved).
        load_epoch - epoch to load.
        extension - checkpoint extension.

    Returns:
        Path to the epoch's checkpoint.
    """
    try:
        ckpt_path = glob.glob(os.path.join(exp_dir_trial, '*epoch=' + str(load_epoch) + f'*{extension}'))
        assert len(ckpt_path) == 1, f'Multiple checkpoints for epoch {load_epoch}: {ckpt_path}.'

    except:
        raise ValueError(
            'Epoch {} is not in the checkpoint directory.'.format(str(load_epoch))
        )
    return ckpt_path[0]


def get_best_ckpt_path(exp_dir_trial: str, monitor: str, monitor_mode: str, extension: str = '.ckpt') -> str:
    """
    Get the best performing checkpoint from the experiment directory.

    Argument/s:
        exp_dir_trial - Experiment directory for the trial (where the checkpoints are saved).
        monitor - Monitored metric.
        monitor_mode - Metric monitoring mode, either "min" or "max".
        extension - checkpoint extension.

    Returns:
        Path to the epoch's checkpoint.
    """

    ckpt_list = glob.glob(os.path.join(exp_dir_trial, f'*=*{monitor}=*{extension}'))

    if not ckpt_list:
        raise ValueError(f'No checkpoints exist for the regex: *=*{monitor}=*{extension} in the checkpoint directory: {exp_dir_trial}.')

    scores = [
        re.findall(r'[-+]?\d*\.\d+|\d+', i.rsplit('=', 1)[1])[0] for i in ckpt_list
    ]

    if monitor_mode == 'max':
        ckpt_path = ckpt_list[np.argmax(scores)]
    elif monitor_mode == 'min':
        ckpt_path = ckpt_list[np.argmin(scores)]
    else:
        raise ValueError("'monitor_mode' must be max or min, not {}.".format(monitor_mode))
    return ckpt_path


def get_test_ckpt_path(
        exp_dir_trial: str, 
        monitor: Optional[str] = None, 
        monitor_mode: Optional[str] = None, 
        test_epoch: Optional[int] = None, 
        test_ckpt_path: Optional[str] = None,
        extension: Optional[str] = '.ckpt',
) -> str:
    """
    Get the test checkpoint.

    Argument/s:
        exp_dir_trial - Experiment directory for the trial (where the checkpoints are saved).
        monitor - Monitored metric.
        monitor_mode - Metric monitoring mode, either "min" or "max".
        test_epoch - epoch to test.
        test_ckpt_path - path to checkpoint to be tested.
        extension - checkpoint extension.

    Returns:
        Path to the epoch's checkpoint.
    """

    set_options_bool = list(map(bool, [test_epoch is not None, test_ckpt_path]))
    assert set_options_bool.count(True) <= 1, f'Both "test_epoch" and "test_ckpt_path" cannot be set.'

    if test_ckpt_path:
        assert os.path.isfile(test_ckpt_path), f'File does not exist: {test_ckpt_path}.'
        ckpt_path = test_ckpt_path
    elif test_epoch is not None:
        ckpt_path = get_epoch_ckpt_path(exp_dir_trial, test_epoch, extension)
    else:
        ckpt_path = get_best_ckpt_path(exp_dir_trial, monitor, monitor_mode, extension)

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
        f.write(ckpt_path + '\n')


def resume_from_ckpt_path(
        exp_dir_trial: str,
        resume_last: bool = False,
        resume_epoch: int = None,
        resume_ckpt_path: str = None,
        extension: str = '.ckpt',
):
    """
    Resume training from the specified checkpoint.

    Argument/s:
        exp_dir_trial - Experiment directory for the trial (where the checkpoints are saved).
        resume_last - resume from last epoch.
        resume_epoch - get the path of the checkpoint for a given epoch.
        resume_ckpt_path - outright provide the checkpoint path.
        extension - checkpoint extension.

    Returns:
          ckpt_path - path to a checkpoint.
    """

    ckpt_path = None

    options = ['resume_last', 'resume_epoch', 'resume_ckpt_path']
    set_options_bool = list(map(bool, [resume_last, resume_epoch is not None, resume_ckpt_path]))
    set_options = [j for i, j in enumerate(options) if set_options_bool[i]]
    assert set_options_bool.count(True) in [0, 1], \
        f'A maximum of one of these options can be set: {options}. The following are set: {set_options}.'

    if resume_last:
        last_ckpt_path = os.path.join(exp_dir_trial, f'last{extension}')

        # last.ckpt will be a directory with deepspeed:
        if os.path.isfile(last_ckpt_path) or os.path.isdir(last_ckpt_path):
            ckpt_path = os.path.join(exp_dir_trial, f'last{extension}')
        else:
            warnings.warn('last.ckpt does not exist, starting training from epoch 0.')
    elif resume_epoch is not None:
        ckpt_path = get_epoch_ckpt_path(exp_dir_trial, resume_epoch, extension)
        raise ValueError('"resume_epoch" is never used and will be removed in the future as it interferes with the resume from resume_from_ckpt_path logic.')
    
    elif resume_ckpt_path:
        ckpt_path = resume_ckpt_path
        raise ValueError('"resume_ckpt_path" is never used and will be removed in the future as it interferes with the resume from resume_from_ckpt_path logic.')

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

    # Print GPU usage:
    # if torch.cuda.is_available():
    #     try:
    #         for gpu in GPUtil.getGPUs():
    #             print(f'Initial utilisation on GPU:{gpu.id} is {gpu.memoryUtil:0.3f}.')
    #     except Exception as e:
    #         print(f'NVIDIA-SMI is not working: {e}.')

    # Determine the visibility of devices:
    if cuda_visible_devices and (not submit):
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
        print(f'CUDA_VISIBLE_DEVICES: {cuda_visible_devices}')
        