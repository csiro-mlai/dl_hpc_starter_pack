import copy
import datetime
import glob
import importlib
import os
import re
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Optional, Pattern, Union
from collections import OrderedDict

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir


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


def load_config_and_update_args(cmd_line_args: Namespace, print_args: bool = False) -> None:
    """
    Loads the configuration .yaml file and updates the args object.

    Argument/s:
        cmd_line_args - command line arguments object.
        print_args - print the arguments for the job.
    """
    # Make a deepcopy of the command line arguments:
    args = copy.deepcopy(cmd_line_args)

    # Add the working directory to paths:
    args.work_dir = args.work_dir if 'work_dir' in args else None
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

    # Add the task, configuration name, and the trial number to the experiment directory:
    args.trial = args.trial if args.trial is not None else 0
    args.exp_dir_trial = os.path.join(args.exp_dir, args.task, args.config_name, 'trial_' + f'{args.trial}')
    Path(args.exp_dir_trial).mkdir(parents=True, exist_ok=True)

    # Do not resume from last if fast_dev_run is True:
    # args.resume_last = False if args.fast_dev_run else True

    # Prevent auto-resubmit if not resume_last:
    args.resume_last, args.auto_resubmit = True, True
    if args.resume_ckpt_path or args.resume_epoch:
        args.resume_last, args.auto_resubmit = False, False
        
    if print_args:
        print(f'args: {args.__dict__}')

    # Print GPU usage and set GPU visibility:
    gpu_usage_and_visibility(args.cuda_visible_devices, args.submit)

    return args, cmd_line_args


def get_epoch_ckpt_path(exp_dir_trial: str, load_epoch: int, extension: Union[list, str] = ['', '.ckpt']) -> str:
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


def get_best_ckpt_path(exp_dir_trial: str, monitor: str, monitor_mode: str, extension: Union[list, str] = ['', '.ckpt']) -> str:
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

    extension = list(extension) if isinstance(extension, str) else extension

    ckpt_list = list(OrderedDict.fromkeys([j for i in extension for j in glob.glob(os.path.join(exp_dir_trial, f'*=*{monitor}=*{i}'))]))

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
        extension: Optional[Union[list, str]] = ['', '.ckpt'],
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
        assert os.path.isfile(test_ckpt_path) or os.path.isdir(test_ckpt_path), f'Checkpoint does not exist: {test_ckpt_path}.'
        ckpt_path = test_ckpt_path
    elif test_epoch is not None:
        ckpt_path = get_epoch_ckpt_path(exp_dir_trial, test_epoch, extension)
    else:
        ckpt_path = get_best_ckpt_path(exp_dir_trial, monitor, monitor_mode, extension)

    return ckpt_path


def write_test_ckpt_path(ckpt_path: str, exp_dir_trial: str, file_name: str = 'test_ckpt_path'):
    """
    Write ckpt_path used for testing to a text file.

    Argument/s:
        ckpt_path - path to the checkpoint of the epoch that scored
            highest for the given validation metric.
        exp_dir_trial - experiment directory for the trial.
        file_name - name of the text file.
    """
    with open(os.path.join(exp_dir_trial, f'{file_name}.txt'), 'a') as f:
        f.write(ckpt_path + '\n')


def resume_from_ckpt_path(
        exp_dir_trial: str,
        resume_last: bool = False,
        resume_epoch: int = None,
        resume_ckpt_path: str = None,
        extension: Union[list, str] = ['', '.ckpt'],
    ):
    """
    Resume training from the specified checkpoint.

    Argument/s:
        exp_dir_trial - Experiment directory for the trial (where the checkpoints are saved).
        resume_last - resume from last epoch.
        resume_epoch - get the path of the checkpoint for a given epoch.
        resume_ckpt_path - outright provide the checkpoint path.
        extension - checkpoint extension (None for automatic detection of extension).

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
        last_ckpt_path = list(OrderedDict.fromkeys([j for i in extension for j in glob.glob(os.path.join(exp_dir_trial, f'last{i}'))]))       
        assert len(last_ckpt_path) <= 1, f'There cannot be more than one file or directory with "last" as the name: {last_ckpt_path}'
        if not last_ckpt_path:
            warnings.warn('The "last" checkpoint does not exist, starting training from epoch 0.')
        ckpt_path = last_ckpt_path[0] if last_ckpt_path else ckpt_path
    elif resume_epoch is not None:
        ckpt_path = get_epoch_ckpt_path(exp_dir_trial, resume_epoch, extension)
    
    elif resume_ckpt_path:
        ckpt_path = resume_ckpt_path

    if ckpt_path is not None:
        print('Resuming training from {}.'.format(ckpt_path))

    return ckpt_path


def get_experiment_scores_csv_paths(
        config_name: str, 
        trial: Union[str, int], 
        task: str, 
        exp_dir: str, 
        log_dir: str = 'lightning_logs', 
        glob_filter: str = '**/*.csv', 
        start_cutoff_date: Optional[datetime.datetime] = None,
        end_cutoff_date: Optional[datetime.datetime] = None,
        date_regex: Pattern[str] = r'([0-9]{2}\-[0-9]{2}\-[0-9]{4}\_[0-9]{2}\-[0-9]{2}\-[0-9]{2})',
        strptime_format: str = '%d-%m-%Y_%H-%M-%S',
        trial_prefix: str = 'trial_' 
    ):
    regex = os.path.join(exp_dir, task, config_name, f'{trial_prefix}{trial}', log_dir, glob_filter)
    csv_logs = glob.glob(regex, recursive=True)

    if len(csv_logs) == 0:
        warnings.warn(f'No files found for {regex}.')
    else:
        datetime_list = []
        for i in csv_logs:
            if bool(re.search(date_regex, i)):     
                datetime_list.append(datetime.datetime.strptime(re.search(date_regex, i)[1], strptime_format))
            else:
                datetime_list.append(datetime.datetime.fromtimestamp(os.path.getctime(i)))

        if start_cutoff_date:
            csv_logs = [i for i, j in zip(csv_logs, datetime_list) if j > start_cutoff_date]
            datetime_list = [i for i in datetime_list if i > start_cutoff_date]  # Remove here for end_cuttoff_date.

        if end_cutoff_date:
            csv_logs = [i for i in csv_logs if datetime.datetime.fromtimestamp(os.path.getctime(i)) < end_cutoff_date]

    # Sort base on last modification date:
    csv_logs.sort(key=lambda x: os.path.getmtime(x))

    return csv_logs


def read_and_concatenate_experiment_scores(config_name, trial, only_most_recent=False, **kwargs):
    csv_logs = get_experiment_scores_csv_paths(config_name=config_name, trial=trial, **kwargs)
    csv_logs = csv_logs[-1:] if only_most_recent else csv_logs
    return pd.concat([pd.read_csv(f) for f in csv_logs], ignore_index=True) if csv_logs else None 


def get_experiment_best_scores(config_name, trial, monitor, monitor_mode, subset, **kwargs):
    assert monitor_mode == 'max' or monitor_mode == 'min'
    df = read_and_concatenate_experiment_scores(config_name=config_name, trial=trial, **kwargs)
    if isinstance(df, pd.DataFrame):
        best_epoch = df['epoch'][df[monitor].idxmax()] if monitor_mode == 'max' else df['epoch'][df[monitor].idxmin()]
        df = df[df['epoch'] == best_epoch]
        df = df[df.columns[df.columns.str.startswith(f'{subset}_')].tolist() + ['epoch']]
        df = df.dropna(how='any').copy() #.iloc[[-1]]
        df.insert(0, 'trial', trial) 
        df.insert(0, 'config', config_name) 
    return df


def get_experiment_last_scores(config_name, trial, subset, **kwargs):
    df = read_and_concatenate_experiment_scores(config_name=config_name, trial=trial, **kwargs)
    if isinstance(df, pd.DataFrame):
        df = df[df.columns[df.columns.str.startswith(f'{subset}_')]]
        df = df.iloc[[-1]]
        df.insert(0, 'trial', trial) 
        df.insert(0, 'config', config_name) 
    return df


def get_experiment_all_scores(config_name, trial, subset, **kwargs):
    df = read_and_concatenate_experiment_scores(config_name=config_name, trial=trial, **kwargs)
    if isinstance(df, pd.DataFrame):
        df = df[df.columns[df.columns.str.startswith(f'{subset}_')].tolist() + ['epoch', 'step']].copy()
        df.insert(0, 'trial', trial) 
        df.insert(0, 'config', config_name) 
    return df


def get_config_scores(config_trial_list, score_type, **kwargs):
    df_list = []
    for i in config_trial_list:
        if score_type == 'last':
            df = get_experiment_last_scores(config_name=i['config'], trial=i['trial'], **kwargs)
        elif score_type == 'best':
            df = get_experiment_best_scores(config_name=i['config'], trial=i['trial'], **kwargs)
        elif score_type == 'all':
            df = get_experiment_all_scores(config_name=i['config'], trial=i['trial'], **kwargs)
        if isinstance(df, pd.DataFrame):
            df_list.append(df)
    return pd.concat(df_list)


def get_melted_config_scores(id_vars=['config', 'trial'], **kwargs):
    df = get_config_scores(**kwargs)
    df = df.melt(
        id_vars=id_vars,
        var_name='metric',
        value_name='score',
    )
    return df


def get_config_epochs_steps(config_name, trial, **kwargs):
    df = read_and_concatenate_experiment_scores(config_name=config_name, trial=trial, **kwargs)
    if isinstance(df, pd.DataFrame):
        df = df[['epoch', 'step']]
        df.insert(0, 'trial', trial) 
        df.insert(0, 'config', config_name) 
    return df


def get_config_max_epochs(config_trial_list, **kwargs):
    df_list = []
    for i in config_trial_list:
        df = get_config_epochs_steps(config_name=i['config'], trial=i['trial'], **kwargs)
        if isinstance(df, pd.DataFrame):
            df_list.append(df)
    df = pd.concat(df_list).reset_index()
    return df.loc[df.groupby(['config', 'trial'])['epoch'].idxmax()].reset_index(drop=True)


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
 