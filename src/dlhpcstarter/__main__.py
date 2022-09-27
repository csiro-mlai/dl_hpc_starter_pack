from argparse import Namespace
from dlhpcstarter.command_line_arguments import read_command_line_arguments
from dlhpcstarter.utils import importer, gpu_usage_and_visibility, load_config_and_update_args
from dlhpcstarter.cluster import ClusterSubmit
from typing import Callable
import sys


def main() -> None:
    """
    The main function handles the jobs for a task. It does the following in order:
        1. Get command line arguments using argparse (or input your own arguments object to main()).
        2. Import the 'stages' function for the task.
        3. Load the configuration for the job and add it to 'args'.
        4. Submit the job to the cluster manager (or run locally).
    """

    if sys.path[0] != '':
        sys.path.insert(0, '')

    """    
    1. Get command line arguments using argparse for the job:
    """
    args = read_command_line_arguments()

    """
    2. Import the 'stages' function for the task:
    
        Imports the function that handles the training and testing stages for the task. This is the stages() function
        defined in the task's stages.py. The model is also defined in the stages function based on the configuration.
    
        For example: stages() in task.cifar10.stages 
    
    """
    stages_fnc = importer(definition='stages', module='.'.join(['task', args.task, 'stages']))

    """
    3. Load the configuration for the job and add it to 'args':
    
        This contains the paths, model configuration, the training and test configuration, the device & cluster manager 
        configuration.
    """
    load_config_and_update_args(args=args, print_args=True)

    """
    4. Submit the job to the cluster manager (or run locally if args.submit = False).
    """
    submit(args=args, stages_fnc=stages_fnc)


def submit(stages_fnc: Callable, args: Namespace):
    """
    Submit the job to the cluster manager, e.g., SLURM, or run the job locally, i.e., args.submit = False.

    Argument/s:
        stages_fnc - function that handles the creation, training, and testing of a model.
        args - arguments containing the configuration for the job and the model
    """

    if not args.submit:

        # Run locally
        stages_fnc(args)

    else:

        # Cluster object
        cluster = ClusterSubmit(
            fnc_kwargs=args,
            fnc=stages_fnc,
            save_dir=args.exp_dir_trial,
            time_limit=args.time_limit,
            num_gpus=args.num_gpus,
            num_nodes=args.num_nodes,
            num_workers=args.num_workers,
            memory=args.memory,
            python_cmd='',
            entrypoint='dlhpcstarter',
        )

        # Cluster commands
        cluster.add_manager_cmd(cmd='ntasks-per-node', value=args.num_gpus if args.num_gpus else 1)

        # Source virtual environment
        cluster.add_command('source ' + args.venv_path)

        # Debug flags
        cluster.add_command('export NCCL_DEBUG=INFO')
        cluster.add_command('export PYTHONFAULTHANDLER=1')

        # Request the quality of service for the job
        if args.qos:
            cluster.add_manager_cmd(cmd='qos', value=args.qos)

        # Email job status
        cluster.notify_job_status(email=args.email, on_done=True, on_fail=True)

        # Submit job to workload manager
        job_display_name = args.task + '_' + args.config

        if args.trial is not None:
            job_display_name = job_display_name + f'_trial_{args.trial}'

        cluster.submit(job_display_name=job_display_name + '_')


if __name__ == '__main__':
    main()
