import copy
import os
import sys
from pathlib import Path

from omegaconf.listconfig import ListConfig

from dlhpcstarter.cluster import ClusterSubmit
from dlhpcstarter.command_line_arguments import read_command_line_arguments
from dlhpcstarter.utils import importer, load_config_and_update_args


def submit(args, cmd_line_args, stages_fnc):
    """
    4. Submit the job to the cluster manager (or run locally if args.submit = False).
    """
    
    if not args.submit:

        # Run locally:
        stages_fnc(args)

    else:

        # Cluster object:
        cluster = ClusterSubmit(
            args=args,
            fnc=stages_fnc,
            save_dir=args.exp_dir_trial,
            time_limit=args.time_limit,
            num_gpus=args.devices,
            num_nodes=args.num_nodes,
            num_workers=args.num_workers,
            memory=args.memory,
            python_cmd='python3' if not hasattr(args, 'python_cmd') else args.python_cmd,
            entrypoint='dlhpcstarter' if not hasattr(args, 'entrypoint') else args.entrypoint,
            no_cpus_per_task=args.no_cpus_per_task,
            no_gpus_per_node=args.no_gpus_per_node,
            no_ntasks_per_node=args.no_ntasks_per_node,
            srun_options=args.srun_options,
            email=args.email, 
            email_on_complete=True, 
            email_on_fail=True,
            cmd_line_args=cmd_line_args,
            auto_resubmit_method='signal' if not hasattr(args, 'auto_resubmit_method') else args.auto_resubmit_method,
            auto_resubmit=args.auto_resubmit,
        )

        # Source virtual environment:
        if args.venv_path:
            cluster.add_command('source ' + args.venv_path)

        # Add options to the cluster manager submission script:
        if 'cluster_manager_script_commands' in args:
            for i in args.cluster_manager_script_commands:
                cluster.add_command(i)

        # Add options to the cluster manager submission script:
        if 'cluster_manager_script_clean_up_commands' in args:
            for i in args.cluster_manager_script_clean_up_commands:
                cluster.add_clean_up_command(i)

        # Add cluster manager options:
        if 'cluster_manager_options' in args:
            for i in args.cluster_manager_options:
                cluster.add_manager_option(option=i[0], value=i[1])

        # Request the quality of service for the job:
        if args.qos:
            cluster.add_manager_option(option='qos', value=args.qos)

        # Submit job to workload manager:
        job_display_name = args.task + '_' + args.config_name

        if args.trial is not None:
            job_display_name = job_display_name + f'_trial_{args.trial}'

        cluster.submit(job_display_name=job_display_name + '_')

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
    cmd_line_args = read_command_line_arguments()

    """
    2. Load the configuration for the job and add it to 'args':
    
        This contains the paths, model configuration, the training and test configuration, the device & cluster manager 
        configuration.
    """
    args, cmd_line_args = load_config_and_update_args(cmd_line_args=cmd_line_args)

    """
    3. Import the 'stages' function for the task:
    
        Imports the function that handles the training and testing stages for the task. The default location of the 
        stages() function is in the task's stages.py. The model is also initialised in the stages function based on the 
        configuration.
        
        The definition and module of the stages function can also be manually set using 'stages_definition' and 
        'stages_module', respectively.
    
        For example: stages() in task.cifar10.stages 
    
    """
    args.stages_definition = 'stages' if args.stages_definition is None else args.stages_definition
    args.stages_module = 'stages_module' if args.stages_module is None else args.stages_module
    stages_fnc = importer(definition=args.stages_definition, module=args.stages_module)

    args.search_space = None if 'search_space' not in args else args.search_space
    args.search_space_ignore_keys = [] if 'search_space_ignore_keys' not in args else args.search_space_ignore_keys

    if args.search_space is None or bool(args.manager_script_path):
        submit(args, cmd_line_args, stages_fnc)
        
    else:

        # Check if path:
        def is_path(string):
            if not isinstance(string, str):
                return False
            try:
                path = Path(string)
                return len(path.parts) > 1
            except ValueError:
                return False

        # Search space:
        def format_dict(d, ignore=None):
            parts = []
            for key, value in d.items():
                if is_path(value) or key in ignore:
                    continue
                if isinstance(value, ListConfig):
                    value = '_'.join(map(str, value))
                parts.append(f'{key}_{value}')
            return '_'.join(parts)

        # Get the length of the first list in the dictionary to compare against:
        first_list_len = len(next(iter(args.search_space.values())))

        # Assert that all lists in the dictionary have the same length:
        assert all(len(i) == first_list_len for i in args.search_space.values()), "Not all lists in the search space have the same length."

        base_config_name = args.config_name
        keys, lists = zip(*args.search_space.items())
        for values in zip(*lists):
            args_copy = copy.deepcopy(args)
            cmd_line_args_copy = copy.deepcopy(args)
            config_changes = dict(zip(keys, values))
            for key, value in config_changes.items():
                setattr(args_copy, key, value)
            
            args_copy.config_name = base_config_name + '_' + format_dict(config_changes, args.search_space_ignore_keys)
            print(f'Running search config: {args_copy.config_name}')
            args_copy.exp_dir_trial = os.path.join(args_copy.exp_dir, args_copy.task, args_copy.config_name, 'trial_' + f'{args_copy.trial}')
            cmd_line_args_copy.exp_dir_trial = args_copy.exp_dir_trial
            
            del args_copy.search_space

            submit(args_copy, cmd_line_args_copy, stages_fnc)


if __name__ == '__main__':
    main()
