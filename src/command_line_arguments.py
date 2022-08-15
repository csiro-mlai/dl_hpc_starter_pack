from argparse import ArgumentParser


def str_to_bool(s):
    return s.lower() in ('yes', 'true', 't', '1')


def read_command_line_arguments():
    """
    Reads the command line arguments.

    Partial parsing is used. This is because the main function is executed twice when using a cluster manager, with the
    second run having extra arguments. If partial parsing is not used, arguments from the configuration file that are
    not defined by the argparse will be stored as command line arguments in the manager submission script during the
    first pass and then fed to the argparse on the second pass. Hence, ignoring the unknown arguments using partial
    parsing is necessary for the current setup.

    Partial parsing:
        https://docs.python.org/3/library/argparse.html#partial-parsing

    Returns:
        Object containing the model's configuration.
    """

    parser = ArgumentParser()

    # Directories
    parser.add_argument('--exp-dir', '--exp_dir', default=None, type=str, help='Experiment outputs save directory')
    parser.add_argument('--work-dir', '--work_dir', default=None, type=str, help='Working directory')
    parser.add_argument('--dataset-dir', '--dataset_dir', default=None, type=str, help='The dataset directory')
    parser.add_argument('--ckpt-zoo-dir', '--ckpt_zoo_dir', default=None, type=str, help='The checkpoint zoo directory')

    # Configuration name and task arguments
    parser.add_argument('--task', type=str, help='The name of the task')
    parser.add_argument('--config', type=str, help='Name of the configuration in task/TASK_NAME/config')
    parser.add_argument('--definition', type=str, help='Class definition of the model')
    parser.add_argument('--module', type=str, help='Name of the module in task/TASK_NAME/model')

    parser.add_argument('--model', type=str, help='Name of the model in task/TASK_NAME/model')  # depreciated.

    # Training arguments
    parser.add_argument('--train', default=False, type=str_to_bool, help='Perform training')
    parser.add_argument(
        '--resumable', default=False, type=str, help='Resumable training. Automatic resubmission to cluster manager',
    )
    parser.add_argument(
        '--resume-epoch', '--resume_epoch', default=None, type=int, help='Epoch to resume training from',
    )
    parser.add_argument(
        '--resume-ckpt-path', '--resume_ckpt_path', default=None, type=str, help='Checkpoint to resume training from',
    )

    # Inference & test arguments
    parser.add_argument('--test', default=False, type=str_to_bool, help='Evaluate the model on the test set')
    parser.add_argument('--test-epoch', '--test_epoch', default=None, type=int, help='Test epoch')
    parser.add_argument(
        '--test-ckpt-path', '--test_ckpt_path', default=None, type=str, help='Path to checkpoint to be tested',
    )

    # Model arguments
    parser.add_argument(
        '--warm-start-ckpt-path', '--warm_start_ckpt_path', default=None, type=str, help='Checkpoint for warm-starting',
    )

    # PyTorch Lightning arguments
    parser.add_argument(
        '--debug', default=False, type=str_to_bool, help='One mini-batch for training, validation, and testing',
    )

    # Cluster manager & hardware arguments
    parser.add_argument(
        '--num-workers', '--num_workers', default=1, type=int, help='No. of workers per DataLoader & GPU'
    )
    parser.add_argument('--num-gpus', '--num_gpus', default=1, type=int, help='Number of GPUs per node')
    parser.add_argument('--num-nodes', '--num_nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--memory', default='16GB', type=str, help='Amount of memory per node')
    parser.add_argument('--time-limit', '--time_limit', default='02:00:00', type=str, help='Job time limit')
    parser.add_argument('--submit', default=False, type=str_to_bool, help='Submit job to the cluster manager')
    parser.add_argument('--qos', default=None, type=str, help='Quality of service')
    parser.add_argument('--begin', default='now', type=str, help='When to begin the Slurm job, e.g. now+1hour')
    parser.add_argument('--slurm-cmd-path', '--slurm_cmd_path', type=str)
    parser.add_argument('--email', default=None, type=str, help='Email for cluster manager notifications')

    # Job arguments
    parser.add_argument('--trial', default=0, type=int, help='The trial number')

    # parser.add_argument('--trial', default=None, type=int, help='The trial number')
    # parser.add_argument(
    #     '--trials', default=None, help='The trials for a model (could be an integer or a list of integers)',
    # )

    # PyCharm arguments
    parser.add_argument('--mode', default=str)

    # System arguments
    parser.add_argument(
        '--cuda-visible-devices', '--cuda_visible_devices', default=None, type=str, help='Visible CUDA devices'
    )
    parser.add_argument('--venv-path', '--venv_path', type=str, help='Path to ''bin/activate'' of the virtualenv')

    args, unknown = parser.parse_known_args()

    return args
