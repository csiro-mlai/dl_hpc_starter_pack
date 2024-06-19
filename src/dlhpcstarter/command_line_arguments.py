import argparse
import ast


def dict_type(arg_str):
    try:
        return ast.literal_eval(arg_str)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid dictionary format")


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

    parser = argparse.ArgumentParser(description='Command line arguments')

    # Required arguments:
    required = parser.add_argument_group('Required named arguments')
    required.add_argument('--task', '-t', type=str, help='The name of the task', required=True)
    required.add_argument('--config', '-c', type=str, help='Configuration name in task/TASK_NAME/config', required=True)

    # Directory paths:
    directories = parser.add_argument_group('Directory paths')
    directories.add_argument('--exp-dir', '--exp_dir', type=str, help='Experiment outputs save directory')
    directories.add_argument('--exp-dir-trial', '--exp_dir_trial', type=str, help='Experiment outputs save directory for the trial')
    directories.add_argument('--work-dir', '--work_dir', type=str, help='Working directory')
    directories.add_argument('--dataset-dir', '--dataset_dir', type=str, help='The dataset directory')
    directories.add_argument('--ckpt-zoo-dir', '--ckpt_zoo_dir', type=str, help='The checkpoint zoo directory')

    # Model module and definition:
    model = parser.add_argument_group('Model module name and definition')
    model.add_argument('--definition', type=str, help='Class definition of the model')
    model.add_argument('--module', type=str, help='Name of the module')

    # Stages module and definition:
    model = parser.add_argument_group('Stages module name and definition')
    model.add_argument('--stages_definition', type=str, help='Definition of stages')
    model.add_argument('--stages_module', type=str, help='Name of the module')

    # Training arguments:
    training_arguments = parser.add_argument_group('Training arguments')
    training_arguments.add_argument('--train',  default=None, action='store_true', help='Perform training')
    training_arguments.add_argument('--trial', type=int, help='The trial number')
    training_arguments.add_argument('--resume-epoch', '--resume_epoch', type=int, help='Epoch to resume training from')
    training_arguments.add_argument(
        '--resume-ckpt-path', '--resume_ckpt_path', type=str, help='Checkpoint to resume training from',
    )
    training_arguments.add_argument(
        '--warm-start-ckpt-path', '--warm_start_ckpt_path', type=str, help='Checkpoint for warm-starting',
    )
    training_arguments.add_argument('--monitor', type=str, help='Metric to monitor')
    training_arguments.add_argument(
        '--monitor-mode',
        '--monitor_mode',
        type=str,
        help='whether the monitored metric is to be maximised or minimised (''max'' or ''min'')',
    )
    training_arguments.add_argument('--warm-start-ckpt-path-strict', '--warm_start_ckpt_path_strict', default=None, action='store_true', help='Strict checking of the checkpoint for warm starting')

    # Test arguments:
    test = parser.add_argument_group('Testing arguments')
    test.add_argument('--test', default=None, action='store_true', help='Evaluate the model on the test set')
    test.add_argument('--validate', default=None, action='store_true', help='Evaluate the model on the validation set')
    test.add_argument('--test-epoch', '--test_epoch', type=int, help='Test epoch')
    test.add_argument('--test-ckpt-path', '--test_ckpt_path', type=str, help='Path to checkpoint to be tested')
    test.add_argument('--test-ckpt-name', '--test_ckpt_name', type=str, help='Name of the Hugging Face Hub checkpoint to be tested')
    test.add_argument('--other-exp-dir', '--other_exp_dir', type=str, help='Test the checkpoints of another configuration')
    test.add_argument('--test-without-ckpt', '--test_without_ckpt', default=None, action='store_true', help='Test the model without loading a checkpoint')

    # PyTorch Lightning Trainer arguments:
    trainer = parser.add_argument_group('PyTorch Lightning Trainer arguments')
    trainer.add_argument('--fast-dev-run', '--fast_dev_run',  default=None, action='store_true', help='https://lightning.ai/docs/pytorch/stable/common/trainer.html#fast-dev-run')

    # Distributed computing arguments:
    distributed = parser.add_argument_group('Distributed computing arguments')
    distributed.add_argument('--num-workers', '--num_workers', type=int, help='No. of workers per DataLoader & GPU')
    distributed.add_argument('--devices', type=int, help='Number of devices per node')
    distributed.add_argument('--num-nodes', '--num_nodes', type=int, help='Number of nodes')
    distributed.add_argument('--force-single-device', '--force_single_device', default=None, action='store_true', help='Overwrites the number of devices and nodes specified in the configuration to one')
    distributed.add_argument('--force-single-node', '--force_single_node', default=None, action='store_true', help='Overwrites the number of nodes specified in the configuration to one')
    
    # Cluster manager arguments:
    cluster = parser.add_argument_group('Cluster manager arguments')
    cluster.add_argument('--memory', type=str, help='Amount of memory per node')
    cluster.add_argument('--time-limit', '--time_limit', type=str, help='Job time limit')
    cluster.add_argument('--submit',  default=None, action='store_true', help='Submit job to the cluster manager')
    cluster.add_argument('--qos', type=str, help='Quality of service')
    cluster.add_argument('--begin', type=str, help='When to begin the Slurm job, e.g. now+1hour')
    cluster.add_argument('--manager-script-path', '--manager_script_path', type=str)
    cluster.add_argument('--srun-options', '--srun_options', type=str, help='Options for srun')
    cluster.add_argument('--email', type=str, help='Email for cluster manager notifications')
    cluster.add_argument('--no-cpus-per-task', '--no_cpus_per_task', default=None, action='store_true', help='Prevent the --cpus-per-task option from being placed in the Slurm script')
    cluster.add_argument('--no-gpus-per-node', '--no_gpus_per_node', default=None, action='store_true', help='Prevent the --gpus-per-node option from being placed in the Slurm script')
    cluster.add_argument('--no-ntasks-per-node', '--no_ntasks_per_node', default=None, action='store_true', help='Prevent the --no-ntasks-per-node option from being placed in the Slurm script')
    cluster.add_argument('--one-epoch-only', '--one_epoch_only', default=None, action='store_true', help='Only performs one epoch of training')
    # cluster.add_argument('--auto-resubmit-method', '--auto_resubmit_method', type=str, help='Auto resubmission method')

    # System arguments:
    system = parser.add_argument_group('System arguments')
    system.add_argument('--cuda-visible-devices', '--cuda_visible_devices', type=str, help='Visible CUDA devices')
    system.add_argument('--venv-path', '--venv_path', type=str, help='Path to ''bin/activate'' of the virtualenv')
    
    # Dataloader arguments:
    dataloader = parser.add_argument_group('Dataloader arguments')        
    dataloader.add_argument('--dataloader-zero-workers', '--dataloader_zero_workers', default=None, action='store_true', help='Use zero workers with the dataloader.')
    dataloader.add_argument('--mbatch-size-one', '--mbatch_size_one', default=None, action='store_true', help='Sets the mini-batch size to one.')
    
    # General arguments:
    general = parser.add_argument_group('General arguments')
    general.add_argument('--debug', default=None, action='store_true', help='A general debugging flag')
    general.add_argument('--float32-matmul-precision', '--float32_matmul_precision', default=None, type=str, help='Tensor core precision')

    args, unknown = parser.parse_known_args()

    return args
