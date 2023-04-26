import os
from argparse import Namespace
from lightning.pytorch import seed_everything
from dlhpcstarter.tools.ext.collect_env_details import main as collect_env_details
from dlhpcstarter.trainer import trainer_instance
from dlhpcstarter.utils import (
    get_test_ckpt_path,
    importer,
    load_config_and_update_args,
    resume_from_ckpt_path,
    write_test_ckpt_path,
)


def stages(args: Namespace):
    """
    Handles the training and testing stages for the task. This is the stages() function
        defined in the task's stages.py.

    Argument/s:
        args - an object containing the configuration for the job.
    """

    # Set seed number (using the trial number) for deterministic training
    seed_everything(args.trial, workers=True)

    # Print environment details
    collect_env_details()

    # Get configuration & update args attributes
    # Note: this needs to be called again for submitted jobs due to partial parsing.
    load_config_and_update_args(args)

    # Model definition
    TaskModel = importer(definition=args.definition, module=args.module)

    # Trainer
    trainer = trainer_instance(**vars(args))

    # Train
    if args.train:

        # For deterministic training: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # Warm-start from checkpoint
        if args.warm_start_ckpt_path:
            model = TaskModel.load_from_checkpoint(checkpoint_path=args.warm_start_ckpt_path, **vars(args))
            print('Warm-starting using: {}.'.format(args.warm_start_ckpt_path))
        else:
            model = TaskModel(**vars(args))

        # Compile model
        # model = torch.compile(model)

        # Train
        ckpt_path = resume_from_ckpt_path(args.exp_dir_trial, args.resume_last, args.resume_epoch, args.resume_ckpt_path)
        trainer.fit(model, ckpt_path=ckpt_path)

    # Test
    if args.test:

        if args.fast_dev_run:
            model = TaskModel(**vars(args))
        else:

            # Get the path to the best performing checkpoint
            ckpt_path = get_test_ckpt_path(args.exp_dir_trial, args.monitor_mode, args.test_epoch, args.test_ckpt_path)
            print('Testing checkpoint: {}.'.format(ckpt_path))
            write_test_ckpt_path(ckpt_path, args.exp_dir_trial)

            model = TaskModel.load_from_checkpoint(checkpoint_path=ckpt_path, **vars(args))

        # Compile model
        # model = torch.compile(model)

        trainer.test(model)
