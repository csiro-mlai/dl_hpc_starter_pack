from argparse import Namespace
from pytorch_lightning.utilities.seed import seed_everything
from lib.tools.ext.collect_env_details import main as collect_env_details
from lib.trainer import trainer_instance
from lib.utils import get_test_ckpt_path, importer, load_config_and_update_args, write_test_ckpt_path


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

    # Trainer
    trainer = trainer_instance(**vars(args))

    # Model definition
    model_def = importer(definition=args.definition, module='.'.join(['task', args.task, 'model', args.module]))

    # Train
    if args.train:

        # Model (includes dataset)
        model = model_def(**vars(args))

        # Train
        trainer.fit(model)

    # Test
    if args.test:

        if args.debug:

            model = model_def(**vars(args))

        else:

            # Get the path to the best performing checkpoint
            ckpt_path = get_test_ckpt_path(args.exp_dir, args.monitor_mode, args.test_epoch, args.test_ckpt_path)
            print('Testing checkpoint: {}.'.format(ckpt_path))
            write_test_ckpt_path(ckpt_path, args.exp_dir)

            model = model_def.load_from_checkpoint(checkpoint_path=ckpt_path, **vars(args))

        trainer.test(model)
