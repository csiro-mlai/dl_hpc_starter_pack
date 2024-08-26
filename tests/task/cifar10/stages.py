import os
import re
import socket
import subprocess
from argparse import Namespace
from pathlib import Path

import torch
import transformers
from lightning.pytorch import seed_everything

from dlhpcstarter.trainer import trainer_instance
from dlhpcstarter.utils import (
    get_test_ckpt_path,
    importer,
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
    subprocess.run(['nvidia-smi'])

    print(args)
    
    print(f"Hostname: {socket.gethostname()}")
    
    args.warm_start_modules = False

    # Set seed number (using the trial number) for deterministic training
    seed_everything(args.trial, workers=True)

    if torch.cuda.is_available():
        print(f'Device capability: {torch.cuda.get_device_capability()}')
        if args.float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(args.float32_matmul_precision)

    # Model definition:
    TaskModel = importer(definition=args.definition, module=args.module)

    # Create symbolic link for the .neptune directory:
    # symlink_dir = os.path.join(args.exp_dir, args.task, '.neptune')
    # if not os.path.isdir(symlink_dir):
    #     Path(symlink_dir).mkdir(parents=True, exist_ok=True)
    # if not os.path.islink('./.neptune'):
    #     os.symlink(symlink_dir, './.neptune', target_is_directory=True)

    # Trainer:
    trainer = trainer_instance(**vars(args))

    # Train:
    if args.train:

        # For deterministic training: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # Resume from checkpoint:
        ckpt_path = resume_from_ckpt_path(args.exp_dir_trial, args.resume_last, args.resume_epoch, args.resume_ckpt_path) if not args.fast_dev_run else None

        # Warm start from checkpoint if not resuming:
        if args.warm_start_ckpt_path and ckpt_path is None and not args.fast_dev_run:
            print('Warm-starting using: {}.'.format(args.warm_start_ckpt_path))

            if args.warm_start_optimiser:
                model = TaskModel(**vars(args))
                model = torch.compile(model) if args.compile else model
                trainer.fit(model, ckpt_path=args.warm_start_ckpt_path)                
            else:                    
                strict = args.warm_start_ckpt_path_strict if args.warm_start_ckpt_path_strict is not None else True
                model = TaskModel.load_from_checkpoint(checkpoint_path=args.warm_start_ckpt_path, strict=strict, **vars(args))
                if args.allow_warm_start_optimiser_partial:
                    assert not strict
                    model.warm_start_optimiser_partial = True
                model = torch.compile(model) if args.compile else model
                trainer.fit(model)

        # # Warm-start from other experiment:
        # elif 'warm_start_exp_dir' in args.__dict__:
        #     if args.warm_start_exp_dir:
                
        #         assert isinstance(args.warm_start_exp_dir, str)

        #         # The experiment trial directory of the other configuration:
        #         warm_start_exp_dir_trial = os.path.join(args.warm_start_exp_dir, f'trial_{args.trial}')

        #         # Get the path to the best performing checkpoint
        #         ckpt_path = get_test_ckpt_path(
        #             warm_start_exp_dir_trial, 
        #             args.warm_start_monitor, 
        #             args.warm_start_monitor_mode, 
        #             args.test_epoch, 
        #             args.test_ckpt_path,
        #         )

        #         model = TaskModel.load_from_checkpoint(checkpoint_path=ckpt_path, **vars(args))
        #         print('Warm-starting using: {}.'.format(ckpt_path))

        # # Warm-start from Hugging Face checkpoint:
        # elif 'warm_start_name' in args.__dict__:
        #     if args.warm_start_name:
        #         model = TaskModel(**vars(args))
        #         hf_ckpt = transformers.AutoModel.from_pretrained(args.warm_start_name, trust_remote_code=True)
        #         model.encoder_decoder.load_state_dict(hf_ckpt.state_dict())
        # elif resume_ckpt_path is not None:
        #     model = TaskModel.load_from_checkpoint(checkpoint_path=ckpt_path, **vars(args))

        # Resume training from ckpt_path:
        elif ckpt_path is not None:
            model = TaskModel(**vars(args))
            model = torch.compile(model) if args.compile else model
            trainer.fit(model, ckpt_path=ckpt_path)

        # Let the module warm start itself if ckpt_path is None:
        else:
            args.warm_start_modules = True
            model = TaskModel(**vars(args))
            model = torch.compile(model) if args.compile else model
            trainer.fit(model)

    # Test:
    if args.test:

        args.warm_start_modules = False
        model = TaskModel(**vars(args))
        
        ckpt_path = None
        if args.test_ckpt_name and not args.test_without_ckpt:
            assert 'model' not in locals(), 'if "test_ckpt_name" is defined in the config, it will overwrite the model checkpoint that has been trained.'
            hf_ckpt = transformers.AutoModel.from_pretrained(args.test_ckpt_name, trust_remote_code=True)
            model.encoder_decoder.load_state_dict(hf_ckpt.state_dict())
        elif not args.fast_dev_run and not args.test_without_ckpt:

            if args.other_exp_dir:

                # The experiment trial directory of the other configuration:
                other_exp_dir_trial = os.path.join(args.other_exp_dir, f'trial_{args.trial}')

                ckpt_path = get_test_ckpt_path(
                    other_exp_dir_trial, args.other_monitor, args.other_monitor_mode, 
                )
            
            else:

                # Get the path to the best performing checkpoint
                ckpt_path = get_test_ckpt_path(
                    args.exp_dir_trial, args.monitor, args.monitor_mode, args.test_epoch, args.test_ckpt_path,
                )

            print('Testing checkpoint: {}.'.format(ckpt_path))
            write_test_ckpt_path(ckpt_path, args.exp_dir_trial)
                    
        trainer.test(model, ckpt_path=ckpt_path)
        