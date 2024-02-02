import hashlib
import inspect
import logging
import signal
import time
from datetime import timedelta
from typing import Optional

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (Callback, EarlyStopping,
                                         LearningRateMonitor, ModelCheckpoint)
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies import DeepSpeedStrategy

logging.getLogger(
    "neptune.new.internal.operation_processors.async_operation_processor",
).setLevel(logging.CRITICAL)


def trainer_instance(
    task: str,
    config_name: str,
    trial: int,
    monitor: Optional[str] = None,
    monitor_mode: str = 'min',
    early_stopping: bool = False,
    patience: int = 0,
    min_delta: float = 0.0,
    divergence_threshold: Optional[float] = None,
    exp_dir_trial: Optional[str] = None,
    sched_inter: Optional[str] = None,  # 'step', 'epoch', or None.
    save_top_k: int = 1,
    every_n_epochs: Optional[int] = 1,
    every_n_train_steps: Optional[int] = None,
    neptune_api_key: Optional[str] = None,
    neptune_username: Optional[str] = None,
    neptune_mode: Optional[str] = 'async',
    num_nodes: int = 1,
    devices: Optional[int] = 1,
    submit: bool = False,
    mbatch_size: Optional[int] = None,
    accumulated_mbatch_size: Optional[int] = None,
    deterministic: bool = True,
    num_sanity_val_steps: int = 0,
    loggers: Optional[list] = None,
    callbacks: Optional[list] = None,  # [RichProgressBar()]
    plugins: Optional[list] = None,
    enable_progress_bar: Optional[bool] = None,
    one_epoch_only: bool = False,
    learning_rate_monitor: bool = False,
    **kwargs,
) -> Trainer:
    """
    Creates an instance of lightning.pytorch.Trainer using key callbacks and loggers. Also changes some
    defaults for the init of lightning.pytorch.Trainer.

    Parameters for lightning.pytorch.Trainer are described here:
        https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
    These will be captured by kwargs and passed to lightning.pytorch.Trainer

    Argument/s:
        task - name of the task.
        config_name - name of the configuration.
        trial - trial identifier.
        monitor - metric to monitor for EarlyStopping and ModelCheckpoint.
        monitor_mode - whether the metric to be monitored is to be maximised or
            minimised.
        early_stopping - stop training when a monitored metric has stopped
            improving.
        patience - no. of epochs with no improvement after which training will
            be stopped.
        min_delta - minimum change in the monitored quantity to qualify as an
            improvement.
        divergence_threshold - stop training as soon as the monitored quantity becomes worse than this threshold.
        exp_dir_trial - experiment directory for the trial. All outputs are saved to this path.
        sched_inter - learning rate scheduler interval ('step' or 'epoch').
        save_top_k - best k models saved according to the monitored metric. If
            0, no models are saved. If -1, all models are saved.
        every_n_epochs - save model every n epochs.
        every_n_epochs - save model every n training steps.
        neptune_api_key - API key, found on neptune.ai, for NeptuneLogger.
        neptune_username - Username for on neptune.ai, for NeptuneLogger.
        neptune_mode - https://docs.neptune.ai/api/connection_modes/.
        num_nodes - number of nodes for the job.
        devices - number of devices per node.
        submit - submit to cluster manager.
        mbatch_size - mini-batch size of dataloaders.
        accumulated_mbatch_size - desired accumulated mini-batch size.
        deterministic - ensures that the training is deterministic.
        num_sanity_val_steps - runs n validation batches before starting the training routine.
        loggers - loggers for Trainer.
        callbacks - callbacks for Trainer.
        plugins - plugins for Trainer.
        kwargs - keyword arguments for Trainer.
        enable_progress_bar - show the progress bar (will be turned off for submissions).
        one_epoch_only - perform only one epoch of training.
        learning_rate_monitor - add the LearningRateMonitor as a callback.
    """
    accumulate_grad_batches = None
    loggers = [] if loggers is None else loggers
    callbacks = [] if callbacks is None else callbacks
    plugins = [] if plugins is None else plugins

    if submit:
        enable_progress_bar = False if enable_progress_bar is None else enable_progress_bar

    # Unsure if Lightning's SLURMEnvironment features fault-tolerant training:
    # if submit:

    #     # See: https://github.com/Lightning-AI/lightning/issues/6389#issuecomment-1377759948
    #     class DisabledSLURMEnvironment(SLURMEnvironment):
    #         def detect() -> bool:
    #             return False

    #         @staticmethod
    #         def _validate_srun_used() -> None:
    #             return

    #         @staticmethod
    #         def _validate_srun_variables() -> None:
    #             return
            
    #     plugins.append(DisabledSLURMEnvironment(auto_requeue=False))

    # Deepspeed has its own autocast capabilities:
    # if 'strategy' in kwargs and 'precision' in kwargs:
    #     if 'deepspeed' in kwargs['strategy'] and kwargs['precision'] == 16:
    #         raise ValueError('DeepSpeed and "precision=16" are incompatible as DeepSpeed has its own autocast functionality.')

    # DeepSpeed config:
    if 'deepspeed_config' in kwargs:
        kwargs['strategy'] = DeepSpeedStrategy(
            **kwargs['deepspeed_config']
        )

    # Loggers
    loggers.append(CSVLogger(exp_dir_trial))

    # Remove 'lightning_logs' structure for tensorboard to allow different sessions to be grouped:
    loggers.append(
        TensorBoardLogger(exp_dir_trial, default_hp_metric=False, version='', name='tensorboard')
    )  
    if neptune_api_key is not None:
        name = f'{config_name}_t_{trial}'
        custom_run_id = str(
            int.from_bytes(
                hashlib.sha256(name.encode(encoding="ascii",errors="ignore")).digest()[:12], 'little'
            )
        ) 
        assert len(custom_run_id) <= 32, '"custom_run_id" must be less than or equal to 32 characters'
        assert neptune_username is not None, 'You must specify your neptune.ai username.'
        loggers.append(
            NeptuneLogger(
                api_key=neptune_api_key,
                project=f'{neptune_username}/{task.replace("_", "-")}',
                name=name,
                tags=[config_name, f'trial_{trial}'],
                prefix='log',
                custom_run_id=custom_run_id,
                capture_stdout=False,
                capture_stderr=False,
                capture_hardware_metrics=True,
                capture_traceback=False,
                log_model_checkpoints=False,
                source_files=[],
                flush_period=60,
                mode=neptune_mode,
            )
        )

    # Model checkpointing
    assert (every_n_epochs is not None) or (every_n_train_steps is not None), 'Neither "every_n_epochs" or ' \
        '"every_n_train_steps" is set. No checkpoints will be saved.'
    assert save_top_k != 0, '"save_top_k" is 0, therefore, no checkpoints will be saved.'

    if every_n_epochs:
        save_top_k = save_top_k if monitor is not None else -1
        callbacks.append(
            ModelCheckpoint(
                dirpath=exp_dir_trial,
                monitor=monitor,
                mode=monitor_mode,
                save_top_k=save_top_k,
                every_n_epochs=every_n_epochs,
                filename='{epoch:d}-{step:d}-{' + monitor + ':f}' if monitor else '{epoch:d}-{step:d}',
                save_last=True,
                enable_version_counter=False,
            )
        )
        # if 'strategy' in kwargs:
        #     if isinstance(kwargs['strategy'], str):
        #         if 'deepspeed_stage_3' in kwargs['strategy']: 
        #             callbacks[-1].FILE_EXTENSION = ""
        #     elif isinstance(kwargs['strategy'], DeepSpeedStrategy):
        #         if kwargs['deepspeed_config']['stage'] == 3:
        #             callbacks[-1].FILE_EXTENSION = ""

    # if every_n_train_steps:
    #     if resumable:
    #         raise ValueError('Cannot resume training from a checkpoint that ended before the epoch ended. Fault '
    #                          'tolerant training needs to be implemented for this: '
    #                          'https://pytorch-lightning.readthedocs.io/en/latest/clouds'
    #                          '/fault_tolerant_training_expert.html#enable-fault-tolerant-behavior-on-your-own-cluster')
    #     callbacks.append(
    #         ModelCheckpoint(
    #             dirpath=exp_dir_trial,
    #             monitor=monitor,
    #             mode=monitor_mode,
    #             save_top_k=save_top_k,
    #             every_n_train_steps=every_n_train_steps,
    #             filename='{step:d}-{' + monitor + ':f}',
    #             save_last=False,  # cannot resume from this checkpoint.
    #         )
    #     )

    # Perform only one epoch of training:
    if one_epoch_only:
        class OneEpochOnlyCallback(Callback):
            def __init__(self, neptune_api_key=None):
                self.start_time = time.time()
                self.neptune_api_key=neptune_api_key
            def on_validation_epoch_end(self, trainer, pl_module):
                trainer.should_stop = True
                pl_module.trainer.should_stop = True
            # def on_train_end(self, trainer, pl_module):
            #     elapsed_time = (time.time() - self.start_time) / 3600
            #     print(f'Training epoch elapsed time (hours): {elapsed_time}')
            #     pl_module.log('elapsed_time_hours', elapsed_time / 3600, on_step=True, on_epoch=False)
            def teardown(self, trainer, pl_module, stage):
                if stage == 'fit':
                    signal.alarm(1) 
        callbacks.append(OneEpochOnlyCallback(neptune_api_key=neptune_api_key))

    # Early stopping
    if early_stopping:
        if 'strategy' in kwargs:
            assert 'deepspeed' not in kwargs['strategy'], 'DeepSpeed does not work with early stopping currently.'
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=monitor_mode,
                min_delta=min_delta,
                patience=patience,
                divergence_threshold=divergence_threshold,
                verbose=True,
            )
        )

    # Learning rate monitor:
    if learning_rate_monitor:
        callbacks.append(LearningRateMonitor(log_momentum=True, log_weight_decay=True))

    # Accumulate gradient batches
    if accumulated_mbatch_size:
        total_devices = devices * num_nodes if devices * num_nodes > 0 else 1
        accumulate_grad_batches = accumulated_mbatch_size / (mbatch_size * total_devices)
        assert accumulate_grad_batches.is_integer(), f'Desired accumulated_mbatch_size ({accumulated_mbatch_size}) ' \
                                                     f'can not be attained with mbatch_size={mbatch_size}, devices=' \
                                                     f'{devices}, and num_nodes={num_nodes}'
        accumulate_grad_batches = int(accumulate_grad_batches)
    else:
        accumulate_grad_batches = 1

    # Remove keyword arguments not associated with lightning.pytorch.Trainer.
    # Parameters associated with lightning.pytorch.Trainer:
    #   https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
    kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(Trainer).parameters}

    # lightning.pytorch.Trainer
    print(f'PTL no. devices: {devices}.')
    print(f'PTL no. nodes: {num_nodes}.')
    return Trainer(
        default_root_dir=exp_dir_trial, # Needed for hpc_ckpt save path. 
        logger=loggers,
        callbacks=callbacks,
        plugins=plugins,
        devices=devices,
        num_nodes=num_nodes,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=deterministic,
        num_sanity_val_steps=num_sanity_val_steps,
        enable_progress_bar=enable_progress_bar,
        **kwargs,
    )
