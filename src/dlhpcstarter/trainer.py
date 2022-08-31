from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from src.dlhpcstarter.tools.mods.logger import CSVLogger
from src.dlhpcstarter.utils import resume_from_ckpt_path
from typing import Optional
import inspect
import logging
logging.getLogger(
    "neptune.new.internal.operation_processors.async_operation_processor",
).setLevel(logging.CRITICAL)


def trainer_instance(
    monitor: str,
    monitor_mode: str,
    task: str,
    config: str,
    trial: int,
    early_stopping: bool = False,
    patience: int = 0,
    min_delta: float = 0.0,
    divergence_threshold: Optional[float] = None,
    exp_dir_trial: Optional[str] = None,
    resumable: bool = True,
    resume_epoch: Optional[int] = None,
    resume_ckpt_path: Optional[str] = None,
    sched_inter: Optional[str] = None,  # 'step', 'epoch', or None.
    save_top_k: int = 1,
    every_n_epochs: Optional[int] = 1,
    every_n_train_steps: Optional[int] = None,
    debug: bool = False,
    submit: bool = False,
    neptune_api_key: Optional[str] = None,
    neptune_username: Optional[str] = None,
    num_nodes: int = 1,
    num_gpus: Optional[int] = None,
    mbatch_size: Optional[int] = None,
    accumulated_mbatch_size: Optional[int] = None,
    deterministic: bool = True,
    num_sanity_val_steps: int = 0,
    loggers: Optional[list] = None,
    callbacks: Optional[list] = None,  # [RichProgressBar()]
    plugins: Optional[list] = None,
    **kwargs,
) -> Trainer:
    """
    Creates an instance of pytorch_lightning.Trainer using key callbacks and loggers. Also changes some
    defaults for the init of pytorch_lightning.Trainer.

    Parameters for pytorch_lightning.Trainer are described here:
        https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
    These will be captured by kwargs and passed to pytorch_lightning.Trainer

    Argument/s:
        monitor - metric to monitor for EarlyStopping and ModelCheckpoint.
        monitor_mode - whether the metric to be monitored is to be maximised or
            minimised.
        task - name of the task.
        config - name of the configuration.
        trial - trial identifier.
        early_stopping - stop training when a monitored metric has stopped
            improving.
        patience - no. of epochs with no improvement after which training will
            be stopped.
        min_delta - minimum change in the monitored quantity to qualify as an
            improvement.
        divergence_threshold - stop training as soon as the monitored quantity becomes worse than this threshold.
        exp_dir_trial - experiment directory for the trial. All outputs are saved to this path.
        resumable - whether the last completed epoch is saved to enable resumable training.
        resume_epoch - the epoch to resume training from.
        resume_ckpt_path - resume training from the specified checkpoint.
        sched_inter - learning rate scheduler interval ('step' or 'epoch').
        save_top_k - best k models saved according to the monitored metric. If
            0, no models are saved. If -1, all models are saved.
        every_n_epochs - save model every n epochs.
        every_n_epochs - save model every n training steps.
        debug - training, validation, and testing are completed using one mini-batch.
        submit - submit to cluster manager.
        neptune_api_key - API key, found on neptune.ai, for NeptuneLogger.
        neptune_username - Username for on neptune.ai, for NeptuneLogger.
        num_nodes - number of nodes for the job.
        num_gpus - number of GPUs per node.
        mbatch_size - mini-batch size of dataloaders.
        accumulated_mbatch_size - desired accumulated mini-batch size.
        deterministic - ensures that the training is deterministic.
        num_sanity_val_steps - runs n validation batches before starting the training routine.
        loggers - loggers for Trainer.
        callbacks - callbacks for Trainer.
        plugins - plugins for Trainer.
        kwargs - keyword arguments for Trainer.
    """
    accumulate_grad_batches = None
    loggers = [] if loggers is None else loggers
    """
    Potential default callbacks to add:
        - SLURMEnvironment(auto_requeue=True)  # This could be used in place of the auto requeue in transmodal.cluster
    """
    callbacks = [] if callbacks is None else callbacks
    plugins = [] if plugins is None else plugins
    if submit:
        plugins.append(SLURMEnvironment(auto_requeue=False))

    # Loggers
    loggers.append(CSVLogger(exp_dir_trial, name='', version=''))
    loggers.append(TensorBoardLogger(exp_dir_trial, name='', version='', default_hp_metric=False))
    if neptune_api_key is not None:
        custom_run_id = f'{config[:16]}_trial_{trial}'
        assert len(custom_run_id) <= 32, '"custom_run_id" must be less than or equal to 32 characters'
        assert neptune_username is not None, 'You must specify your neptune.ai username.'
        loggers.append(
            NeptuneLogger(
                api_key=neptune_api_key,
                project=f'{neptune_username}/{task.replace("_", "-")}',
                prefix='log',
                custom_run_id=custom_run_id,
                capture_stdout=False,
                capture_stderr=False,
                capture_hardware_metrics=True,
                capture_traceback=False,
                log_model_checkpoints=False,
                source_files=[],
                flush_period=30,
            )
        )

    # Model checkpointing
    assert (every_n_epochs is not None) or (every_n_train_steps is not None), 'Neither "every_n_epochs" or ' \
        '"every_n_train_steps" is set. No checkpoints will be saved.'
    assert save_top_k != 0, '"save_top_k" is 0, therefore, no checkpoints will be saved.'

    if every_n_epochs:
        callbacks.append(
            ModelCheckpoint(
                dirpath=exp_dir_trial,
                monitor=monitor,
                mode=monitor_mode,
                save_top_k=save_top_k,
                every_n_epochs=every_n_epochs,
                filename='{epoch:d}-{' + monitor + ':f}',
                save_last=True,
            )
        )

    if every_n_train_steps:
        if resumable:
            raise ValueError('Cannot resume training from a checkpoint that ended before the epoch ended. Fault '
                             'tolerant training needs to be implemented for this: '
                             'https://pytorch-lightning.readthedocs.io/en/latest/clouds'
                             '/fault_tolerant_training_expert.html#enable-fault-tolerant-behavior-on-your-own-cluster')
        callbacks.append(
            ModelCheckpoint(
                dirpath=exp_dir_trial,
                monitor=monitor,
                mode=monitor_mode,
                save_top_k=save_top_k,
                every_n_train_steps=every_n_train_steps,
                filename='{step:d}-{' + monitor + ':f}',
                save_last=False,  # cannot resume from this checkpoint.
            )
        )

    # Early stopping
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=monitor_mode,
                min_delta=min_delta,
                patience=patience,
                divergence_threshold=divergence_threshold,
                verbose=False,
            )
        )

    # Learning rate monitor
    if sched_inter is not None:
        callbacks.append(LearningRateMonitor(logging_interval=sched_inter))

    # Accumulate gradient batches
    if accumulated_mbatch_size:
        devices = num_gpus * num_nodes if num_gpus * num_nodes > 0 else 1
        accumulate_grad_batches = accumulated_mbatch_size / (mbatch_size * devices)
        assert accumulate_grad_batches.is_integer(), f'Desired accumulated_mbatch_size ({accumulated_mbatch_size}) ' \
                                                     f'can not be attained with mbatch_size={mbatch_size}, num_gpus=' \
                                                     f'{num_gpus}, and num_nodes={num_nodes}'
        accumulate_grad_batches = int(accumulate_grad_batches)

    # Resume from checkpoint
    ckpt_path = resume_from_ckpt_path(exp_dir_trial, resumable, resume_epoch, resume_ckpt_path)

    # Remove keyword arguments not associated with pytorch_lightning.Trainer.
    # Parameters associated with pytorch_lightning.Trainer:
    #   https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
    kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(Trainer).parameters}

    # pytorch_lightning.Trainer
    return Trainer(
        logger=loggers,
        callbacks=callbacks,
        plugins=plugins,
        fast_dev_run=debug,
        gpus=num_gpus,
        num_nodes=num_nodes,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=deterministic,
        num_sanity_val_steps=num_sanity_val_steps,
        resume_from_checkpoint=ckpt_path,
        **kwargs,
    )
