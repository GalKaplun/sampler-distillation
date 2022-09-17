import hydra
import wandb
import logging
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.utils import log_hyperparams

log = logging.getLogger(__name__)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def train(config: DictConfig):
    log.info(f"Instantiating logger <{config.logger._target_}>")
    logger: WandbLogger = hydra.utils.instantiate(config.logger)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    run_id = config.run_id
    save_keyword = 'noisy' if config.datamodule.add_noise else 'clean'
    if config.datamodule.dataset.lower() == 'cifar5m':
        val_check_interval = 400
        lr_log_interval = 'step'
        if config.model.teach_arch.lower() != 'non':
            save_keyword = f'-{config.model.max_teachers}_{config.model.teach_arch.lower()}_teachers'
        else:
            save_keyword = 'from_labels'
    else:
        lr_log_interval = 'epoch'
        val_check_interval = 1.
    save_path = f'./models/{config.datamodule.dataset.lower()}/{save_keyword}/{config.model.arch}-{run_id}/'

    checkpoint_callback = ModelCheckpoint(dirpath=save_path,
                                          filename=f"{save_keyword}-{config.model.arch}-{{step}}",
                                          monitor=None,
                                          save_top_k=-1,
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval=lr_log_interval)

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=f'{save_keyword}/checkpoints{run_id}',
        val_check_interval=val_check_interval
    )
    
    config.model.learning_rate = config.model.learning_rate * (config.datamodule.batch_size / 256)

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    log.info("Logging hyperparameters!")
    log_hyperparams(config=config, trainer=trainer)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    wandb.finish()
