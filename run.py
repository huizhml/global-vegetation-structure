import logging
import sys
import wandb
import torch
import argparse
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import Logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.getLogger()

import os
os.environ['USE_PYGEOS'] = '0'
os.system("taskset -c -p 0-95 %d" % os.getpid())
os.environ['NUMEXPR_MAX_THREADS'] = '64'

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }

class LoggerSaveConfigCallback(SaveConfigCallback):
    def __init__(self, config: dict, parser: argparse.ArgumentParser):
        super().__init__(config, parser)
        self.overwrite = True

    def save_config(self, trainer: Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.config.get('fit', self.config)
            config = namespace_to_dict(config)
            trainer.logger.log_hyperparams({"config": config})

class MyLightningCLI(LightningCLI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def add_arguments_to_parser(self, parser) -> None:
    #     import ipdb; ipdb.set_trace()
    #     parser.link_arguments('model.init_args.in_channels', 'model.init_args.encoder.init_args.in_channels')
    #     parser.link_arguments('model.init_args.activation_layer', 'model.init_args.encoder.init_args.activation_layer')
    

    # def after_fit(self):
    #     send_email(address=self.config["notification_email"], message="trainer.fit finished")


def cli_main():
    cli = MyLightningCLI(model_class=pl.LightningModule, 
                         datamodule_class=pl.LightningDataModule, 
                         subclass_mode_model=True,
                         subclass_mode_data=True, 
                         save_config_callback=LoggerSaveConfigCallback,
                        #  parser_kwargs={"parser_mode": "omegaconf"},
                         ) # omegaconf: allow variable interpolation
    cli.model = torch.compile(cli.model)
    # cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # test on best model
    # cli.trainer.test(cli.model, datamodule=cli.datamodule)
    


if __name__ == '__main__':

    # from ipdb import launch_ipdb_on_exception
    # with launch_ipdb_on_exception():
        cli_main()