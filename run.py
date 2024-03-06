import logging
import wandb
import argparse
from lightning.pytorch.cli import LightningCLI
import lightning.pytorch as pl

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.getLogger()

import os
os.environ['USE_PYGEOS'] = '0'

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }

class MyLightningCLI(LightningCLI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_arguments_to_parser(self, parser) -> None:
        # parser.add_argument("--notification_email", default="huzh@di.ku.dk")
        parser.link_arguments('model.init_args.in_channels', 'model.init_args.backbone_model.init_args.in_channels')
        parser.link_arguments('model.init_args.activation_layer', 'model.init_args.backbone_model.init_args.activation_layer')
    
    def instantiate_classes(self):
        super().instantiate_classes()
        # UPDATE wandb config
        # if subcommand fit is used,
        config = self.config.get('fit', self.config)
        config = namespace_to_dict(config)
        wandb.config.update(config)

    # def after_fit(self):
    #     send_email(address=self.config["notification_email"], message="trainer.fit finished")


def cli_main():
    cli = MyLightningCLI(model_class=pl.LightningModule, datamodule_class=pl.LightningDataModule, subclass_mode_model=True,subclass_mode_data=True, save_config_callback=None)
    # cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # test on best model
    # cli.trainer.test(cli.model, datamodule=cli.datamodule)
    


if __name__ == '__main__':

    # from ipdb import launch_ipdb_on_exception
    # with launch_ipdb_on_exception():
        cli_main()