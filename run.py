import logging
import sys
import copy
from argparse import Namespace
from typing import Any, Dict, List, Optional, Union, Callable, Type
import random
import math
import numpy as np
import wandb
import torch
import argparse
from tqdm import tqdm
from lightning import LightningModule, LightningDataModule
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback, ReduceLROnPlateau, LRSchedulerTypeUnion
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from lightning.pytorch.cli import LightningArgumentParser
from torch.optim.optimizer import Optimizer

ArgsType = Optional[Union[List[str], Dict[str, Any], Namespace]]

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.getLogger()

# wandb.require("core") default in wandb 0.18.0
import os
os.environ['USE_PYGEOS'] = '0'
os.system("taskset -c -p 0-95 %d" % os.getpid())
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }

class LoggerSaveConfigCallback(SaveConfigCallback):
    def __init__(self, config: dict, parser: argparse.ArgumentParser):
        super().__init__(config, parser)
        self.overwrite = True

    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger) and stage == "fit":
            config = self.config.get('fit', self.config)
            config = namespace_to_dict(config)
            trainer.logger.log_hyperparams({"config": config})


def update_namespace_from_nested_dict(namespace, nested_dict, prefix="", partial_update:str=None):
    for key, value in nested_dict.items():
        # Create a variable name based on the current prefix
        # keep those training configurations unchanged

        if prefix=="" and (key not in ['seed_everything', 'optimizer', 'lr_scheduler', 'data', 'model']):
            continue
        if partial_update and key != partial_update:
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            # Recursively update for nested dictionaries
            update_namespace_from_nested_dict(namespace, value, prefix=f"{full_key}")
        else:
            # Set the value in the namespace
            namespace[full_key] = value


class MyLightningCLI(LightningCLI):
    def __init__(self, 
        model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]] = None,
        datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,):
        """Receives as input pytorch-lightning classes (or callables which return pytorch-lightning classes), which are
        called / instantiated using a parsed configuration file and / or command line args.

        Parsing of configuration from environment variables can be enabled by setting ``parser_kwargs={"default_env":
        True}``. A full configuration yaml would be parsed from ``PL_CONFIG`` if set. Individual settings are so parsed
        from variables named for example ``PL_TRAINER__MAX_EPOCHS``.

        For more info, read :ref:`the CLI docs <lightning-cli>`.

        Args:
            model_class: An optional :class:`~lightning.pytorch.core.LightningModule` class to train on or a
                callable which returns a :class:`~lightning.pytorch.core.LightningModule` instance when
                called. If ``None``, you can pass a registered model with ``--model=MyModel``.
            datamodule_class: An optional :class:`~lightning.pytorch.core.datamodule.LightningDataModule` class or a
                callable which returns a :class:`~lightning.pytorch.core.datamodule.LightningDataModule` instance when
                called. If ``None``, you can pass a registered datamodule with ``--data=MyDataModule``.
            save_config_callback: A callback class to save the config.
            save_config_kwargs: Parameters that will be used to instantiate the save_config_callback.
            trainer_class: An optional subclass of the :class:`~lightning.pytorch.trainer.trainer.Trainer` class or a
                callable which returns a :class:`~lightning.pytorch.trainer.trainer.Trainer` instance when called.
            trainer_defaults: Set to override Trainer defaults or add persistent callbacks. The callbacks added through
                this argument will not be configurable from a configuration file and will always be present for
                this particular CLI. Alternatively, configurable callbacks can be added as explained in
                :ref:`the CLI docs <lightning-cli>`.
            seed_everything_default: Number for the :func:`~lightning.fabric.utilities.seed.seed_everything`
                seed value. Set to True to automatically choose a seed value.
                Setting it to False will avoid calling ``seed_everything``.
            parser_kwargs: Additional arguments to instantiate each ``LightningArgumentParser``.
            subclass_mode_model: Whether model can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            subclass_mode_data: Whether datamodule can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``. Command line style
                arguments can be given in a ``list``. Alternatively, structured config options can be given in a
                ``dict`` or ``jsonargparse.Namespace``.
            run: Whether subcommands should be added to run a :class:`~lightning.pytorch.trainer.trainer.Trainer`
                method. If set to ``False``, the trainer and model classes will be instantiated only.

        """
        self.save_config_callback = save_config_callback
        self.save_config_kwargs = save_config_kwargs or {}
        self.trainer_class = trainer_class
        self.trainer_defaults = trainer_defaults or {}
        self.seed_everything_default = seed_everything_default
        self.parser_kwargs = parser_kwargs or {}
        self.auto_configure_optimizers = auto_configure_optimizers

        self.model_class = model_class
        # used to differentiate between the original value and the processed value
        self._model_class = model_class or LightningModule
        self.subclass_mode_model = (model_class is None) or subclass_mode_model

        self.datamodule_class = datamodule_class
        # used to differentiate between the original value and the processed value
        self._datamodule_class = datamodule_class or LightningDataModule
        self.subclass_mode_data = (datamodule_class is None) or subclass_mode_data

        main_kwargs, subparser_kwargs = self._setup_parser_kwargs(self.parser_kwargs)
        self.setup_parser(run, main_kwargs, subparser_kwargs)
        self.parse_arguments(self.parser, args)

        self.subcommand = self.config["subcommand"] if run else None

        self._set_seed()

        self._add_instantiators()
        if self.subcommand in ['validate', 'test', 'predict']  and self.config[self.subcommand].correct_bias:
            self.config[self.subcommand]['model']['init_args']['evaluate_high_slope'] = True

        self.before_instantiate_classes()
        self.instantiate_classes()

        if self.subcommand in ['validate', 'test', 'predict']  and self.config[self.subcommand].correct_bias:
            self.bias_correction()
            self.model.correct_bias = True
        else:
            self.model.correct_bias = False
        
        if self.subcommand == 'fit' and self.config[self.subcommand].get('use_pretrained_model'):
            self.use_pretrained_model()

        if self.subcommand is not None:
            self._run_subcommand(self.subcommand)

    def use_pretrained_model(self):
        """Load a pretrained model from a checkpoint."""
        # Load the checkpoint into the existing model instance
        from models.modules.xception_blocks import SepConvBlock
        checkpoint = torch.load(self.config[self.subcommand]['ckpt_path'], weights_only=False, map_location=self.model.device)
        self.config[self.subcommand]['ckpt_path'] = None
        self.config_init[self.subcommand]['ckpt_path'] = None # model loaded from the checkpoint defined here
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.sepconv_blocks = self.model.sepconv_blocks[:3]
        self.model.mid_block = SepConvBlock(self.model.activation_layer, norm_layer=self.model.norm_layer, in_channels=self.model.num_sepconv_filters, out_channels=self.model.num_sepconv_filters)
        block = self.config[self.subcommand].model.init_args.nonlin_block
        self.model.nonlin_blocks = self.model._make_sepconv_blocks(block=block, kernerl_sizes=(1, 1), num_blocks=self.config[self.subcommand].model.init_args.num_nonlin_blocks)

    def bias_correction(self):
        # check if the delta bias is logged         
        if not self.config[self.subcommand].recalculate_bias and self.delta_bias is not None:
            print('correct bias using logged delta bias')
            # Load the checkpoint into the existing model instance
            checkpoint = torch.load(self.config[self.subcommand]['ckpt_path'], weights_only=False, map_location=self.model.device)
            self.config[self.subcommand]['ckpt_path'] = None
            self.config_init[self.subcommand]['ckpt_path'] = None # model loaded from the checkpoint defined here
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()
            print(self.model.last_conv.bias)
            self.model.last_conv.bias.data -= self.delta_bias
            print(self.model.last_conv.bias)
            return
        print('correcting bias using training data')
        model = self.model
        # Load the checkpoint into the existing model instance
        checkpoint = torch.load(self.config[self.subcommand]['ckpt_path'], weights_only=False, map_location=self.model.device)
        self.config[self.subcommand]['ckpt_path'] = None
        self.config_init[self.subcommand]['ckpt_path'] = None # model loaded from the checkpoint defined here
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print(model.last_conv.bias)
        if torch.cuda.is_available():
            model.to('cuda')
        
        epochs = len(self.datamodule.train_fp)
        self.datamodule.order = 'SEQUENTIAL'
        for epoch in range(epochs): #TODO: bring back epochs
            train_dataloader = self.datamodule.train_dataloader()
            # c = 0
            for batch in tqdm(train_dataloader):
                batch = [b.to(model.device) for b in batch]
                with torch.no_grad():
                    model.validation_step(batch, None)
                # c += 1
                # if c > 10:
                #     break

        errors_gradual_slope = model.val_metrics['ME/'].sum_per_error / model.val_metrics['ME/'].total
        errors_with_steep_slope =model.val_me_with_steep_slope.sum_per_error / model.val_me_with_steep_slope.total
        errors_gradual_slope_veg = model.val_metrics_veg['ME/'].sum_per_error / model.val_metrics_veg['ME/'].total

        feature_size = errors_gradual_slope.shape[0]
        device = model.last_conv.bias.device
        errors = []
        for err in [errors_gradual_slope, errors_with_steep_slope, errors_gradual_slope_veg]:
            # correct the bias for median prediction only
            err = torch.cat([torch.zeros((feature_size,1), device=device), err.unsqueeze(1), torch.zeros((feature_size,1), device=device)], dim=1)
            err = err.flatten()
            if self.model.out_channels > err.shape[0]*3:
                err = torch.cat([err, torch.zeros(12, device=device)])
            errors.append(err)
        errors = torch.stack(errors)
        errors = errors.T
        data = wandb.Table(data=errors.tolist(), columns=['me_gradual_slope', 'me_with_steep_slope', 'me_gradual_slope_veg'])
        wandb.log({"delta_biases": data})

        # last_conv.bias.data -= errors.squeeze()
        self.delta_bias = data.get_dataframe()[self.config[self.subcommand].bias_correction_column].values
        
        self.model.last_conv.bias.data -= torch.tensor(self.delta_bias).to(self.model.device)
        model.val_metrics.reset()
        model.val_metrics_veg.reset()
        print(model.last_conv.bias)
            
    
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--lr_step_interval", default="step")
        parser.add_argument("--correct_bias",  type=bool, default=True)
        parser.add_argument("--use_pretrained_model",  type=bool, default=False)
        parser.add_argument("--bias_correction_column", default='me_gradual_slope_veg')
        parser.add_argument("--recalculate_bias",  type=bool, default=False)

    
    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        """Parses command line arguments and stores it in ``self.config``."""
        if args is not None and len(sys.argv) > 1:
            rank_zero_warn(
                "LightningCLI's args parameter is intended to run from within Python like if it were from the command "
                "line. To prevent mistakes it is not recommended to provide both args and command line arguments, got: "
                f"sys.argv[1:]={sys.argv[1:]}, args={args}."
            )
        if isinstance(args, (dict, Namespace)):
            self.config = parser.parse_object(args)
        else:
            self.config = parser.parse_args(args)
        
        subcommand = self.config.get('subcommand')
        if subcommand is None:
            run_id = self.config.trainer.logger.init_args.id
            config = self.config
        else:
            config = self.config[subcommand]
            run_id = self.config[subcommand].trainer.logger.init_args.id

        # calculate the number of training steps for cosine scheduler
        if subcommand == "fit" and config.lr_scheduler:
            if config.lr_scheduler.init_args.get('num_warmup_steps') and config.lr_scheduler.init_args.get('num_training_steps') is None:
                steps_per_epoch_per_device = math.floor(5720188 // config.data.init_args.batch_size // torch.cuda.device_count())# 11440128 is the number of samples in the training dataset, TODO: get this from the datamodule
                num_training_steps = steps_per_epoch_per_device * config.trainer.max_epochs
                self.config[subcommand].lr_scheduler.init_args.num_training_steps = num_training_steps
                print(f"num_training_steps: {num_training_steps}")


        if run_id is not None:
            import wandb
            model_alias = 'best'
            api = wandb.Api()
            wandb_project = config.trainer.logger.init_args.project
            run_path = f"{wandb_project}/{run_id}"
            run_ = api.run(run_path)
            
            cfg = config.trainer.logger.init_args
            config.trainer.logger.init_args.resume = "must"
            config.trainer.logger.init_args.name = run_.name
            if subcommand == "fit":
                train_fp = copy.copy(config.data.init_args.train_fp)
                val_fp = copy.copy(config.data.init_args.val_fp)
            elif subcommand == "test":
                test_fp = copy.copy(config.data.init_args.test_fp)
            if subcommand == "fit": #NOTE: resume/continue running
                model_alias = 'latest'
                optim_config = copy.copy(config['optimizer'])
                lr_scheduler_config = copy.copy(config['lr_scheduler'])
                lr_scheduler_config['init_args']['last_epoch'] = run_.config['config']['trainer']['max_epochs']-1 # start from 0
                more_epochs = config.trainer.max_epochs
                update_namespace_from_nested_dict(config, run_.config['config'])
                # update everything from logged config except following
                config['optimizer'] = optim_config
                config['lr_scheduler'] = lr_scheduler_config
                config.trainer.max_epochs = more_epochs + run_.config['config']['trainer']['max_epochs']
                config['data']['init_args']['train_fp'] = train_fp
                config['data']['init_args']['val_fp'] = val_fp
            else:
                # evaluation, use logged config to initialize the model to be able to load the model correctly
                try:
                    logged_config = run_.config['config']
                except:
                    logged_config = {}
                    logged_config['model'] = run_.config
                    logged_config['model']['class_path'] = run_.config['_class_path']
                update_namespace_from_nested_dict(config, logged_config, partial_update='model')
                # config['model']['init_args']['evaluate_high_slope'] = True # not needed for the grouped boxplot

            # check if training/validation data exists
            if subcommand == "fit" and (not os.path.exists(train_fp)):
                self.config[subcommand].data.init_args.train_fp = train_fp
            elif subcommand =='test':
                self.config[subcommand].data.init_args.test_fp = test_fp

            artifacts = run_.logged_artifacts()
            artifacts = [artifact for artifact in artifacts if artifact.type == 'model']
            if isinstance(model_alias, str):
                artifact = [art for art in artifacts if model_alias in art.aliases][0]
            else:
                artifact = artifacts[model_alias]
            # artifact = run_.use_artifact(f'model-{run_id}:best', type='model')
            ckpt_path = artifact.file()
            self.config[subcommand].ckpt_path = ckpt_path

            # check if bias correction has been done and delta bias has been logged
            if subcommand in ['validate', 'test', 'predict']  and self.config[subcommand].correct_bias:
                try:
                    table = api.artifact(f'{wandb_project}/run-{run_id}-delta_biases:latest')
                    df = table.get('delta_biases').get_dataframe()
                    self.delta_bias = df[self.config[subcommand].bias_correction_column].values.astype('float32')
                except:
                    print('delta_bias not logged yet')
                    self.delta_bias = None
            else:    
                self.delta_bias = None

    # override from
    # https://github.com/Lightning-AI/pytorch-lightning/blob/f3f10d460338ca8b2901d5cd43456992131767ec/src/lightning/pytorch/cli.py#L605-L606
    def configure_optimizers(
        self,
        lightning_module: LightningModule,
        optimizer: Optimizer,
        lr_scheduler: Optional[LRSchedulerTypeUnion] = None,
    ) -> Any:
        if lr_scheduler is None:
            return optimizer
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": lr_scheduler.monitor,
                    "interval": self.config.fit.lr_step_interval,
                },
            }
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.config.fit.lr_step_interval,
            },
        }
    
    # def add_arguments_to_parser(self, parser) -> None:
    #     import ipdb; ipdb.set_trace()
    #     parser.link_arguments('model.init_args.in_channels', 'model.init_args.encoder.init_args.in_channels')
    #     parser.link_arguments('model.init_args.activation_layer', 'model.init_args.encoder.init_args.activation_layer')
    

    # def after_fit(self):
    #     send_email(address=self.config["notification_email"], message="trainer.fit finished")


def cli_main():
    cli = MyLightningCLI(model_class=LightningModule, 
                         datamodule_class=LightningDataModule, 
                         subclass_mode_model=True,
                         subclass_mode_data=True, 
                         save_config_callback=LoggerSaveConfigCallback,
                        #  parser_kwargs={"parser_mode": "omegaconf"},
                         ) # omegaconf: allow variable interpolation

    # cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # test on best model
    # cli.trainer.test(cli.model, datamodule=cli.datamodule)

    
if __name__ == '__main__':

    # from ipdb import launch_ipdb_on_exception
    # with launch_ipdb_on_exception():
    cli_main()
        # evaluate_with_dask()