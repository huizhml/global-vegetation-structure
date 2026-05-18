import os
import logging
import sys
import copy
from pathlib import Path
from argparse import Namespace
from typing import Any, Dict, List, Optional, Union, Callable, Type
import random
import math
import numpy as np
import wandb
import torch
import argparse
import atexit
from tqdm import tqdm
from osgeo import gdal
from lightning import LightningModule, LightningDataModule
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback, ReduceLROnPlateau, LRSchedulerTypeUnion
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from lightning.pytorch.cli import LightningArgumentParser
from torch.optim.optimizer import Optimizer
from download.core.utils import create_shared_array


ArgsType = Optional[Union[List[str], Dict[str, Any], Namespace]]

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.getLogger()

# wandb.require("core") default in wandb 0.18.0
# os.environ['USE_PYGEOS'] = '0'
os.system("taskset -c -p 0-95 %d" % os.getpid())
os.environ['NUMEXPR_MAX_THREADS'] = '64'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


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


def update_namespace_from_nested_dict(namespace, nested_dict, prefix="", partial_update: str = None):
    for key, value in nested_dict.items():
        # Create a variable name based on the current prefix
        # keep those training configurations unchanged

        if prefix == "" and (key not in ['seed_everything', 'optimizer', 'lr_scheduler', 'data', 'model']):
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


def verify_naturalness_stats(mean_std_fp, data_file):
    """Fail fast if the naturalness normalisation stats are missing or were
    computed from a different data_file than the one being trained on.

    Read-only and cheap: loads only the small npz's provenance, never scans
    the data and never writes -- so no DDP write race and no silent overwrite.
    step6_calculate_naturalness_stats stays the single, explicit generator.
    """
    step6_cmd = (
        'python -m preprocessing.pipeline.step6_calculate_naturalness_stats '
        f'data_file={data_file} out_fp={mean_std_fp} force=true')
    if mean_std_fp is None:
        raise ValueError('naturalness model.init_args.mean_std_fp is not set')
    msp = Path(mean_std_fp).expanduser()
    if not msp.exists():
        raise FileNotFoundError(
            f'naturalness stats not found: {msp}\n'
            f'Generate them first:\n  {step6_cmd}')
    try:
        with np.load(msp) as _z:
            src = _z['provenance_source'].item() if 'provenance_source' in _z.files else None
    except Exception as e:
        raise ValueError(
            f'could not read {msp}: {e}\nRegenerate it:\n  {step6_cmd}')
    expected = str(Path(data_file).expanduser()) if data_file is not None else None
    if src != expected:
        raise ValueError(
            f'naturalness stats are stale/unverifiable: {msp}\n'
            f'  provenance_source = {src!r}\n'
            f'  data_file         = {expected!r}\n'
            f'Regenerate so normalisation matches the data:\n  {step6_cmd}')
    print(f'[naturalness] stats OK: {msp} (source matches data_file)')


class MyLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--lr_step_interval", default="step")
        parser.add_argument("--correct_bias",  type=bool, default=True)
        parser.add_argument("--quantize_model",  type=bool, default=False)
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
        self.old_id = run_id
        # calculate the number of training steps for cosine scheduler
        if subcommand == "fit" and config.lr_scheduler:
            if config.lr_scheduler.init_args.get('num_warmup_steps') and config.lr_scheduler.init_args.get(
                    'num_training_steps') is None:
                # 11440128 is the number of samples in the training dataset, TODO: get this from the datamodule
                steps_per_epoch_per_device = math.floor(
                    5720188 // config.data.init_args.batch_size // torch.cuda.device_count())
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
            
            
            if subcommand == "fit":
                train_fp = copy.copy(config.data.init_args.train_fp)
                val_fp = copy.copy(config.data.init_args.val_fp)
                cal_fp = copy.copy(config.data.init_args.cal_fp)
            elif subcommand == "test":
                # NaturalnessDataModule has no test_fp (uses data_file); .get
                # keeps the canopy-height path working while not crashing here.
                test_fp = copy.copy(config.data.init_args.get('test_fp'))
            # # NOTE: resume run if training or validating or testing
            if not config.trainer.logger.init_args.resume:
                # Create a new run,
                config.trainer.logger.init_args.id = None
                # evaluation, use logged config to initialize the model to be able to load the model correctly
                try:
                    logged_config = run_.config['config']
                except:
                    logged_config = {}
                    logged_config['model'] = run_.config
                    logged_config['model']['class_path'] = run_.config['_class_path']
                # update_namespace_from_nested_dict(config, logged_config, partial_update='model')
                # config['model']['init_args']['evaluate_high_slope'] = True # not needed for the grouped boxplot
            else:
                # Resume the run, dont change the run name
                config.trainer.logger.init_args.name = run_.name
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
                config['data']['init_args']['cal_fp'] = cal_fp

            
            # if config.trainer.logger.init_args.resume is None:
            #     if subcommand in ['validate', 'test'] or (subcommand == 'fit' and not config.quantize_model):
            #         config.trainer.logger.init_args.resume = 'must'
            #     else:
            #         config.trainer.logger.init_args.resume = False
            #         config.trainer.logger.init_args.id = None
            #         config.trainer.logger.init_args.log_model = False
            # if subcommand == "predict":  # NOTE: prediction will be parallelized, we don't want to log multiple predictions to the same run
            #     config.trainer.logger.init_args.resume = False
            #     config.trainer.logger.init_args.id = None
            #     config.trainer.logger.init_args.log_model = False
            # else:
            #     config.trainer.logger.init_args.resume = "must"


            # check if training/validation data exists
            if subcommand == "fit" and (not os.path.exists(train_fp)):
                self.config[subcommand].data.init_args.train_fp = train_fp
            elif subcommand == 'test' and test_fp is not None:
                self.config[subcommand].data.init_args.test_fp = test_fp

            artifacts = run_.logged_artifacts()
            artifacts = [artifact for artifact in artifacts if artifact.type == 'model']
            if isinstance(model_alias, str):
                artifact = [art for art in artifacts if model_alias in art.aliases][0]
            else:
                artifact = artifacts[model_alias]
            # artifact = run_.use_artifact(f'model-{run_id}:best', type='model')
            ckpt_path = artifact.file() #TODO: replace by get_wandb_model in utils
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
            

    def before_instantiate_classes(self):
        # Auto-wire the naturalness model's input dims from the data config so
        # backbone.in_channels and the model's mean/std selection can never
        # drift from the dataset. A global parser.link_arguments() can't be
        # used here: this CLI is shared across all tasks in subclass mode, and
        # a link on data.init_args.rh_idxs would fail to parse for every
        # non-naturalness config. So gate on the class paths instead.
        cfg = self.config[self.subcommand]
        data_cp = str((cfg.get('data') or {}).get('class_path', '') or '')
        model_cp = str((cfg.get('model') or {}).get('class_path', '') or '')
        if data_cp.endswith('NaturalnessDataModule') and model_cp.endswith('NaturalnessMapping'):
            from datasets.h5_dataset import resolve_rh_idxs, resolve_diversity
            di = cfg['data']['init_args']
            rh_idxs = di.get('rh_idxs', 'rh98')
            use_s2 = bool(di.get('use_s2', False))
            diversity = di.get('diversity', None)
            in_channels = int(
                (12 if use_s2 else 0)
                + len(resolve_rh_idxs(rh_idxs))
                + len(resolve_diversity(diversity)))
            mi = cfg['model']['init_args']
            mi['rh_idxs'] = rh_idxs
            mi['use_s2'] = use_s2
            mi['diversity'] = diversity
            backbone = mi.get('backbone')
            if backbone is not None and backbone.get('init_args') is not None:
                backbone['init_args']['in_channels'] = in_channels
            print(f'[naturalness] auto-set backbone.in_channels={in_channels} '
                  f'(rh_idxs={rh_idxs!r}, use_s2={use_s2}, diversity={diversity!r})')

            # Training only: refuse to start on missing/stale normalisation
            # stats. Eval/predict may legitimately use a different dataset, so
            # there the model's own mean_std_fp existence check suffices.
            if self.subcommand == 'fit':
                verify_naturalness_stats(mi.get('mean_std_fp'), di.get('data_file'))

        # create  shared memory array for caching predictions
        if self.subcommand == 'predict' and self.config[self.subcommand]['data']['init_args'].get('cache_predictions'):
            import zarr
            from pathlib import Path
            tile_id = self.config[self.subcommand]['data']['init_args'].get("tile_id")
            zarr_store_path = Path(self.config[self.subcommand]['data']['init_args']['pred_fp']).expanduser()
            try:
                store = zarr.open(zarr_store_path, mode='r')
            except Exception as e:
                print(f'Error opening zarr store: {e}, conda activate py3!')
            if zarr_store_path.name.endswith('.zarr'):
                key = f'{tile_id}/'
            else:
                key = '' # open a group as a store
            img_width, img_height = store[f'{key}s2'].shape[2:]
            if self.config[self.subcommand]['data']['init_args'].get('predict_full_profile'):
                size = (303, img_width, img_height) 
            else:
                size = (2, img_width, img_height)
            
            dtype = self.config[self.subcommand]['data']['init_args']['output_dtype']
            shm = create_shared_array(f'full_pred_{tile_id}', size, dtype)
            atexit.register(lambda: shm.close())
            atexit.register(lambda: shm.unlink())

    
    def after_instantiate_classes(self):
        # Surface the run the model/checkpoint came from so prediction
        # callbacks can name their outputs by the model rather than the fresh
        # eval run wandb spins up (logger.id is nulled for non-resume eval).
        # Same idiom as NaturalnessDataModule.data_file.
        if getattr(self, 'datamodule', None) is not None:
            self.datamodule.model_run_id = getattr(self, 'old_id', None)
        correct_bias = self.subcommand in ['validate', 'test', 'predict'] and self.config[self.subcommand].correct_bias or (
            self.config[self.subcommand].correct_bias and self.config[self.subcommand].get('quantize_model'))
        if correct_bias:
            self.config[self.subcommand]['model']['init_args']['evaluate_high_slope'] = True
            self.bias_correction()
            self.model.correct_bias = True
        else:
            self.model.correct_bias = False
        
        if self.subcommand == 'fit' and self.config[self.subcommand].get('use_pretrained_model'):
            self.use_pretrained_model()

        if self.config[self.subcommand].get('quantize_model'):
            self.quantize_model()

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
            if self.model.last_conv.bias.shape[0] > self.delta_bias.shape[0]:
                self.delta_bias = torch.cat([torch.tensor(self.delta_bias), torch.zeros(12)])
            self.model.last_conv.bias.data -= self.delta_bias
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
        for epoch in range(epochs):
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
            if self.model.out_channels > err.shape[0]*3:
                err = torch.cat([err, torch.zeros(12, device=device)])
            else:
                err = err.flatten()
            errors.append(err)
        errors = torch.stack(errors)
        errors = errors.T
        data = wandb.Table(
            data=errors.tolist(),
            columns=['me_gradual_slope', 'me_with_steep_slope', 'me_gradual_slope_veg'])
        wandb.log({"delta_biases": data})

        # last_conv.bias.data -= errors.squeeze()
        self.delta_bias = data.get_dataframe()[self.config[self.subcommand].bias_correction_column].values
        if self.model.last_conv.bias.shape[0] > self.delta_bias.shape[0]:
            self.delta_bias = torch.cat([self.delta_bias, torch.zeros(12)])
        self.model.last_conv.bias.data -= torch.tensor(self.delta_bias).to(self.model.device)
        model.val_metrics.reset()
        model.val_metrics_veg.reset()
        print(model.last_conv.bias)

    def sync_data_to_scratch(self):
        from pathlib import Path
        import shutil
        pred_fp = Path(self.config[self.subcommand]['data']['init_args']['pred_fp']).expanduser()
        tile_id = self.config[self.subcommand]['data']['init_args']['tile_id']
        pred_fp = Path(pred_fp).expanduser() / tile_id
        shutil.copytree(pred_fp, f'/scratch/{tile_id}')
        self.config[self.subcommand]['data']['init_args']['pred_fp'] = f'/scratch/{tile_id}'
    
    
    def quantize_model(self):
        import modelopt.torch.quantization as mtq
        import modelopt.torch.opt as mto
        model_path = f'checkpoints/fake_quantized_model_{self.old_id}.pth'
        if os.path.exists(model_path):
            print('model already quantized, loading from checkpoint')
            self.model = mto.restore(self.model, model_path)
            self.config[self.subcommand]['ckpt_path'] = None
            self.config_init[self.subcommand]['ckpt_path'] = None
            return
        config = mtq.INT8_DEFAULT_CFG
        # Temporarily disable distributed mode for calibration dataloader
        # since distributed process group hasn't been initialized yet
        original_distributed = self.datamodule.distributed
        self.datamodule.distributed = False
        
        checkpoint = torch.load(
                self.config[self.subcommand]['ckpt_path'],
                weights_only=False, map_location=self.model.device)
        self.config[self.subcommand]['ckpt_path'] = None
        self.config_init[self.subcommand]['ckpt_path'] = None  # model loaded from the checkpoint defined here
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model = self.model.to('cuda')

        def forward_loop(model):
            for i in range(20):
                dataloader = self.datamodule.train_dataloader()
                for sample in tqdm(dataloader):
                    with torch.no_grad():
                        x = model.transform(sample[0].to('cuda'))
                        x = model.process_slope(x, sample[3].to('cuda'))
                        x = model.process_latlon(x, sample[4].to('cuda'), sample[5].to('cuda'))
                        model.forward(x.float())
        self.model = mtq.quantize(self.model, config, forward_loop)
        mto.save(self.model, model_path)
        self.datamodule.distributed = original_distributed

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
                         parser_kwargs={"parser_mode": "omegaconf"},
                         ) # omegaconf: allow variable interpolation

    # cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # test on best model
    # cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == '__main__':

    # from ipdb import launch_ipdb_on_exception
    # with launch_ipdb_on_exception():
    cli_main()
        # evaluate_with_dask()