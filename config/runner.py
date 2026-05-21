import os
import time
from hydra.utils import instantiate
from omegaconf import OmegaConf
from postprocessing.core.utils import generate_run_log


def run_cli(cfg):
    t0 = time.time()
    print(OmegaConf.to_yaml(cfg))
    if cfg.run.target_type == 'function':
        instantiate(cfg.run)
    elif cfg.run.target_type == 'class':
        obj = instantiate(cfg.run)
        getattr(obj, cfg.run.target_method)(**cfg.run.func_args)
    else:
        raise ValueError(f"Invalid target_type: {cfg.run.target_type}")
    runtime = time.time() - t0
    print(f'Time taken: {runtime:.2f} seconds')
    if cfg.run.get('save_dir', None) is not None:
        generate_run_log(os.path.join(cfg.run.save_dir, 'run.log'), cfg, runtime)
    else:
        print('No save_dir provided, skipping run log')
