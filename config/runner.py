import os
import time
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from postprocessing.core.utils import generate_run_log, link_results_dir


def run_cli(cfg):
    t0 = time.time()
    print(OmegaConf.to_yaml(cfg))

    # Results stay physically in save_dir (the data tree); index them under
    # root_results_dir/<section>/<op> with a symlink pointing back, so all
    # results are browsable in one place under a clean, logical name —
    # decoupled from where the data physically lives, and without moving bytes.
    save_dir = cfg.run.get('save_dir', None)
    root_results_dir = cfg.get('root_results_dir', None)
    if save_dir is not None and root_results_dir is not None:
        op_name = HydraConfig.get().runtime.choices.get('run')
        section = cfg.get('section', None)
        index_name = f'{section}/{op_name}' if section else op_name
        if index_name:
            link_results_dir(save_dir, root_results_dir, index_name)

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
