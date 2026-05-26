from pathlib import Path
import hydra
from config.loader import register
from config.runner import run_cli
from tools.utils import resolve_args

# Register every op under the `postprocessing` section of
# config/postprocessing/config.yaml as a Hydra `run=<name>` choice. Mirrors the
# per-module entrypoints in evaluation/ (see config/eval/config.yaml).
register(
    Path(__file__).resolve().parents[1] / 'config' / 'postprocessing' / 'config.yaml',
    section='postprocessing',
    default_run='add_ours_to_sota_gedi',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    resolve_args(cfg.run)
    run_cli(cfg)


if __name__ == "__main__":
    main()
