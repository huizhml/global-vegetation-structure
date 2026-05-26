from pathlib import Path
import hydra
from config.loader import register
from config.runner import run_cli
from tools.utils import resolve_args

# Register every op under the `visualization` section of config/viz/config.yaml
# as a Hydra `run=<name>` choice. Mirrors the per-module entrypoints in
# evaluation/ (see config/eval/config.yaml).
register(
    Path(__file__).resolve().parents[1] / 'config' / 'viz' / 'config.yaml',
    section='visualization',
    default_run='resample_and_mosaic',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    resolve_args(cfg)
    run_cli(cfg)


if __name__ == "__main__":
    main()
