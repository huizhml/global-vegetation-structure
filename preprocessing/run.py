from pathlib import Path
import hydra
from config.loader import register
from config.runner import run_cli

# Register every op under the `preprocessing` section of
# config/preprocessing/config.yaml as a Hydra `run=<name>` choice. Mirrors the
# per-module entrypoints in evaluation/ (see config/eval/config.yaml).
#
# NOTE: default_run was previously 'repartition_data', which is a postprocessing
# op and was never registered here (so `python -m preprocessing.run` with no
# override failed to compose). Pointed at a real preprocessing op instead.
register(
    Path(__file__).resolve().parents[1] / 'config' / 'preprocessing' / 'config.yaml',
    section='preprocessing',
    default_run='cal_naturalness_input_stats',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == "__main__":
    main()
