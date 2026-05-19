# Conformal Prediction (CP) for Global Vegetation Structure

This package implements Conformal Prediction (CP) calibration and evaluation for global vegetation structure models. It adjusts prediction intervals from quantile regression models to achieve valid marginal coverage guarantees.

## Supported CP methods

| Method     | Description                                        |
|------------|----------------------------------------------------|
| `CQR`      | Conformalized Quantile Regression                  |
| `CQR-r`    | CQR with interval-width normalisation              |
| `CQR-m`    | CQR with asymmetric (median-based) normalisation   |
| `SE-CQR`   | Signed-error CQR                                   |
| `SE-CQR-r` | Signed-error CQR with interval-width normalisation |
| `SE-CQR-m` | Signed-error CQR with asymmetric normalisation     |

## Installation

```bash
pip install -e ./src-cp
```

## Usage

All steps of the workflow — fitting CP quantiles, evaluating CP performance, and generating plots — are performed by `scripts/cp/run_cp.py`. The script requires a YAML configuration file specifying dataset paths, CP settings, and figure settings. An example is provided at `config/cp/extended-config.yaml`.

> **Note:** Input data must be in **metres**. The script converts values to decimetres internally. Providing data in other units will produce incorrect results.

### Run the full workflow

```bash
python3 scripts/cp/run_cp.py --config_path config/cp/extended-config.yaml
```

Intermediate results (fitted CP quantiles, evaluation metrics) and final figures are saved in the same directory as the configuration file.

### Regenerate figures only

If CP quantiles and evaluation results already exist, use `--plot_only` to skip fitting/evaluation and regenerate figures only:

```bash
python3 scripts/cp/run_cp.py --config_path config/cp/extended-config.yaml --plot_only
```

## Package structure

| Module         | Description                                                                             |
|----------------|-----------------------------------------------------------------------------------------|
| `cp.py`        | Core CP methods and `ConformalPredictor` class                                          |
| `dl.py`        | Data loading and preprocessing (parquet, unit conversion, quantile crossing correction) |
| `config.py`    | YAML config parsing and plot style settings                                             |
| `constants.py` | Biome mappings and colour palettes                                                      |
| `plot.py`      | Coverage and interval-width visualisations                                              |
