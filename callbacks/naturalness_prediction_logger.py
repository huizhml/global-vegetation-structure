import torch
from typing import Any
from lightning.pytorch.callbacks.callback import Callback
import pandas as pd
import geopandas as gpd
from pathlib import Path


class NaturalnessPredictionLogger(Callback):
    '''Dump per-sample naturalness predictions from a `test` run to GeoParquet.

    Same contract as callbacks.prediction_logger.PredictionLogger: the
    model's test_step returns (probs, y, rowid); this collects them across
    the epoch and writes one row per sample.

    `rowid` is the join key back to the reference_data_set CSV. Longitude /
    Latitude are pulled from the datamodule's (rowid-indexed) val target
    frame, then the frame is promoted to a GeoDataFrame with point geometry
    in EPSG:4326 (same idiom as evaluation.on_naturalness) and written as
    GeoParquet -- so it opens straight on a map in QGIS / geopandas. Biome
    is intentionally not added here: it is a spatial join done downstream by
    postprocessing.core.extract_sparse_points.add_biome.

    save_dir defaults to None -> write beside the data_file (the directory
    NaturalnessDataModule loaded from). Pass an explicit save_dir to override.

    Columns: rowid, class_idx (truth), pred (argmax), prob_0..prob_{C-1},
    Longitude, Latitude, geometry.
    '''

    def __init__(self,
                 save_dir: str = None,
                 outfile_suffix: str = '',
                 latlon_cols=('Longitude', 'Latitude'),
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.save_dir = Path(save_dir).expanduser() if save_dir else None
        self.outfile_suffix = outfile_suffix
        self.lon_col, self.lat_col = tuple(latlon_cols)

    def _resolve_save_dir(self, trainer) -> Path:
        if self.save_dir is not None:
            save_dir = self.save_dir
        else:
            # Beside the data: NaturalnessDataModule.data_file's directory.
            save_dir = Path(trainer.datamodule.data_file).expanduser().parent
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def on_test_epoch_start(self, trainer, pl_module):
        self._probs, self._y, self._rowid = [], [], []

    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        probs, y, rowid = outputs
        self._probs.append(probs.cpu())
        self._y.append(y.cpu())
        self._rowid.append(rowid.cpu())
        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self, trainer, pl_module):
        probs = torch.cat(self._probs).numpy()
        y = torch.cat(self._y).numpy()
        rowid = torch.cat(self._rowid).numpy()
        pred = probs.argmax(axis=1)

        df = pd.DataFrame({'rowid': rowid, 'class_idx': y, 'pred': pred})
        for c in range(probs.shape[1]):
            df[f'prob_{c}'] = probs[:, c]

        # Attach geolocation by rowid from the val target frame (rowid-indexed),
        # then promote to a GeoDataFrame (EPSG:4326 points) so it maps directly.
        tdf = getattr(trainer.datamodule, 'target_df_val', None)
        have_latlon = (tdf is not None
                       and self.lon_col in tdf.columns
                       and self.lat_col in tdf.columns)
        if have_latlon:
            df[self.lon_col] = tdf[self.lon_col].reindex(df['rowid']).to_numpy()
            df[self.lat_col] = tdf[self.lat_col].reindex(df['rowid']).to_numpy()
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df[self.lon_col], df[self.lat_col]),
                crs='EPSG:4326')
        else:
            print(f'[NaturalnessPredictionLogger] WARNING: '
                  f'{self.lon_col!r}/{self.lat_col!r} unavailable; writing '
                  f'non-spatial parquet (no map geometry)')
            gdf = df

        run_id = trainer.logger._experiment.id
        out_fp = self._resolve_save_dir(trainer) / \
            f'naturalness_predictions_{run_id}{self.outfile_suffix}.parquet'
        gdf.to_parquet(out_fp, index=False)
        acc = float((pred == y).mean()) if len(y) else float('nan')
        print(f'saved {len(df)} naturalness predictions to {out_fp} '
              f'(overall accuracy {acc:.4f})')
        return super().on_test_epoch_end(trainer, pl_module)
