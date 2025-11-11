from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
import rasterio
import pystac
import stackstac
import xarray as xr
import matplotlib.pyplot as plt
import dask.array as da
from rasterio.crs import CRS
import dask
import time

def create_distance_arr(shape):
    rows = np.arange(shape[0], dtype=np.uint16)
    cols = np.arange(shape[1], dtype=np.uint16)

    # distance to top/bottom for each row; left/right for each col
    dy = np.minimum(rows, (shape[0] - 1) - rows)      # shape (H,)
    dx = np.minimum(cols, (shape[1] - 1) - cols)      # shape (W,)
    arr = np.minimum(dy[:, None], dx[None, :])  # shape (H, W), uint16
    return arr[None, :, :]


class Blending:
    def __init__(self, year: int, tile_id: str, stac_collection_dir: str,
                 save_dir: str, linear_correct: bool = True, small_area=True, **kwargs):
        self.year = year
        self.tile_id = tile_id
        self.stac_collection_dir = Path(stac_collection_dir).expanduser()
        self.save_dir = Path(save_dir).expanduser()

    def verify_blending(self):
        pass

    def create_distance_maps(self):
        save_dir = self.save_dir / f'distance_maps'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        @dask.delayed
        def create(tile_id: str):
            item = pystac.Item.from_file(str(self.stac_collection_dir / f'{tile_id}_2024/{tile_id}_2024.json'))
            distance_arr = create_distance_arr(item.assets['RH98'].extra_fields['proj:shape'])
            shape = item.assets['RH98'].extra_fields['proj:shape']
            profile = {
                "driver": "GTiff",
                "dtype": np.uint16,
                "count": 1,
                "width": shape[1],
                "height": shape[0],
                "crs": CRS.from_epsg(item.properties['proj:epsg']),
                "transform": rasterio.Affine(*item.assets['RH98'].extra_fields['proj:transform']),
                "compress": "zstd",
                "predictor": 2,
                "blockxsize": 256,
                "blockysize": 256,
                "tiled": True,
                "interleave": "band",
            }
            with rasterio.open(save_dir / f'{tile_id}.tif', 'w', **profile) as dst:
                dst.write(distance_arr)
            print(f'Saved distance map to {save_dir / f'{tile_id}.tif'}')
        
        tile_ids = self.stac_collection_dir.glob(f'*_2024') #NOTE: 2024 has more tiles, and contains all 2020 tiles
        tile_ids = [tile_id.stem.split('_')[0] for tile_id in tile_ids]
        unfinished_tile_ids = []
        for tile_id in tile_ids:
            if (save_dir / f'{tile_id}.tif').exists():
                continue
            unfinished_tile_ids.append(tile_id)
        if len(unfinished_tile_ids) == 0:
            print('All distance maps already created')
            return
        print(f'Creating distance maps for {len(unfinished_tile_ids)} tiles')
        tasks  = []
        for tile_id in unfinished_tile_ids:
            tasks.append(create(tile_id))
        dask.compute(*tasks)

        
    def verify_blending(self,
            stac_collection, tile_id: str = '21MXQ', linear_correct: bool = True, correct_param_dir: str = None, year: int = 2020,
            small_area=True):
        '''
        Verify blending on a small area & large tiles
        '''
        NODATA = 32767
        save_dir = Path(f'~/data/gvs/deploy/blending').expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        correct_param_dir = Path(correct_param_dir).expanduser()
        # TODO: don't mix current tile id with the others
        current_tile = pystac.Item.from_file(
            str(stac_collection.catalog_dir / stac_collection.collection_id / tile_id / f'{tile_id}.json'))
        current_image = stackstac.stack(
            [current_tile],
            epsg=current_tile.properties['proj:epsg'],
            resolution=10, rescale=False, dtype='float32', fill_value=np.float32(np.nan))
        distance_current_image = create_distance_arr(current_image.shape[2:])
        # apply linear correction
        correct_param = np.load(correct_param_dir / f'correction_stats_{year}_{tile_id}.npz')
        if linear_correct:
            scale = correct_param['scale'][None, :, None, None]  # (time, band, y, x)
            shift = correct_param['shift'][None, :, None, None]
        else:
            scale = np.ones(current_image.shape[1])[None, :, None, None]
            shift = correct_param['bias'][None, :, None, None]
        current_image = current_image * scale + shift
        bbox_local = current_tile.assets['RH98'].extra_fields['proj:bbox']
        test_tiles = ['21MXQ', '21MWP', '21MXP', '21MWQ', '21MYQ', '21MYP']
        items = []
        for tile_id in test_tiles:  # intersects.Name
            if tile_id == current_tile.id:
                continue
            item = pystac.Item.from_file(
                str(stac_collection.catalog_dir / stac_collection.collection_id / tile_id / f'{tile_id}.json'))
            # create the distance map and add to item as an asset
            distance_map_path = Path(f'~/data/GVS/Deploy/blending/distance_maps/{tile_id}_distance_map.tif').expanduser()
            if not distance_map_path.exists():
                self.create_distance_map(item, distance_map_path)
            item.add_asset('distance_to_border', pystac.Asset(
                href=distance_map_path,
                media_type="image/tiff; application=geotiff; profile=cloud-optimized",
                title=f'Distance map - 10m',
                roles=["data"],
            ))
            item_image = stackstac.stack([item], epsg=item.properties['proj:epsg'], resolution=10,
                                        bounds=bbox_local, rescale=False, dtype='float32', fill_value=np.float32(np.nan))
            if item_image.shape[0] == 0:
                print(f'{tile_id} has no overlap with {current_tile.id}')
                continue
            correct_param = np.load(correct_param_dir / f'correction_stats_{year}_{tile_id}.npz')
            if linear_correct:
                scale = correct_param['scale'][None, :, None, None]  # (time, band, y, x)
                shift = correct_param['shift'][None, :, None, None]
            else:
                scale = np.ones(current_image.shape[1])[None, :, None, None]
                shift = correct_param['bias'][None, :, None, None]
            distance_image = item_image.sel(band=['distance_to_border'])
            rhs_image = item_image.isel(band=slice(0, 101))
            rhs_image = rhs_image * scale + shift
            item_image = xr.concat([rhs_image, distance_image], dim='band')
            items.append(item_image)

        # overlapped_images = stackstac.stack(items, epsg=current_tile.properties['proj:epsg'], resolution=10, bounds=bbox_local, rescale=False, dtype='float32', fill_value=np.float32(np.nan))
        overlapped_images = xr.concat(items, dim='time')

        # overlapped_images = overlapped_images.where(overlapped_images != NODATA, np.nan)
        # test for small area
        if small_area:
            distance_current_image = distance_current_image[0, 8932:10980, 0:2048]
            current_image = current_image.isel(x=slice(0, 2048), y=slice(8932, 10980)).sel(band=['RH98']).compute()
            # save current image
            plt.figure()
            current_image.isel(time=0).sel(band=['RH98']).plot()
            plt.savefig(save_dir / f'pred_{current_tile.id}.png')
            plt.close()

            # save valid area of current image
            plt.figure()
            plt.imshow(distance_current_image)
            plt.savefig(save_dir / f'distance_{current_tile.id}.png')
            plt.close()

            # save overlapped images

            overlapped_images = overlapped_images.isel(
                x=slice(0, 2048),
                y=slice(8932, 10980)).sel(
                band=['RH98', 'distance_to_border'])
            overlapped_images = overlapped_images.compute()
            for i, tile_id in enumerate(overlapped_images.id.data):
                plt.figure()
                overlapped_images.isel(time=i).sel(band=['RH98']).plot()
                plt.savefig(save_dir / f'pred_{tile_id}.png')
                plt.close()

            # save valid area of overlapped images
            valid = overlapped_images.sel(band=['distance_to_border'])  # shape = (time, y, x), dtype=bool
            for i, tile_id in enumerate(valid.id.data):
                plt.figure()
                valid.isel(time=i).sel(band=['distance_to_border']).plot()
                plt.savefig(save_dir / f'distance_{tile_id}.png')
                plt.close()

            # valid = valid.chunk({"time": 1, "y": 2048, "x": 2048})
            # weights = xr.apply_ufunc(
            #     smooth_mask,
            #     valid,
            #     input_core_dims=[["y","x"]],
            #     output_core_dims=[["y","x"]],
            #     kwargs={"sigma": 50},   # feather width in pixels
            #     vectorize=True,
            #     # dask="parallelized",
            #     output_dtypes=[np.float32],
            # )
            weights = np.concatenate([overlapped_images.sel(band='distance_to_border').data,
                                    distance_current_image[None, :, :]], axis=0)
            weights = np.nan_to_num(weights, 0)
            sum_w = weights.sum(axis=0)
            weights_normalized = np.where(sum_w > 0, weights / sum_w[None, :, :], 0)
            weights_normalized = weights_normalized[:, None, :, :]  # .repeat(2, axis=1)
            images = xr.concat([overlapped_images.sel(band=['RH98']), current_image], dim='time')
            blended_images = images * weights_normalized
            blended_images = blended_images.sum(dim='time')
            blended_images = blended_images.where(current_image.notnull(), np.nan)
            plt.figure()
            blended_images.isel(band=0, time=0).plot()
            plt.savefig(save_dir / f'{current_tile.id}_blended.png')
            plt.close()

            # difference between blended and current tile
            difference = blended_images - current_image
            plt.figure()
            difference.isel(band=0, time=0).plot()
            plt.savefig(save_dir / f'{current_tile.id}_difference.png')
            plt.close()
        else:
            # test on a large area with a single band
            band_id = ['RH98']
            distance_overlapped_images = overlapped_images.sel(band=['distance_to_border'])
            overlapped_images = overlapped_images.sel(band=band_id)
            current_image = current_image.sel(band=band_id)
            distance_current_image = da.from_array(
                distance_current_image, chunks=distance_overlapped_images.data.chunksize[1:])
            weights = da.concatenate([distance_overlapped_images.data, distance_current_image[None, :, :, :]], axis=0)
            weights = da.nan_to_num(weights, 0)
            sum_w = weights.sum(axis=0)
            weights_normalized = np.where(sum_w > 0, weights / sum_w[None, :, :], 0)
            images = xr.concat([overlapped_images, current_image], dim='time')
            blended_images = images * weights_normalized
            blended_images = blended_images.sum(dim='time')
            blended_images = blended_images.where(current_image.notnull(), np.nan)
            blended_images = blended_images.round()
            blended_images = blended_images.fillna(NODATA)
            blended_images = blended_images.astype(np.int16)
            blended_images = blended_images.compute()
            with rasterio.open(current_tile.assets[band_id[0]].href) as src:
                profile = src.profile
            postfix = '_linear_corrected' if linear_correct else '_bias_corrected'
            print(f'saved to {save_dir / f'{current_tile.id}_blended{postfix}.tif'}')
            with rasterio.open((save_dir / f'{current_tile.id}_blended{postfix}.tif'), 'w', **profile) as dst:
                dst.write(blended_images.isel(band=0, time=0).data, 1)


@dataclass
class Config:
    stac_collection_dir: str = '~/data/gvs/deploy/gvsm_stac_catalog/vsm_local'
    save_dir: str = '~/data/gvs/deploy/blending'
    tile_id: str = '21MXQ'
    year: int = 2020
    
    linear_correct: bool = False
    small_area: bool = False
    task: str = 'create_distance_maps'


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_name="config", version_base='1.2')
def main(cfg):
    print(cfg)
    blending = Blending(**cfg)
    t0 = time.time()
    if cfg.task == 'create_distance_maps':
        blending.create_distance_maps()
    print(f'Time taken: {time.time() - t0} seconds')
if __name__ == '__main__':
    main()
