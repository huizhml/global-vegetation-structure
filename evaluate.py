import wandb
import os
import ee
import h5py
import numpy as np
import dask
import dask.dataframe as dd
import dask_geopandas as dgp
from dask.utils import natural_sort_key
from shapely import wkt
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
import hydra
from retry import retry
import requests
from io import StringIO
from shapely.geometry import shape
import json
from download._utils import authenticate  
# import geemap
# geemap.df_to_ee

authenticate()

def set_fc_properties(row):
    geom = row.geometry
    row = row.drop('geometry')
    fc = ee.Feature(ee.Geometry.Point([geom.x, geom.y]), row.to_dict())
    fc = fc.set('index', row.name)
    return fc

@retry(requests.HTTPError, tries=10, delay=1)
def sample_canopy_height(partition):
    if len(partition) ==0:
        print('empty partition')
        print(partition)
        return
    zone = partition.path.iloc[0].split('/')[1]
    file = f'~/data/GVS/evaluation/existing_canopy_height_pred_val/{zone}.parquet'
    file = Path(file).expanduser()
    print(f'Processing {zone}', file.exists())
    if file.exists():
        old_df = pd.read_parquet(file)
        
        if len(old_df) == len(partition) and (old_df.index == partition.index).all():
            print(f'{zone} exists and matches, skipping')
            return
        #     old_df[['path','in_partition_idx']] = partition[['path','in_partition_idx']]
        #     print(f'{zone} already exists, updating path and in_partition_idx')
        #     old_df.to_parquet(file)
        #     return
        else:
            print(len(old_df), len(partition))
            print('reprocessing', zone)
            os.remove(file)
    # eefc = ee.Geometry.Point([partition.iloc[0].geometry.x, partition.iloc[0].geometry.y])
    # ee.Feature(eefc, {'index', partition.index[0]})
    if len(partition) > 10000:
        partitions = [partition.iloc[i:i+10000] for i in range(0, len(partition), 10000)]
    else:
        partitions = [partition]

    # if len(partitions[-1]) == 1:
    #     partitions[-2] = partitions[-2].append(partitions[-1])
    #     partitions = partitions[:-1]

    df1s, df2s = [], []
    for p in partitions:
        eefc = p.apply(set_fc_properties, axis=1)
        eefc = ee.FeatureCollection(eefc.tolist())
        # points = ee.Geometry.MultiPoint(partition.geometry.apply(lambda x: [x.x, x.y]).tolist())
        img_um = canopy_height_um.filterBounds(eefc).mosaic().divide(100).rename('RH100_UM')
        img_umd = canopy_height_umd.filterBounds(eefc).mosaic().rename('RH95_UMD')
        img_meta = canopy_height_meta.filterBounds(eefc).mosaic().rename('RH95_META')
        group1 = canopy_height_eth.addBands(img_umd)
        group2 = img_um.addBands(img_meta)
        # EPSG:4326
        fc1 = group1.sampleRegions(
            collection=eefc,
            scale=10
        )
        # EPSG:3857
        fc2 = group2.sampleRegions(
            collection=eefc,
            scale=10
        )
        dfs = []
        
        if fc1.size().getInfo() == 0 or fc2.size().getInfo() == 0:
            print('nothing found')
            df1s.append(pd.DataFrame(np.nan, columns=['RH95_UMD','RH98_ETH'], index=p.index))
            df2s.append(pd.DataFrame(np.nan, columns=['RH100_UM','RH95_META'], index=p.index))
            continue
        for fc in [fc1, fc2]:
            download_id = ee.data.getTableDownloadId({'table': fc, 'fileFormat': 'CSV'})
            res = requests.get(ee.data.makeTableDownloadUrl(download_id))
            if res.status_code == 200:
                data = StringIO(res.content.decode('utf-8'))
                df = pd.read_csv(data)
                df = df.drop(columns=['.geo', 'system:index']).set_index('index')
                dfs.append(df)
            else:
                raise requests.HTTPError(f'Failed to download ')
        df1s.append(dfs[0])
        df2s.append(dfs[1])
    if len(df1s) == 0:
        return
    df1s = pd.concat(df1s)
    df2s = pd.concat(df2s)
    if 'RH95_UMD' not in df1s.columns:
        df1s['RH95_UMD'] = pd.NA
    if 'RH98_ETH' not in df1s.columns:
        df1s['RH98_ETH'] = pd.NA
    if 'RH100_UM' not in df2s.columns:
        df2s['RH100_UM'] = pd.NA
    if 'RH95_META' not in df2s.columns:
        df2s['RH95_META'] = pd.NA
    partition.loc[df1s.index, ['RH95_UMD','RH98_ETH']] = df1s[['RH95_UMD','RH98_ETH']]
    partition.loc[df2s.index, ['RH100_UM','RH95_META']] = df2s[['RH100_UM','RH95_META']]
    partition = partition.drop(columns=['geometry'])
    partition.to_parquet(file)



def sample_canopy_height_maps(index_dir, save_dir):
    index_dir = Path(index_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)
    index_fps = list(index_dir.glob('*.parquet'))
    target_files = [save_dir / f'{fp.stem}.parquet' for fp in index_fps]
    index_fps = [str(fp) for fp in index_fps]
    index_fps = sorted(index_fps, key=natural_sort_key)
    for f in target_files:
        if f.exists():
            print(f'{f} exists, skipping')
            print(f'removing {index_dir / f"{f.stem}.parquet"} from index_fps')
            index_fps.remove(str(index_dir / f'{f.stem}.parquet'))

    # index_fps = [fp for fp in index_fps if '01L' in fp]
    # print(index_fps)
    ddf = dgp.read_parquet(index_fps, gather_spatial_partitions=False, columns=['geometry', 'path', 'in_partition_idx', 's2_tile'])
    # df = ddf.get_partition(0).compute()
    # sample_canopy_height(df)
    ddf.map_partitions(sample_canopy_height, meta=(None, None)).compute()


def get_gedi_rhs(h5_dir, canopy_height_df_dir):
    h5_dir = Path(h5_dir).expanduser()
    canopy_height_df_dir = Path(canopy_height_df_dir).expanduser()
    save_dir = canopy_height_df_dir.parent / 'existing_canopy_height_pred_val_with_gedi'
    save_dir.mkdir(exist_ok=True, parents=True)
    files = list(canopy_height_df_dir.glob('*.parquet'))
    files = sorted(files)
    target_files = [save_dir / f'{f.stem}.parquet' for f in files]
    for f in target_files:
        if f.exists():
            print(f'{f} exists, skipping')
            files.remove(canopy_height_df_dir / f'{f.stem}.parquet')
    # files = [f for f in files if '20T' in f.stem]
    df = dd.read_parquet(files)
    # df = df.get_partition(0).compute()
    # get_canopy_height_per_zone(df, h5_dir, save_dir)
    df.map_partitions(get_canopy_height_per_zone, h5_dir, save_dir, meta=(None, None)).compute()

def get_canopy_height_per_zone(partition, h5_dir, save_dir):
    if len(partition) == 0:
        return
    partition.index.name = 'index'
    zone = partition.path.iloc[0].split('/')[1]
    def get_rhs(group):
        path = group.path.iloc[0][4:]
        group = group.sort_values('in_partition_idx')
        with h5py.File(h5_dir / f'{zone}.h5', 'r') as f:
            rhs = f[f'{path}/rhs'][:, [95,98,100]]
        df = pd.DataFrame(rhs, columns=['RH95_GEDI', 'RH98_GEDI', 'RH100_GEDI'], index=group.index)
        df = df.merge(group, left_index=True, right_index=True)
        return df
    res = partition.groupby('path').apply(get_rhs)
    res = res.set_index(res.index.get_level_values(1))
    res.to_parquet(save_dir / f'{zone}.parquet')


def add_biome(val_df_fp, sota_chm_df_dir):
    val_df_fp = Path(val_df_fp).expanduser()
    sota_chm_df_dir = Path(sota_chm_df_dir).expanduser()
    val_df = pd.read_parquet(val_df_fp, columns=['rh100', 'lat', 'lon', 'BIOME']) 
    ddf = dd.read_parquet(f'{sota_chm_df_dir}/*.parquet', index=False) #TODO: add latlon to check alignment
    ddf = ddf.compute() 
    ddf = ddf.drop(columns=['index']).reset_index(drop=True) # NOTE: currently relying on the fixed order to match val_df, (ddf['RH100_GEDI'] == val_df['rh100']).all() is true
    ddf[['BIOME']] = val_df[['BIOME']]
    ddf.to_parquet(sota_chm_df_dir.parent / f'sota_chm_val_with_gedi_biome.parquet')
    print(f'{sota_chm_df_dir.parent / f"sota_chm_val_with_gedi_biome.parquet"} saved')
    return val_df


def compare_result_precision(run_id, corrected=False):
    suffix = '_corrected' if corrected else ''
    ddf_ours = pd.read_parquet(f'output/canopy_height_predictions_{run_id}{suffix}.parquet')
    ddf_ours = ddf_ours[(ddf_ours['slope_mask']==1) & (ddf_ours['veg_mask']==1)]
    columns = ddf_ours.columns
    pred_cols = [c for c in columns if c.startswith('RH') and '_GEDI' not in c]
    gedi_cols = [c for c in columns if c.startswith('RH') and '_GEDI' in c]
    gedi_ref = ddf_ours[gedi_cols].rename(columns=lambda x: x.replace('_GEDI', ''))
    comp_dfs = []
    for round_precision in range(0,3):
        preds = ddf_ours[pred_cols].round(round_precision)
        diff = preds - gedi_ref
        rmse = ((diff)**2).mean()**0.5
        mae = diff.abs().mean()
        me = diff.mean()
        df = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'ME': me}, index=pred_cols)
        # unstack the index to make grouped columns
        df_unstacked = df.unstack(level=1)
        df_unstacked = pd.DataFrame(df_unstacked).T
        comp_dfs.append(df_unstacked)
    comp_dfs = pd.concat(comp_dfs, keys=[f'precision_{i}' for i in range(0,3)])
    comp_dfs.to_csv(f'output/evaluation/compare_round_precision_val_{run_id}{suffix}.csv')
        

def compare_with_sota_maps(sota_chm_df_fp, run_id, corrected=False):
    # df_dir = Path(df_dir).expanduser()

    # ddf = dd.read_parquet(f'{df_dir}/*.parquet', index=False)
    # ddf = ddf.compute()
    sota_chm_df_fp = Path(sota_chm_df_fp).expanduser()
    ddf = pd.read_parquet(sota_chm_df_fp)
    suffix = '_corrected' if corrected else ''
    ddf_ours = pd.read_parquet(f'output/canopy_height_predictions_{run_id}{suffix}.parquet')
    ddf_ours = ddf_ours.rename(columns=lambda x: x+'_ours' if x.startswith('RH') and '_' not in x else x)
    ddf[['RH95_ours', 'RH98_ours', 'RH100_ours', 'slope_mask', 'veg_mask']] = ddf_ours[['RH95_ours', 'RH98_ours', 'RH100_ours', 'slope_mask', 'veg_mask']]
    
    ddf = ddf.dropna(subset=['RH95_META', 'RH95_UMD', 'RH98_ETH', 'RH100_UM'])
    comp_dfs = []
    for filter in ['Base', 'slope', 'veg']:
        if filter == 'Base':
            ddf = ddf
        else:
            ddf = ddf[ddf[f'{filter}_mask']==1]
        rmse = []
        mae = []
        me = []
        for product, name in [('RH95_UMD', 'RH95_GEDI'), ('RH98_ETH', 'RH98_GEDI'), ('RH100_UM', 'RH100_GEDI'), ('RH95_META', 'RH95_GEDI')]:
            rmse.append(((ddf[product] - ddf[name])**2).mean()**0.5)
            mae.append((ddf[product] - ddf[name]).abs().mean())
            me.append((ddf[product] - ddf[name]).mean())
        for name in ['RH95', 'RH98', 'RH100']:
            rmse.append(((ddf[f'{name}_ours'] - ddf[f'{name}_GEDI'])**2).mean()**0.5)
            mae.append((ddf[f'{name}_ours'] - ddf[f'{name}_GEDI']).abs().mean())
            me.append((ddf[f'{name}_ours'] - ddf[f'{name}_GEDI']).mean())
        
        df = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'ME': me}, index=['UMD', 'ETH', 'UM', 'META', 'OURS(RH95)', 'OURS(RH98)', 'OURS(RH100)'])
        comp_dfs.append(df)

    comp_dfs = pd.concat(comp_dfs, axis=1)
    comp_dfs.to_csv(f'output/evaluation/comparison_metrics_{run_id}{suffix}.csv')


@dataclass
class MyConfig:
    sota_chm_df_fp: str = '~/data/GVS/evaluation/sota_chm_val_with_gedi_biome.parquet'
    run_id: str = ''
    output_dir: str = 'output'
    task: str = 'aggregate_gedi_by_biome'
    corrected: bool = False

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(config_name="my_config", version_base="1.2")
def main(cfg):
    compare_result_precision(cfg.run_id, cfg.corrected)
    # add_biome(val_df_fp='~/data/GVS/train_subsets/val_filtered_v1.parquet', sota_chm_df_dir='~/data/GVS/evaluation/existing_canopy_height_pred_val_with_gedi')

if __name__ == '__main__':
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=8, dashboard_address=':38787')
    client = Client(cluster)

    index_dir = '~/data/GVS/split_test0.1_cal0.1_val0.1_seed42_v1/index_table_val'
    save_dir = '~/data/GVS/evaluation/existing_canopy_height_pred_val'
    canopy_height_eth = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').rename('RH98_ETH')
    canopy_height_umd = ee.ImageCollection("users/potapovpeter/GEDI_V27")
    canopy_height_meta = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight")
    canopy_height_um = ee.ImageCollection('projects/worldwidemap/assets/canopyheight2020')

    # sample_canopy_height_maps(index_dir, save_dir)
    h5_dir = '~/data/GVS/split_test0.1_cal0.1_val0.1_seed42_v1/val_h5s'
    # get_gedi_rhs(h5_dir, save_dir)
    # for file in Path('~/data/GVS/evaluation/existing_canopy_height_pred_val_with_gedi').expanduser().glob('*.parquet'):
    #     df = pd.read_parquet(file)
    #     zone = file.stem
    #     if 'RH95_UMD' not in df.columns:
    #         try:
    #             file2 = Path(f'~/data/GVS/evaluation/existing_canopy_height_pred_val/{zone}.parquet').expanduser()
    #             df_ = pd.read_parquet(file2)
    #             df.loc[:, 'RH95_UMD'] = df_['RH95_UMD']
    #             df.to_parquet(file)
    #         except:
    #             print(f'Failed to update {zone}')
    #             os.remove(file2)
    #             os.remove(file)
        
    main()

    
