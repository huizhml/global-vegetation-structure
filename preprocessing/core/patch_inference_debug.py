import wandb
import hydra
from omegaconf import DictConfig
from download.core.utils import get_class
import torch
from typing import Union
import geopandas as gpd
from shapely.geometry import box
from download.core.constants import S2_ITEM_PROPS
from download.core.utils import get_patch, row_to_stac_item
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

bands = [
            'B01', 'B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',
            'B09', 'B11', 'B12', 'SCL'
        ]

def get_model(model_cfg, run_id: str, wandb_project: str, bias_correction_column: str = None, model_alias: Union[str, int] = 'best'):
    # init model
    activation_layer = get_class(model_cfg.init_args.activation_layer)()
    del model_cfg.init_args.activation_layer
    model = get_class(model_cfg.class_path)(**model_cfg.init_args, activation_layer=activation_layer)
    # load model weights
    api = wandb.Api()
    run = api.run(f"{wandb_project}/{run_id}")
    artifacts = run.logged_artifacts()
    artifacts = [artifact for artifact in artifacts if artifact.type == 'model']
    if isinstance(model_alias, str):
        artifact = [art for art in artifacts if model_alias in art.aliases][0]
    else:
        artifact = artifacts[model_alias]
    ckpt_path = artifact.file()
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # bias correction
    table = api.artifact(f'{wandb_project}/run-{run_id}-delta_biases:latest')
    df = table.get('delta_biases').get_dataframe()
    delta_bias = df[bias_correction_column].values.astype('float32')
    if model.last_conv.bias.shape[0] > delta_bias.shape[0]:
        delta_bias = torch.cat([torch.tensor(delta_bias), torch.zeros(12)])
    model.last_conv.bias.data -= delta_bias
    return model

def get_data(metadata_file, tile_id, n_iamges_per_tile):
    s2_df = gpd.read_parquet(metadata_file)
    tile_df = s2_df[s2_df['s2:mgrs_tile'] == tile_id].set_index('id')
    if len(tile_df)>n_iamges_per_tile:
        if (tile_df['s2:nodata_pixel_percentage']==0).sum() > 0:
            tile_df = tile_df.sort_values(['s2:nodata_pixel_percentage', 'eo:cloud_cover']).head(n_iamges_per_tile)
        else:
            idx = tile_df.groupby('orbit')['eo:cloud_cover'].nsmallest(n_iamges_per_tile//2).index.get_level_values(1)
            tile_df = tile_df.loc[idx]
    tile_df['datetime'] = tile_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    bbox = box(*tile_df.total_bounds)
    items = row_to_stac_item(tile_df, S2_ITEM_PROPS)
    epsg = items[0].properties['proj:epsg']
    image = get_patch(items, bands, dtype='uint16', fill_value=np.uint16(0))
    image.name = 's2'
    del image.attrs['spec']
    del image.attrs['crs']
    return image
    

def batch_inference(patch):
    # not finished
    print(patch.shape)
    return patch
    
    
def visualize_rh98(images, name=''):
    for i in range(images.shape[0]):
        image = images[i].numpy()
        plt.figure()
        plt.imshow(image)
        plt.colorbar()
        plt.savefig(f'output/32MRE_rh98_{name}_{i}.png')
        plt.close()
    
def visualize_rgb(images, name=''):
    for i in range(images.shape[0]):
        rgb = images[i].numpy()[1:4].transpose(1,2,0)
        rgb = np.clip(rgb, 0, 3000)
        rgb = rgb/3000
        plt.figure()
        plt.imshow(rgb)
        plt.colorbar()
        plt.savefig(f'output/32MRE_rgb_{name}_{i}.png')
        plt.close()

def visualize_all_bands(image):
    lower = [500, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 1000, 500, 500]
    upper = [2000, 2000, 2000, 2000, 2000, 5000, 5000, 5000, 5000, 5000, 3000, 2000]
    for i in range(image.shape[0]):
        plt.figure()
        data = image[i].clip(lower[i], upper[i])
        plt.imshow(data, cmap='gray')
        plt.colorbar()
        plt.savefig(f'output/32MRE_band_{i}.png')
        plt.close()


def buffer_mask(scl, buffer_radius:int=7):
    y, x = torch.meshgrid(torch.arange(-buffer_radius, buffer_radius+1), torch.arange(-buffer_radius, buffer_radius+1), indexing='ij')
    kernel = ((x**2 + y**2) <= buffer_radius**2).float().unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(scl.device)
    scl = scl.to(torch.float32)
    dilated = F.conv2d(scl, kernel, padding=buffer_radius)
    return (dilated > 0).float()


@hydra.main(config_name='patch_inference_debug', config_path='config', version_base="1.2")
def main(cfg: DictConfig):
    from kornia.enhance import normalize
    from datasets.transforms import MEAN, STD
    import matplotlib.pyplot as plt
    from datasets.zarr_dataset_deploy import S2DatasetStream
    
    model = get_model(cfg.model, cfg.run_id, cfg.wandb_project, cfg.bias_correction_column, cfg.model_alias)
    model = model.to('cuda')
    dataset = S2DatasetStream(
        metadata_file='~/data/gvs/deploy/deploy_s2_items_2024_part7.parquet',
        h5_dir='~/flash/data/gvs/deploy/inference_2020',
        tile_id='32MPD',
        prediction_dir='~/data/gvs/deploy/predictions_2024/11UMP_GTiff',
        input_lat_lon=True,
        patch_size=544,
        border=16,
    )
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=collate_batch)
    # for data in dataloader:
    #     print(data[0].shape)
    full_pred = []
    full_scl = []
    full_lc = []
    # for i, slices in dataset.patch_coords_dict.items():
    #     _, y_topleft, x_topleft = slices
    #     if y_topleft >= 0 and x_topleft >= 0:
    #         idx = i
    #         break
    idx = 0
    for i in range(idx, idx+2):
        image, scl, latlon = dataset[i]
        # image = image[5:6]
        # scl = scl[5:6]
        # latlon = latlon[5:6]
        # visualize all bands
        # visualize_all_bands(image[0].numpy())
        visualize_rgb(image, name=f'raw_batch_{i}')
            
        image = image.to('cuda')
        scl = scl.to('cuda').float()
        latlon = latlon.to('cuda').float()
        # visualize_rgb((image.float()*(~scl_mask_buffered)).cpu(), name=f'bufferred_mask_batch_{i}')        
        # raw prediction
        image_normalized = normalize(image.float(), MEAN, STD)
        raw_input = torch.cat([image_normalized, latlon], dim=1)
        with torch.no_grad():
            pred_raw = model(raw_input.float())
        lc = pred_raw[:, 303:].argmax(dim=1, keepdim=True)
        pred_raw = pred_raw[:, :303]
        full_lc.append(lc)
        full_pred.append(pred_raw)
        full_scl.append(scl)
        
    lc = torch.cat(full_lc, dim=0)
    visualize_rh98(lc.squeeze(1).cpu(), name='lc')
    lc = lc.float()
    # SCL mask
    scl = torch.cat(full_scl, dim=0)
    scl_mask = torch.isin(scl, dataset.scl_exclude_labels)
    scl_cloud_mask = torch.isin(scl, dataset.scl_cloud)
    scl_cloud_mask_buffered = buffer_mask(scl_cloud_mask, dataset.scl_cloud_mask_buffer)
    scl_mask_buffered = torch.logical_or(scl_mask, scl_cloud_mask_buffered)
    scl_mask = torch.logical_or(scl_mask, scl_cloud_mask)
    nodata_mask = scl == 0
    scl[nodata_mask] = float('nan')
    lc[nodata_mask] = float('nan')
    nodata_mask = nodata_mask.repeat(1, 303, 1, 1)
    water_mask_scl = torch.mode(scl, dim=0).values == dataset.scl_water
    
    # LC mask
    built_up_mask = torch.mode(lc, dim=0).values == dataset.esa_built_up
    water_mask_esa = torch.mode(lc, dim=0).values == dataset.esa_water
    lc_mask = lc == dataset.esa_snow
    
    pred_raw = torch.cat(full_pred, dim=0)
    pred_raw[nodata_mask] = float('nan') # mask the prediction when input is nodata
    pred_raw_masked = torch.where(scl_mask | lc_mask, torch.nan, pred_raw)
    pred_raw_copy = pred_raw_masked.cpu()
    visualize_rh98(pred_raw_copy[:, 295], name='raw')
    pred_raw_masked, _ = torch.nanmedian(pred_raw_masked, dim=0)
    pred_raw_masked = torch.where(water_mask_scl | built_up_mask | water_mask_esa, torch.nan, pred_raw_masked)
    pred_raw_masked.mul_(10).round_()  # In-place operations
    # pred_raw_masked = torch.nan_to_num(pred_raw_masked, nan=dataset.nodata_value)            
    # Move to CPU once and do all numpy operations together
    pred_raw_masked = pred_raw_masked.cpu().numpy()#.astype(dataset.output_dtype)   
    rh98_raw = pred_raw_masked[295]
    plt.figure()
    plt.imshow(rh98_raw)
    plt.colorbar()
    plt.savefig('output/32MRE_rh98_raw_lc_masked.png', dpi=300)
    
    pred_raw_buffered = torch.where(scl_mask_buffered | lc_mask, torch.nan, pred_raw)
    pred_raw_buffered_copy = pred_raw_buffered.cpu()
    visualize_rh98(pred_raw_buffered_copy[:, 295], name='buffered_mask')
    pred_raw_buffered, _ = torch.nanmedian(pred_raw_buffered, dim=0)
    pred_raw_buffered = torch.where(water_mask_scl | built_up_mask | water_mask_esa, torch.nan, pred_raw_buffered)
    pred_raw_buffered.mul_(10).round_()  # In-place operations
    # pred_raw_buffered = torch.nan_to_num(pred_raw_buffered, nan=dataset.nodata_value)            
    # Move to CPU once and do all numpy operations together
    pred_raw_buffered = pred_raw_buffered.cpu().numpy()#.astype(dataset.output_dtype)   
    rh98_buffered = pred_raw_buffered[295]
    plt.figure()
    plt.imshow(rh98_buffered)
    plt.colorbar()
    plt.savefig('output/32MRE_rh98_buffered_mask.png', dpi=300)
    
    diff = rh98_buffered - rh98_raw
    plt.figure()
    plt.imshow(diff)
    plt.colorbar()
    plt.savefig('output/32MRE_rh98_diff_buffered_mask.png')
    
    # # # imputed prediction
    # scl_cloud_mask = torch.isin(scl, torch.tensor([8,9], device='cuda'))
    # mean_input = torch.from_numpy(dataset.mean[:10, :, None, None]).to('cuda')
    # masked_mean_input = mean_input * scl_cloud_mask
    # image = image.float()
    # image_imputed = torch.where(scl_cloud_mask, 0, image)
    # image_imputed = image_imputed + masked_mean_input
    # image_imputed_copy = image_imputed.cpu()
    # for i in range(image_imputed_copy.shape[0]):
    #     visualize_rgb(i, image_imputed_copy[i], name='imputed')
    # image_imputed_normalized = normalize(image_imputed.float(), MEAN, STD)
    # imputed_input = torch.cat([image_imputed_normalized, latlon], dim=1)
    # with torch.no_grad():
    #     pred_impute = model(imputed_input.float())
    # pred_impute = pred_impute[:, :303]
    # pred_impute_copy = pred_impute.cpu()
    # for i in range(pred_impute.shape[0]):
    #     visualize_rh98(i, pred_impute_copy[i, 295], name='impute')
    # for i in range(image_imputed_copy.shape[0]):
    #     visualize_rh98(i, pred_impute_copy[i, 295] - pred_raw_copy[i, 295], name='imputed_delta')
    # pred_impute[nodata_mask] = float('nan') # mask the prediction when input is nodata
    # pred_impute = torch.where(scl_mask, torch.nan, pred_impute)
    # pred_impute, _ = torch.nanmedian(pred_impute, dim=0)
    # pred_impute = torch.where(water_mask_scl, torch.nan, pred_impute)
    # pred_impute.mul_(10).round_()  # In-place operations
    # pred_impute = torch.nan_to_num(pred_impute, nan=dataset.nodata_value)            
    # # Move to CPU once and do all numpy operations together
    # pred_impute = pred_impute.cpu().numpy().astype(dataset.output_dtype)   
    # rh98_impute = pred_impute[295]
    # rh98_delta_impute = rh98_impute - rh98_raw
    
    
    # plt.figure()
    # plt.imshow(rh98_raw)
    # plt.colorbar()
    # plt.savefig('output/32MRE_rh98_raw.png')

    # plt.figure()
    # plt.imshow(rh98_impute)
    # plt.colorbar()
    # plt.savefig('output/32MRE_rh98_impute.png')

    # plt.figure()
    # plt.imshow(rh98_delta_impute)
    # plt.colorbar()
    # plt.savefig('output/32MRE_rh98_delta_impute.png')
    
    # patch_without_blob = np.load('test_without_blob.npy')
    # patch_without_blob = torch.from_numpy(patch_without_blob).float()

    # patch_with_blob = np.load('test_with_blob.npy')
    # patch_with_blob = torch.from_numpy(patch_with_blob).float()
    
    # patch_impute = np.load('test_impute.npy')
    # patch_impute = torch.from_numpy(patch_impute).float()
    # pred_with_blob = model(patch_with_blob)
    # rh98_with_blob = pred_with_blob[0,295].detach().numpy()
    # pred_without_blob = model(patch_without_blob)
    # rh98_without_blob = pred_without_blob[0,295].detach().numpy()
    # rh98_delta = rh98_with_blob - rh98_without_blob
    # pred_impute = model(patch_impute)   
    # rh98_impute = pred_impute[0,295].detach().numpy()
    # rh98_delta_impute = rh98_impute - rh98_without_blob

    # plt.figure()
    # plt.imshow(rh98_without_blob)
    # plt.colorbar()
    # plt.savefig('output/rh98_without_blob.png')

    # plt.figure()
    # plt.imshow(rh98_with_blob)
    # plt.colorbar()
    # plt.savefig('output/rh98_with_blob.png')

    # vmin, vmax = rh98_delta.min(), rh98_delta.max()
    # plt.figure()
    # plt.imshow(rh98_delta, vmin=vmin, vmax=vmax)
    # plt.colorbar()
    # plt.savefig('output/rh98_delta.png')

    # plt.figure()
    # plt.imshow(rh98_impute)
    # plt.colorbar()
    # plt.savefig('output/rh98_impute.png')
    
    # plt.figure()
    # plt.imshow(rh98_delta_impute, vmin=vmin, vmax=vmax)
    # plt.colorbar()
    # plt.savefig('output/rh98_delta_impute.png')

    import ipdb; ipdb.set_trace()

    # image = get_data(cfg.data.metadata_file, cfg.data.tile_id, cfg.data.n_iamges_per_tile)
    # model = model.to('cuda')
    # data = image.data.rechunk((10, 13,512,512))
    # res = data.map_overlap(batch_inference, depth={2:cfg.data.border, 3:cfg.data.border}, boundary='reflect', meta=data)
    # res.compute()

if __name__ == '__main__':
    main()