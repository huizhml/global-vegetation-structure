#%%
rh_dtype = {f'rh{i}': 'float32' for i in range(101)}
gedi_attr_dtype = {
    # 'system:index': 'object',
    'beam': 'uint16', # can tell if it's full or half
    'delta_time': 'float64',
    'digital_elevation_model': 'float32',
    'digital_elevation_model_srtm': 'float32',
    'elev_highestreturn': 'float32',
    'elev_lowestmode': 'float32',
    'elevation_bias_flag': 'uint8',
    'energy_total': 'float32',
    'landsat_treecover': 'float64',
    'landsat_water_persistence': 'uint8',
    'leaf_off_doy': 'int16',
    'leaf_off_flag': 'uint8',
    'leaf_on_doy': 'int16',
    'leaf_on_cycle': 'uint8',
    'modis_nonvegetated': 'float64',
    'modis_nonvegetated_sd': 'float64',
    'modis_treecover': 'float64',
    'modis_treecover_sd': 'float64',
    'num_detectedmodes': 'uint8',
    'pft_class': 'uint8',
    'region_class': 'uint8',
    'selected_algorithm': 'uint8',
    'selected_mode': 'uint8',
    'selected_mode_flag': 'uint8',
    'sensitivity': 'float32',
    # 'shot_number': 'uint64',
    'solar_azimuth': 'float32',
    'solar_elevation': 'float32',
    'surface_flag': 'uint8',
    'urban_focal_window_size': 'uint8',
    'urban_proportion': 'uint8',
}
latlon_dtype = {
    'lat_highestreturn': 'float64',
    'lon_highestreturn': 'float64',
}
dtypes = {'shot_number': 'uint64', **gedi_attr_dtype, **latlon_dtype, **rh_dtype}
# %%
STAC_ITEM_KEYS = [
    'id', 'geometry', 'bbox', 'assets'
]
S2_ITEM_PROPS = ['datetime', 
    # 'platform', 
    'proj:epsg', 
    # 'instruments', 
    # 's2:mgrs_tile', 
    # 'constellation', 
    # 's2:granule_id', 
    # 'eo:cloud_cover', 
    # 's2:datatake_id', 
    # 's2:product_uri', 
    # 's2:datastrip_id', 
    # 's2:product_type', 
    # 'sat:orbit_state', 
    # 's2:datatake_type', 
    # 's2:generation_time', 
    # 'sat:relative_orbit', 
    # 's2:water_percentage', 
    # 's2:mean_solar_zenith', 
    # 's2:mean_solar_azimuth', 
    # 's2:processing_baseline', 
    # 's2:snow_ice_percentage', 
    # 's2:vegetation_percentage', 
    # 's2:thin_cirrus_percentage', 
    # 's2:cloud_shadow_percentage', 
    # 's2:nodata_pixel_percentage', 
    # 's2:unclassified_percentage', 
    # 's2:dark_features_percentage', 
    # 's2:not_vegetated_percentage', 
    # 's2:degraded_msi_data_percentage',
    # 's2:high_proba_clouds_percentage', 
    # 's2:reflectance_conversion_factor', 
    # 's2:medium_proba_clouds_percentage', 
    # 's2:saturated_defective_pixel_percentage'
    ]
WC_ITEM_PROPS = [
    # 'created', 
    # 'mission',
    # 'datetime', 
    # 'platform', 
    # 'grid:code', 
    # 'proj:epsg', 
    # 'description', 
    # 'instruments', 
    # 'end_datetime', 
    # 'start_datetime', 
    'esa_worldcover:product_tile', 
    # 'esa_worldcover:product_version'
    ]
# for test
bounds = (454103.12321006873, 4824679.480536657, 454243.12321006873, 4824819.480536657)
epsg = 32655