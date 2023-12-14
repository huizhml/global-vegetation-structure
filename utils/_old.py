
    # def download_track(self, gedi_track_df, zone):
    #     parquet_id = gedi_track_df.parquet_id[0]
    #     s2_parquet = self.gediFolder / zone / f'{parquet_id}_s2.parquet'
    #     s2_h5 = self.gediFolder / zone / f'{parquet_id}.h5'
    #     if s2_parquet.exists() and s2_h5.exists():
    #         # print(f'{parquet_id}_s2.parquet & {parquet_id}.h5 already exists')
    #         return
    #     # print(f'Downloading {parquet_id}_s2.parquet & {parquet_id}.h5')
    #     # gedi_track_df = gedi_track_df.compute()
    #     gedi_track_df['.geo'] = gedi_track_df['.geo'].apply(
    #         lambda x: shape(json.loads(x)))
    #     gedi_track_df = gpd.GeoDataFrame(gedi_track_df,
    #                                      geometry=gedi_track_df['.geo'],
    #                                      crs='epsg:4326')
    #     gedi_track_df.drop('.geo', axis=1, inplace=True)
    #     bounds = MultiPoint(gedi_track_df['geometry']).bounds
    #     gediDate = np.datetime64(gedi_track_df['date'][0], 'ns')
    #     items = self.queryS2(bounds, gediDate, self.queryDaysRange)

    #     data_rows = []
    #     for item in items:
    #         properties = item.properties.copy()
    #         properties['s2_id'] = item.id
    #         properties['geometry'] = shape(item.geometry)
    #         # properties['scl'] = item.assets['SCL']
    #         data_rows.append(properties)

    #     s2_df = gpd.GeoDataFrame(data_rows,
    #                              geometry='geometry',
    #                              crs="epsg:4326")

    #     s2_df = s2_df[['s2_id', 'datetime', 'geometry', 's2:mgrs_tile']]  #
    #     s2_df.rename(columns={'datetime': 's2_datetime'}, inplace=True)
    #     s2_df['s2_datetime'] = pd.to_datetime(
    #         s2_df['s2_datetime']).dt.tz_localize(None)
    #     s2_df['delta_day'] = (np.abs(s2_df['s2_datetime'] - gediDate) /
    #                           np.timedelta64(1, 'D')).astype(int)

    #     gedi_query_df = gedi_track_df[[
    #         'shot_number', 'parquet_id', 'leaf_on_doy', 'leaf_off_doy',
    #         'geometry'
    #     ]]

    #     # except:
    #     # print('finish', parquet_id)
    #     joined = gpd.sjoin(gedi_query_df,
    #                        s2_df,
    #                        how='left',
    #                        predicate='within')

    #     best_s2_ids = joined.groupby('shot_number').apply(self.get_best_tile)


    #     gedi_track_df = pd.merge(gedi_track_df, best_s2_ids, on='shot_number')
    #     gedi_track_df.to_parquet(s2_parquet)

    #     # # Save Dask array to HDF5 file
    #     # with h5py.File(s2_h5, 'w') as h5file:
    #     #     dset = h5file.create_dataset('s2', shape=best_patches.shape, dtype=best_patches.dtype)
    #     #     da.store(best_patches, dset)

    #     # print(f'time taken for {len(gedi_track_df)} points: {time.time() - t0}')

    # def download_zone(self, zone):
    #     zoneFolder = self.gediFolder / zone
    #     gediDf = dd.read_parquet(zoneFolder / f'*.parquet')

    #     if self.debug:
    #         group_keys = gediDf.parquet_id.unique().compute()
    #         # res = gediDf.groupby('parquet_id').apply(self.download_track, zone, meta=object).compute()
    #         test_group_key = group_keys.iloc[
    #             1]  #'2019240043440_O04011_04_T00686' #group_keys.iloc[1]
    #         # test_point = gediDf[gediDf['shot_number'] == '000e0000000000007df7'].compute()
    #         test_group = gediDf[gediDf.parquet_id == test_group_key].compute()
    #         self.download_track(test_group, zone)

    #     else:
    #         gediDf.groupby('parquet_id').apply(self.download_track,
    #                                            zone,
    #                                            meta=object).compute()
