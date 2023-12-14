from pathlib import Path
import zarr
import xarray as xr


class S2GEDIVis:

    def __init__(self, data_folder='data/GEDI', zone='01G', year=2019):
        self.data_folder = Path.home() / data_folder


    def get_zone_size(self, zone):
        store = zarr.open(self.data_folder / f'{zone}.zarr', mode='r')
        lens = []
        for key, group in store.groups():
            lens.append(group['time'].shape[0])
        return sum(lens)
    

if __name__ == '__main__':
    vis = S2GEDIVis()
    size = vis.get_zone_size('01G')
    print(size)
