import bisect
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from collections import defaultdict
import zarr
import xarray as xr
import numpy as np
import random
import xbatcher
import multiprocessing as mp
from queue import Empty
from typing import List, Dict, Tuple

class MultiZarrGroupDataset(Dataset):
    def __init__(self, zarr_dir, data_array='image', label_array='rhs'):
        """
        Args:
            zarr_paths (list): List of paths to Zarr stores.
            data_array (str): Name of the data array within each group.
            label_array (str): Name of the label array within each group.
        """
        zarr_dir = Path(zarr_dir).expanduser()
        zarr_paths = []
        for year in range(2019, 2023):
            zarr_paths.append(f'{zarr_dir}/year_{year}.zarr')
        self.zarr_paths = zarr_dir/'year_2019.zarr'
        self.data_array = data_array
        self.label_array = label_array
        self._initialized = False
        
        
    def _initialize(self):
        if self._initialized:
            return
        self.store = zarr.open(self.zarr_paths, mode='r')
        # Store groups and their cumulative lengths
        self.groups = []  # List of tuples: (group, num_samples)
        self.cumulative_lengths = []
        total_samples = 0
        
        # Open each Zarr store and process its groups
        # for path in zarr_paths:
        #     root = zarr.open(path, mode='r')
            # Get sorted group names to ensure consistent order across stores
        group_names = sorted(self.store.group_keys())
        
        for name in group_names:
            group = self.store[name]
            # # Validate required arrays exist
            # if self.data_array not in group or self.label_array not in group:
            #     raise KeyError(f"Missing '{data_array}' or '{label_array}' in {path}/{name}")
            
            data = group[self.data_array]
            num_samples = data.shape[0]
            
            self.groups.append((name, num_samples))
            total_samples += num_samples
            self.cumulative_lengths.append(total_samples)
    
        if not self.cumulative_lengths:
            raise ValueError("No valid groups found across all Zarr stores")
        
        self._initialized = True
    
    def __len__(self):
        if not self._initialized:
            self._initialize()
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        self._initialize()
        # Find which group contains the index
        group_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        
        # Calculate local index within the group
        if group_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[group_idx - 1]
        
        # Fetch data and labels
        
        group_name, _ = self.groups[group_idx]
        data = self.store[f'{group_name}/{self.data_array}'][local_idx]
        label = self.store[f'{group_name}/{self.label_array}'][local_idx]
        
        return data, label
    

def zarrdataset_worker_init_fn(worker_id):
    """ZarrDataset multithread workers initialization function.
    """

    worker_info = torch.utils.data.get_worker_info()
    w_sel = slice(worker_id, None, worker_info.num_workers)

    dataset_obj = worker_info.dataset

    # Reset the random number generators in each worker.
    torch_seed = torch.initial_seed()
    random.seed(torch_seed)
    np.random.seed(torch_seed % (2**32 - 1))

    dataset_obj._worker_sel = w_sel
    dataset_obj._worker_id = worker_id
    dataset_obj._num_workers = worker_info.num_workers

class MultiZarrIterableDataset(IterableDataset):
    def __init__(self, zarr_dir, data_array='image', label_array='rhs'):
        """
        Args:
            zarr_paths (list): List of paths to Zarr stores.
            data_array (str): Name of the data array within each group.
            label_array (str): Name of the label array within each group.
        """
        zarr_dir = Path(zarr_dir).expanduser()
        self.zarr_paths = []
        for year in range(2019, 2023):
            self.zarr_paths.append(f'{zarr_dir}/year_{year}.zarr')
        self.data_array = data_array
        self.label_array = label_array
        self._initialized = False
        

    
    def _initialize(self):
        # Precompute all (store_path, group_name) pairs
        self.all_groups = []
        for path in self.zarr_paths:
            store = zarr.open(path, mode='r')
            group_names = sorted(store.group_keys())  # Ensure consistent order
            for name in group_names:
                self.all_groups.append((path, name))
        
        # Precompute total samples (optional, for __len__)
        self.total_samples = 0
        for path, group_name in self.all_groups:
            store = zarr.open(path, mode='r')
            group = store[group_name]
            self.total_samples += group[self.data_array].shape[0]
        self._initialized = True
    
    def __iter__(self):
        self._initialize()
        worker_info = get_worker_info()
        
        if worker_info is None:  # Single-process mode
            groups = self.all_groups
        else:  # Split work across workers
            per_worker = len(self.all_groups) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.all_groups)
            groups = self.all_groups[start:end]
        
        # Group by store path to minimize store opens
        store_groups = defaultdict(list)
        for path, group_name in groups:
            store_groups[path].append(group_name)
        
        # Stream data from assigned groups
        for path, group_names in store_groups.items():
            store = zarr.open(path, mode='r')
            for group_name in group_names:
                group = store[group_name]
                data = group[self.data_array]
                labels = group[self.label_array]
                for i in range(data.shape[0]):
                    yield (
                        data[i],  # Zero-copy to tensor
                        labels[i]
                    )

class XBatcherPyTorchDataset(Dataset):
    def __init__(self, batch_generator: xbatcher.BatchGenerator):
        self.bgen = batch_generator

    def __len__(self):
        return len(self.bgen)

    def __getitem__(self, idx):
        # load before stacking
        batch = self.bgen[idx].load()
        return batch['image'].data, batch['rhs'].data





class ParallelZarrLoader(IterableDataset):
    def __init__(self,
                 zarr_paths: List[str],
                 global_index: List[Tuple[str, str, int]],
                 data_key: str = "image",
                 label_key: str = "rhs",
                 cache_size: int = 1024,
                 num_workers: int = 4,
                 seed: int = None):
        """
        Multi-process Zarr loader with shared sample pool
        
        Args:
            zarr_paths: List of paths to Zarr stores
            data_key: Key for data arrays in groups
            label_key: Key for label arrays in groups
            cache_size: Total number of samples to keep in memory
            num_workers: Number of worker processes
            seed: Random seed for reproducibility
        """
        self.zarr_paths = zarr_paths
        self.data_key = data_key
        self.label_key = label_key
        self.cache_size = cache_size
        self.num_workers = num_workers
        self.seed = seed or random.randint(0, 2**32-1)
        
        # Build global index without opening stores
        self.index = global_index
        self.total_samples = sum(g[2] for g in self.index)
        
        # Create shared structures
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue(maxsize=cache_size)
        self.stop_event = mp.Event()
        
        # Start workers
        self.workers = [
            mp.Process(target=self._worker_main,
                       args=(self.task_queue,
                             self.result_queue,
                             self.stop_event,
                             self.seed + i))
            for i in range(num_workers)
        ]
        for w in self.workers:
            w.start()
            
        # Pre-fill task queue
        self._fill_task_queue()



    def _fill_task_queue(self):
        """Distribute sampling tasks across workers"""
        weights = [samples for _, _, samples in self.index]
        total_weight = sum(weights)
        
        for _ in range(self.cache_size * 2):  # Keep buffer of future tasks
            # Weighted random selection of dataset
            idx = random.choices(range(len(self.index)), weights=weights)[0]
            path, group, total = self.index[idx]
            
            # Generate random sample index within dataset
            sample_idx = random.randint(0, total - 1)
            self.task_queue.put((path, group, sample_idx))

    def _worker_main(self, task_queue, result_queue, stop_event, seed):
        """Worker process main loop"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Keep open stores per worker to reduce open/close overhead
        open_stores: Dict[str, zarr.Group] = {}
        
        try:
            while not stop_event.is_set():
                try:
                    path, group_name, sample_idx = task_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Open store if not already open
                if path not in open_stores:
                    open_stores[path] = zarr.open(path, mode='r')
                
                # Access data
                group = open_stores[path][group_name]
                data = group[self.data_key][sample_idx]
                label = group[self.label_key][sample_idx]
                
                # Convert to tensor and put in result queue
                result_queue.put((
                    torch.as_tensor(data),
                    torch.as_tensor(label)
                ))
        finally:
            # Cleanup open stores
            for store in open_stores.values():
                store.store.close()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                return self.result_queue.get(timeout=1)
            except Empty:
                if self.stop_event.is_set():
                    raise StopIteration
                continue

    def __len__(self):
        return self.total_samples

    def __del__(self):
        """Cleanup workers when done"""
        self.stop_event.set()
        for w in self.workers:
            if w.is_alive():
                w.join(timeout=1)
            if w.is_alive():
                w.terminate()


if __name__ == '__main__':
    # Example usage
    from tqdm import tqdm
    import dask
    dask.config.set(scheduler="threads", num_workers=8)
    zarr_dir = '/home/ksb781/data/GVS/GEDI_S2_zarr'
    zarr_paths = []
    for year in range(2019, 2023):
        zarr_paths.append(f'{zarr_dir}/year_{year}.zarr')


    index = []
    for path in zarr_paths:
        # Use zarr's metadata-only mode to avoid opening full store
        store = zarr.open(path, mode='r')
        for group_name in sorted(store.group_keys()):
            group = store[group_name]
            num_samples = group['image'].shape[0]  # Metadata access only
            index.append((path, group_name, num_samples))


    dataset = ParallelZarrLoader(zarr_paths, num_workers=4, global_index=index)

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            num_workers=4,  # Separate from dataset workers
            prefetch_factor=2
        )
    # ############################ Using xbatcher ############################
    # ### Very slow
    # ds = xr.open_zarr(zarr_dir + '/year_2019.zarr', group='35H', chunks={})
    # bgen = xbatcher.BatchGenerator(
    #     ds=ds,
    #     input_dims={'shot_number': 1, 'band': 14, 'y':15, 'x':15, 'rh': 101},
    #     # batch_dims={'shot_number': 4096, 'band': 14, 'y':15, 'x':15, 'rh': 101},    
    # )
    # dataset = XBatcherPyTorchDataset(bgen)
    # dataloader = DataLoader(dataset, batch_size=4096,
    #     shuffle=True,
    #     num_workers=32,
    #     prefetch_factor=3,
    #     persistent_workers=True,
    #     multiprocessing_context="forkserver")
    # for batch in tqdm(dataloader):
    #     img, label = batch
    #     print(img.shape)
    # ############################ Using map-style dataset ############################
    # # not working
    # # dataset = MultiZarrGroupDataset(zarr_dir)
    # # loader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=4)

    # ############################ Using iterable dataset ############################
    # dataset = MultiZarrIterableDataset(zarr_dir)
    # dataloader = DataLoader(dataset, batch_size=1024, num_workers=4,  prefetch_factor=2, worker_init_fn=zarrdataset_worker_init_fn)
    # # for img, label in tqdm(dataset):
    # #     print(img.shape)


    for batch in tqdm(dataloader):
        img, label = batch

    # ############################ Using ZarrDataset ############################
    # import zarrdataset as zds
    # my_dataset = zds.ZarrDataset(
    #     dict(
    #         modality="images",
    #         filenames=[zarr_dir + '/year_2019.zarr'],
    #         source_axes="TCYX",
    #         data_group="35H"
    #     )
    # )
    # my_dataloader = DataLoader(my_dataset,
    #                        batch_size=32,
    #                        num_workers=0,
    #                        worker_init_fn=zds.zarrdataset_worker_init_fn)

    # for x in my_dataloader:
    #     print(x.shape)


    # for batch in tqdm(loader):
    #     data, label = batch
    #     print(data.shape, label.shape)
    # print(len(dataset))  # Total number of samples
    # print(dataset[0])  # Fetch first sample
    # print(dataset[-1])  # Fetch last sample
    # print(dataset[100])  # Fetch sample at index 100
    # print(dataset[1000])  # Fetch sample at index 1000