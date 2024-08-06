import lzma
import pickle
import time
from contextlib import contextmanager
from multiprocessing import Queue, Process, Event, Semaphore, Pool
from pathlib import Path
from typing import Iterator, List, Tuple, Union

import h5py
import numpy as np
from torch.utils.data import IterableDataset
from tqdm import tqdm


@contextmanager
def get_semaphore(semaphore: Semaphore, block: bool = True, timeout: float = None): # type: ignore
    held = semaphore.acquire(block=block, timeout=timeout)
    try:
        yield held
    finally:
        if held:
            semaphore.release()


@contextmanager
def open_files(files: List[str]):
    open_files = [h5py.File(file, mode="r") for file in files]
    try:
        yield open_files
    finally:
        [file.close() for file in open_files]


def h5_tree(group):  # memory should be 2*n*int32
    # creates a tree with the number of samples of datasets at the leafs,
    # each node gives back its count and the subcounts

    tree: List[Tuple[List, int]] = []
    nrows = group.get_num_objs()
    for i in range(nrows):
        group_type = group.get_objtype_by_idx(i)
        group_name = group.get_objname_by_idx(i)
        if group_type == 1:
            if group_name == b"latlon":
                latlon = h5py.h5d.open(group, group_name)
                num_samples = latlon.shape[0]
                latlon.close()
                # we hit the leaf
                return num_samples, None
            continue

        # to the next node
        group_ = h5py.h5g.open(group, group_name)
        tree.append(h5_tree(group_))
        group_.close()
    return sum([node[0] for node in tree]), tree


def get_node_key(node, idx, group):
    # NOTE: if slow one could use low-level API to retrieve sample directly
    if node is None:
        return "", idx

    # find out which subgroup to use
    offsets = np.cumsum([n[0] for n in node])
    subgroup_idx = np.argmax(offsets > idx)
    # get relevant data for next branch
    _, node_ = node[subgroup_idx]
    group_name = group.get_objname_by_idx(subgroup_idx)

    reset_idx = offsets[subgroup_idx - 1] if subgroup_idx != 0 else 0
    idx_ = idx - reset_idx
    group_ = h5py.h5g.open(group, group_name)

    group_name_node, idx_node = get_node_key(node_, idx_, group_)
    group_.close()
    return group_name.decode('utf-8') + "/" + group_name_node, idx_node


# Worker process to fill the queue with random samples
def worker(file_list: List[str], queue_list: List[Queue], semaphore_list: List[Semaphore], # type: ignore
           done_event: Event, seed: int, tree_list: List[List], num_tree_samples_list: List[int]): # type: ignore
    rs = np.random.RandomState(seed=seed)
    with open_files(file_list) as f_list:
        while not done_event.is_set():  # only until we get the done event
            for f, queue, semaphore, tree, num_tree_samples in zip(
                    f_list, queue_list, semaphore_list, tree_list, num_tree_samples_list):
                with get_semaphore(semaphore, False) as success:  # don't block the queue but wait here instead
                    if success:
                        if num_tree_samples == 0:
                            continue
                        r_idx = rs.randint(num_tree_samples)
                        k, idx = get_node_key(tree, r_idx, f.id)

                        sample = f[k]

                        image = sample["image"][idx]
                        wc = image[13, 7, 7:8]
                        image = image[:12]
                        data = (
                            image,
                            sample["rhs"][idx],
                            wc,
                            sample["slope"][idx][7, 7:8],
                            sample["latlon"][idx]
                        )
                        # NOTE: could apply transforms here

                        queue.put(data)


def getting_index_tree(args):
    file, index_file = args
    # creation/loading  parallelized
    if index_file.exists():
        with open(index_file, "rb") as f:
            compressed_pickle = f.read()
        depressed_pickle = lzma.decompress(compressed_pickle)
        num_tree_samples, tree = pickle.loads(depressed_pickle)
    else:
        with h5py.File(file, mode="r") as f:
            # creating the index structure (needed because of string keys)
            num_tree_samples, tree = h5_tree(f.id)
            with lzma.open(index_file, "wb") as f:
                pickle.dump((num_tree_samples, tree), f)
    return num_tree_samples, tree


def find_partitions_args(num_samples_list: List[int], target_number: int):
    num_samples_list = np.array(num_samples_list, dtype=np.int32)
    index = np.arange(len(num_samples_list))
    partitions = []
    while target_number > 0:
        total = np.sum(num_samples_list)
        p_size = total / target_number

        num_samples_list = np.array(num_samples_list)
        res = p_size - num_samples_list
        cond = res <= 0
        partitions.extend([[idx] for idx in index[cond]])  # these are the overfull ones
        if len(partitions) == target_number:
            return partitions

        target_number = target_number - len(partitions)
        index = index[~cond]
        num_samples_list = num_samples_list[~cond]

        if len(partitions) < target_number or not cond.any():  # no possiblity or need to split further we can just greedily select partitions
            num_samples_list = num_samples_list.tolist()
            index = index.tolist()
            for _ in range(target_number):
                current_partition = []
                current_sum = 0
                res = (p_size - np.array(num_samples_list)).tolist()

                while len(index) > 0 and current_sum < p_size:
                    idx = np.argmin(res)
                    current_partition.append(index[idx])
                    current_sum += num_samples_list[idx]
                    del index[idx], num_samples_list[idx], res[idx]

                if len(current_partition) > 0:
                    partitions.append(current_partition)
                if len(index) == 0:
                    break
            return partitions


class IterH5Dataset(IterableDataset):
    """A IterableDataloader that provides random samples (with replacement) from
    a list of hdf5 files. Will balance the samples coming from each file."""

    def __init__(self, file_list: List[str], num_samples: int, cache_size: int, index_folder: Union[str, Path],
                 num_jobs: int = 1, seed: int = None):
        """
        :param file_list: lists to load from equally
        :param num_samples: number of samples per iterator object
        :param cache_size: maximum number of samples to precache
        :param index_folder: folder where all the index trees are stored
        :param num_jobs: how many processes be created (default: 1)
        :param seed: random seed to go through data (if not set, it will use a random seed)
        """
        self.file_list = file_list
        self.num_samples = num_samples
        self.started = False

        assert num_jobs > 0, "need at least one job per file"

        self.processes = []
        self.queues = []
        self.done_event: Event = Event() # type: ignore
        if seed is None:
            # use current time as seed
            seed = time.time()
        self.rs = np.random.RandomState(seed)

        self.index_folder = Path(index_folder)
        self.index_folder.mkdir(exist_ok=True)
        self.num_tree_samples_list = []

        # getting the index trees
        print("Build tree index structure")
        index_trees = []
        with Pool(processes=num_jobs) as pool:
            results = pool.imap(
                getting_index_tree, [(file, self.index_folder / (Path(file).stem + ".xz")) for file in file_list]
            )

            pbar = tqdm(total=len(file_list))
            for num_tree_samples, tree in results:
                index_trees.append(tree)
                self.num_tree_samples_list.append(num_tree_samples)
                pbar.update()
            pbar.close()
        self.p_queues = np.array(self.num_tree_samples_list) / np.sum(self.num_tree_samples_list)

        print("Finding best partitions")
        partition_args = find_partitions_args(self.num_tree_samples_list, num_jobs)
        jobs_left = num_jobs - len(partition_args)
        assert jobs_left >= 0, "too many jobs assigned to partitions"
        jobs_per_partition = [int(self.p_queues[p_arg].sum() * jobs_left) + 1 for p_arg in partition_args]

        file_list = np.array(file_list)
        index_trees = np.array(index_trees, dtype=object)
        self.num_tree_samples_list = np.array(self.num_tree_samples_list)
        q_list = np.array([Queue() for _ in range(len(file_list))])
        s_list = np.array([Semaphore(max(1, int(cache_size * self.p_queues[i]))) for i in range(len(file_list))])

        print("Prepare jobs")
        p_idx = 0
        for i, p_args in enumerate(partition_args):
            for j in range(jobs_per_partition[i]):
                p = Process(target=worker,
                            args=(file_list[p_args], q_list[p_args], s_list[p_args], self.done_event,
                                  seed + p_idx, index_trees[p_args], self.num_tree_samples_list[p_args]))
                p_idx += 1
                self.processes.append(p)
        self.queues = q_list
        self.semaphores = s_list
        self.p_queues = np.array(self.num_tree_samples_list) / np.sum(self.num_tree_samples_list)

    def __len__(self):
        return self.num_samples

    def __iter__(self) -> Iterator:
        # start worker_processes
        if not self.started:
            [p.start() for p in self.processes]
            self.started = True

        retrieved_samples = 0
        while retrieved_samples < self.num_samples:
            q_idx = self.rs.choice(len(self.queues), p=self.p_queues)
            q = self.queues[q_idx]
            data = q.get()

            # convert to usable types
            data = [d.astype(np.float32) for d in data]
            yield data
            retrieved_samples += 1

    def cleanup(self):
        try:
            self.done_event.set()
        except AttributeError:
            pass

        for p in self.processes:
            p.join(timeout=2)

        for p in self.processes:
            if p.is_alive():
                p.terminate()

        for p in self.processes:
            p.join(timeout=3)

        for p in self.processes:
            if p.is_alive():
                p.kill()

        for p in self.processes:
            if p.is_alive():
                p.join()
            p.close()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
