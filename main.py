import torch
from torch.utils.data import DataLoader


from anemoi.datasets.data import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.dataset import NativeGridDataset
from anemoi.training.data.dataset import worker_init_func
from anemoi.training.data.grid_indices import BaseGridIndices, FullGrid
from anemoi.utils.dates import frequency_to_seconds

import os
import time
import numpy as np
from sys import getsizeof
from collections.abc import Callable
from hydra.utils import instantiate
from pathlib import Path
from torch_geometric.data import HeteroData

training_batch_size=1
num_gpus_per_node=1
num_nodes=1
num_gpus_per_model=1
multistep_input=2
read_group_size=num_gpus_per_model
frequency="6h"
timestep="6h"
frequency_s = frequency_to_seconds(frequency)
timestep_s = frequency_to_seconds(timestep)
time_inc = timestep_s // frequency_s

def format(size):
    for unit in ('', 'K', 'M', 'G'):
        if size < 1024:
            break
        size /= 1024.0
    return "%.1f%s" % (size, unit)
 

def graph_data(graph_filename=None) -> HeteroData:
    """Graph data.

    Creates the graph in all workers.
    """
    overwrite=False
    if graph_filename is not None:
        graph_filename = Path(graph_filename)

        if graph_filename.exists() and not overwrite:
            return torch.load(graph_filename, weights_only=False)

    else:
        graph_filename = None

    from anemoi.graphs.create import GraphCreator

    #graph_config = convert_to_omegaconf(self.config).graph
    #return GraphCreator(config=graph_config).create(
    #    overwrite=self.config.graph.overwrite,
    #    save_path=graph_filename,
    #)
    
def grid_indices(self) -> type[BaseGridIndices]:
        reader_group_size = self.config.dataloader.read_group_size

        grid_indices = instantiate(
            self.config.model_dump(by_alias=True).dataloader.grid_indices,
            reader_group_size=reader_group_size,
        )
        grid_indices.setup(self.graph_data)
        return grid_indices



def _get_dataset(
    data_reader: Callable,
    grid_indices,
    shuffle: bool = True,
    rollout: int = 1,
    label: str = "generic",
) -> NativeGridDataset:

    r = rollout

    # Compute effective batch size
    effective_bs = (
        training_batch_size
        * num_gpus_per_node
        * num_nodes
        // num_gpus_per_model)

    return NativeGridDataset(
        data_reader=data_reader,
        rollout=r,
        multistep=multistep_input,
        timeincrement=time_inc,
        shuffle=shuffle,
        grid_indices=grid_indices,
        label=label,
        effective_bs=effective_bs,
    )

def load_batch(dl_iter):  
    s = time.time()
    batch = next(dl_iter)
    elapsed = time.time() - s
    size=batch.element_size() * batch.nelement()
    print(f"{format(size)} batch loaded in {elapsed:.2f}s.")
    if elapsed > 1:
        throughput =(size /elapsed)
        print(f"Dataloader throughput: {format(throughput)}/s")
    return batch, np.array([elapsed, size])

def simulate_iter(rollout=1):
    throughput=0.2
    sleep_time=1/(throughput/rollout)
    print(f"Simulating an iter with {throughput=} and {rollout=}, sleeping for {sleep_time:.2f}s")
    time.sleep(sleep_time)

def setup_grid_indices(graph_filename):
    print(f"Loading the graph '{graph_filename}'...")
    grid_indices = FullGrid("data", read_group_size)
    grid_indices.setup(graph_data(graph_filename=graph_filename))
    print("Graph loaded.")
    return grid_indices


def create_dataset(dataset, grid_indices, rollout=1):
    r=rollout
    print(f"Opening the dataset '{dataset}' (rollout {r})...")
    ds = _get_dataset(
                #open_dataset(self.config.model_dump().dataloader.training),
                open_dataset(dataset=dataset, start=None, end=202312, frequency=frequency, drop=[]),
                grid_indices,
                #SyntheticDataset(),
                label="train",
                rollout=r
            )
    print("Dataset loaded.")
    return ds

def create_dataloader(ds, b=1, w=1, pf=1, pin_mem=True):
    seed=int(time.time())
    os.environ["ANEMOI_BASE_SEED"] = str(seed)
    print(f"Creating Dataloader (batch size: {b}, num_workers: {w}, prefetch_factor {pf}, {pin_mem=}, {seed=})...")
    dataloader = DataLoader(
            ds,
            batch_size=b, #self.config.model_dump().dataloader.batch_size[stage],
            # number of worker processes
            num_workers=w, #self.config.model_dump().dataloader.num_workers[stage],
            pin_memory=pin_mem, #self.config.dataloader.pin_memory,
            worker_init_fn=worker_init_func,
            # prefetch batches
            prefetch_factor=pf, #self.config.dataloader.prefetch_factor,
            persistent_workers=True,
    )
    print("Dataloader created.")
    return dataloader

def clear_page_cache():
    print("Clearing the page cache...")
    os.system("echo 1 > /proc/sys/vm/drop_caches")
    print("Page cache cleared.")

def run(dataloader_iterator, count=5, simulate_compute=True):
    #clear_page_cache() #permission denied on Atos
    roll_av=False
    averages = np.array([0.0,0.0])
    for i in range(0,count):
        print(f"Iteration {i}")
        _, metrics = load_batch(dataloader_iterator)
        #NEED TO KEEP CODE UNDER HERE TO A MINIMUM
        if roll_av:
            #averages = tuple((a * (i) + m) / (i+1) for a, m in zip(averages, metrics))
            averages += metrics
            print(f"Av time: {averages[0]:.2f}s, Av throughput: {format(averages[1]/averages[0])}")
        else:
            averages += metrics
        if simulate_compute:
            simulate_iter()
    if not roll_av:
        #averages = tuple((av/count for av in averages))
        averages /= count
        print(f"Av time: {averages[0]:.2f}s, Av throughput: {format(averages[1]/averages[0])}, Total time for {count} runs: {averages[0]*count:.2f}s")
            
        
def get_bm_config(test="single-worker-bm"):
    #set defaults
    config = {}
    config["res"] = "o1280"
    config["graph_filename"] = "graphs/o1280.graph"
    config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o1280-2016-2023-6h-v1.zarr"]
    config["rollouts"] = [1]
    config["batch_sizes"]=[1]
    config["num_workers"]=[1]
    config["prefetch_factors"]=[1]
    config["pin_mem"]=[True]
    config["per_worker_count"]=False #if true, multiples count by nw
    
    tests=["single-worker-bm", "rollout-bm", "multi-worker-bm"]
    
    if test == "single-worker-bm":
        pass
    elif test == "rollout-bm":
        config["rollouts"] = [1,12]
    elif test == "multi-worker-bm":
        config["num_workers"] = [1,2,4,8]
        config["per_worker_count"]=True
    else:
        raise ValueError(f"Error. invalid benchmark. Please select one of '{tests}'")
    
    print(f"Running a benchmark with the config '{test}'")
    return config 
            
        
def manager():
    #TODO support looping over resolutions with different graphs and dataset lists
    config = get_bm_config("single-worker-bm")
    count=1
    
    gi = setup_grid_indices(config["graph_filename"])
    
    for dataset in config["datasets"]:
        for r in config["rollouts"]:
            ds = create_dataset(dataset, grid_indices=gi, rollout=r)
            for bs in config["batch_sizes"]:
                for pf in config["prefetch_factors"]:
                    for nw in config["num_workers"]:
                        for pm in config["pin_mem"]:
                            
                            if config["per_worker_count"]:
                                base_count=count
                                count=count*nw
                            dl_iter = iter(create_dataloader(ds, b=bs, w=nw, pf=pf, pin_mem=pm))
                            print(f"Starting {count} runs with {r=}, {bs=}, {pf=}, {nw=}, {pm=}")
                            run(dl_iter, count=count, simulate_compute=False) 
                            if config["per_worker_count"]:
                                count=base_count
                                
                            
manager()
#TODO   add gpu support
#       measure time spent in HtoD copies
#       add multiprocess support
#       Incorporate memory monitor, launch it before and rename csv output
#       Split the 'anemoi' and benchmarking code into different files

