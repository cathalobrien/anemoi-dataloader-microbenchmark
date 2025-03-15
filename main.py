import torch
from torch.utils.data import DataLoader


from anemoi.datasets.data import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.dataset import NativeGridDataset
from anemoi.training.data.dataset import worker_init_func
from anemoi.training.data.grid_indices import BaseGridIndices, FullGrid
from anemoi.utils.dates import frequency_to_seconds

import os
import subprocess
import time
import numpy as np
from sys import getsizeof
from collections.abc import Callable
from hydra.utils import instantiate
from pathlib import Path
from torch_geometric.data import HeteroData
import multiprocessing as mp

from memory_monitor import run_mem_monitor
from plot import plot_mem_monitor

try:
    from mpi4py import MPI
except ImportError:
    MPI4PY_AVAILABLE=False
else:
    MPI4PY_AVAILABLE=True
    comm = MPI.COMM_WORLD

training_batch_size=1
multistep_input=2
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

    #TODO support creating graphs on the fly
    #graph_config = convert_to_omegaconf(self.config).graph
    #return GraphCreator(config=graph_config).create(
    #    overwrite=self.config.graph.overwrite,
    #    save_path=graph_filename,
    #)

def _get_dataset(
    data_reader: Callable,
    grid_indices,
    shuffle: bool = True,
    rollout: int = 1,
    label: str = "generic",
    num_gpus_per_node : int = 1,
    num_nodes : int = 1,
    num_gpus_per_model : int = 1
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
    local_load_time = time.time() - s
    if MPI4PY_AVAILABLE:
        comm.Barrier() #every process has to load the batch
    elapsed = time.time() - s
    size=batch.element_size() * batch.nelement()
    p0print(f"p0: {format(size)} batch loaded in {elapsed:.2f}s ({elapsed-local_load_time:.2f}s in barrier). ")
    if elapsed > 1:
        throughput =(size /elapsed)
        p0print(f"p0: dataloader throughput: {format(throughput)}/s")
    return batch, np.array([elapsed, size])

def simulate_iter(rollout=1):
    throughput=0.2
    sleep_time=1/(throughput/rollout)
    p0print(f"Simulating an iter with {throughput=} and {rollout=}, sleeping for {sleep_time:.2f}s")
    time.sleep(sleep_time)

def setup_grid_indices(graph_filename, read_group_size=1):
    p0print(f"Loading the graph '{graph_filename}' (read group size: {read_group_size})...")
    grid_indices = FullGrid("data", read_group_size)
    grid_indices.setup(graph_data(graph_filename=graph_filename))
    p0print("Graph loaded.")
    return grid_indices


def create_dataset(dataset, grid_indices, rollout=1, num_nodes=1, num_gpus_per_node=1, num_gpus_per_model=1):
    r=rollout
    p0print(f"Opening the dataset '{dataset}' (rollout {r}, {int(num_nodes)} nodes with {num_gpus_per_node} 'GPUs' per node, model split over {num_gpus_per_model} 'GPUs')...")
    ds = _get_dataset(
                #open_dataset(self.config.model_dump().dataloader.training),
                open_dataset(dataset=dataset, start=None, end=202312, frequency=frequency, drop=[]),
                grid_indices,
                #SyntheticDataset(),
                label="train",
                rollout=r,
                num_nodes=num_nodes,
                num_gpus_per_node=num_gpus_per_node,
                num_gpus_per_model=num_gpus_per_model,
            )
    p0print("Dataset loaded.")
    return ds

def create_dataloader(ds, b=1, w=1, pf=1, pin_mem=True):
    seed=int(time.time())
    os.environ["ANEMOI_BASE_SEED"] = str(seed)
    p0print(f"Creating Dataloader (batch size: {b}, num_workers: {w}, prefetch_factor {pf}, {pin_mem=}, {seed=})...")
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
    p0print("Dataloader created.")
    return dataloader

def clear_page_cache():
    p0print("Clearing the page cache...")
    os.system("echo 1 > /proc/sys/vm/drop_caches")
    p0print("Page cache cleared.")
    
def get_parallel_info():
    """Reads Slurm env vars, if they exist, to determine if inference is running in parallel"""
    
    if MPI4PY_AVAILABLE:
        global_rank = comm.Get_rank()
        world_size= comm.Get_size()
    else:
        global_rank = 1
        world_size = 1

    #TODO change to MPI4PY, this is not portable
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))  # Rank within a node, between 0 and num_gpus
    procs_per_node = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1)) #number of processes on the current node
    num_nodes= world_size/procs_per_node
    p0print(f"Running in parallel across {world_size} processes over {int(num_nodes)} nodes")

    return global_rank, local_rank, world_size, procs_per_node, num_nodes

def p0print(str):
    #I dont love reading an env var for every print
    if MPI4PY_AVAILABLE and comm.Get_rank() == 0:
        print(str)

#TODO distinguish between per node and multinode throughput
#TODO gather metrics from all procs, rather then naively extrapolating from p0
def run(dataloader_iterator, count=5, simulate_compute=True, proc_count=1):
    #clear_page_cache() #permission denied on Atos
    roll_av=False
    averages = np.array([0.0,0.0])
    for i in range(0,count):
        p0print(f"Iteration {i}")
        _, metrics = load_batch(dataloader_iterator)
        #TODO add a barrier for all procs here
        #NEED TO KEEP CODE UNDER HERE TO A MINIMUM
        if roll_av:
            #averages = tuple((a * (i) + m) / (i+1) for a, m in zip(averages, metrics))
            averages += metrics
            p0print(f"Av time: {averages[0]:.2f}s, Av global throughput: {format(proc_count * averages[1]/averages[0])}B/s")
        else:
            averages += metrics
        if simulate_compute:
            simulate_iter()
    if not roll_av:
        #averages = tuple((av/count for av in averages))
        averages /= count
        p0print(f"Av time: {averages[0]:.2f}s, Av global throughput: {format(proc_count * averages[1]/averages[0])}B/s, Total time for {count} runs: {averages[0]*count:.2f}s")
            
        
def get_bm_config(test="single-worker-bm"):
    #set defaults
    config = {}
    config["test"]=test
    config["res"] = "o1280"
    config["graph_filename"] = "graphs/o1280.graph"
    config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o1280-2016-2023-6h-v1.zarr"]
    config["rollouts"] = [1]
    config["batch_sizes"]=[1]
    config["num_workers"]=[1]
    config["prefetch_factors"]=[1]
    config["pin_mem"]=[True]
    config["per_worker_count"]=True #if true, multiples count by nw
    config["monitor_memory"]=True
    
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
    
    p0print(f"Running a benchmark with the config '{test}'")
    return config 

def spawn_memory_monitor(config,count, r, bs, pf, nw, pm, rank, procs_per_node):
    if rank == 0 and config["monitor_memory"]:
        #generate a csv filename based on config
        test=config["test"]
        res=config["res"]
        
        os.makedirs(f"out/{test}", exist_ok=True)
        filename=f"out/{test}/mem-usage-{res}-{count}loads-{r}r-{bs}bs-{pf}pf-{nw}nw-{pm}pm-{procs_per_node}ppn.csv"
        #p = subprocess.run(["python " "memory_monitor.py " " False", " True ", filename])
        #never does anything
        p = mp.Process(target=run_mem_monitor, args=(False,True, filename))
        p.start()
        return p, filename
    else:
        return None, None
    
def terminate_memory_monitor(p, filename, test):
    #TODO call a mem monitor internal function to close the file before terminating
    if p is not None:
        p.terminate()
    if filename is not None:
        plot_mem_monitor(filename,show_plot=False,outdir=f"out/{test}")
            
        
def manager():
    #TODO support looping over resolutions with different graphs and dataset lists
    config = get_bm_config("single-worker-bm")
    count=10
    config["num_workers"]=[6]
    
    #config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o1280-2016-2023-6h-v1.zarr", "/lus/h2tcst01/ai-bm/datasets/aifs-od-an-oper-0001-mars-o1280-2016-2023-6h-v1.zarr"]
    
    #get parallel_info (#TODO refactor into a function which returns RGS, MCGS)
    global_rank, local_rank, world_size, procs_per_node, num_nodes = get_parallel_info()
    num_gpus_per_node=procs_per_node
    num_gpus_per_model=world_size #TODO allow a mix of data and model parallelism
    read_group_size=num_gpus_per_model
    
    gi = setup_grid_indices(config["graph_filename"], read_group_size=read_group_size)
    
    for dataset in config["datasets"]:
        for r in config["rollouts"]:
            ds = create_dataset(dataset, grid_indices=gi, rollout=r, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, num_gpus_per_model=num_gpus_per_model)
            for bs in config["batch_sizes"]:
                for pf in config["prefetch_factors"]:
                    for nw in config["num_workers"]:
                        for pm in config["pin_mem"]:
                            
                            if config["per_worker_count"]:
                                base_count=count
                                count=count*nw
                            
                            #TODO break these lines before run into a prelude function, could even make run a class with an __init__ and __del__
                            dl_iter = iter(create_dataloader(ds, b=bs, w=nw, pf=pf, pin_mem=pm))
                            p0print(f"Proc {global_rank}: Starting {count} runs with {r=}, {bs=}, {pf=}, {nw=}, {pm=}")
                            mem_monitor_proc, csv = spawn_memory_monitor(config, count, r, bs, pf, nw, pm, global_rank, procs_per_node)
                            
                            run(dl_iter, count=count, simulate_compute=False, proc_count=world_size)
                            
                            terminate_memory_monitor(mem_monitor_proc, csv, config["test"])
                            if config["per_worker_count"]:
                                count=base_count
                                

if __name__ == "__main__":
    manager()
#def __main__():
#    manager()
        
#TODO   add gpu support
#       measure time spent in HtoD copies
#       Split the 'anemoi' and benchmarking code into different files
#       plot av global throughput across various tests as a bar chart
#       Sync metrics across all procs, and measure min, max, variance etc

