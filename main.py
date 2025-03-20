import torch
from torch.utils.data import DataLoader


from anemoi.datasets.data import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.dataset import NativeGridDataset
from anemoi.training.data.dataset import worker_init_func as default_worker_init_func
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
import argparse
import math

from memory_monitor import run_mem_monitor
from plot import plot_mem_monitor, plot_anemoi_dataloader_benchmark

USE_MPI=False #If you want to disable MPI while having it installed
if USE_MPI:
    try:
        from mpi4py import MPI
    except ImportError:
        MPI4PY_AVAILABLE=False
    else:
        MPI4PY_AVAILABLE=True
        comm = MPI.COMM_WORLD
else:
    MPI4PY_AVAILABLE=False

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

def load_batch(dl_iter, verbose=True):  
    s = time.time()
    batch = next(dl_iter)
    local_load_time = time.time() - s
    if MPI4PY_AVAILABLE:
        comm.Barrier() #every process has to load the batch
    elapsed = time.time() - s
    if verbose: #could lead to slightly less accurate results since there's more work in between loading batches
        size=batch.element_size() * batch.nelement()
        p0print(f"p0: {format(size)}B batch loaded in {elapsed:.2f}s ({elapsed-local_load_time:.2f}s in barrier). ")
        if elapsed > 1:
            throughput =(size /elapsed)
            p0print(f"p0: dataloader throughput: {format(throughput)}B/s")
    return batch, elapsed

def simulate_iter(rollout=1, single_step_throughput=0.2, sleep_time=0):
    if (sleep_time == 0):
        #TODO is the cost of r=2 2xr=1? the #batches only goes from 3 to 4
        
        sleep_time=1/(single_step_throughput/rollout)
        p0print(f"Simulating an iter with {single_step_throughput=} and {rollout=}, sleeping for {sleep_time:.2f}s")
    else:
        p0print(f"Simulating a {sleep_time:.2f}s iter")
    time.sleep(sleep_time)

#you can check these by editing anemoi/training/data/grid_indices.py and editing compute_grid_sizes to print 'graph[self.nodes_name].num_nodes'
def get_grid_points(res):
    if res == "o2560":
        return 26306560
    elif res == "o1280":
        return 6599680
    elif res == "n320":
        return 542080
    else:
        return 0

#This file calculates the number of grid points for each reader in read group
#A graph filename is optional
#I have hardcoded the number of grid points for some common resolutions
#If your resolution is not hardcoded, you can provide a graph and the grid points will be read from there
#def setup_grid_indices(res, graph_filename=None, read_group_size=1):
def setup_grid_indices(res, graph_filename=None, read_group_size=1):
    start_time=time.time()
    grid_indices = FullGrid("data", read_group_size)
    #if a graph path is given, load it
    if graph_filename is not None:
        if not os.path.isfile(graph_filename):
            raise ValueError(f"Error! '{graph_filename}' not found")
        p0print(f"Loading the graph '{graph_filename}' (read group size: {read_group_size})...")
        grid_indices.setup(graph_data(graph_filename=graph_filename))
        p0print(f"Graph loaded in {time.time() - start_time:.2f}s.")
    else:
        grid_points=get_grid_points(res)
        if grid_points == 0:
            raise ValueError(f"Warning! res '{res}' has an unknown number of grid points. Please rerun and pass a graph path as 'graph_filename'")
        p0print(f"Running '{res}' resolution with {grid_points} points globally (read group size: {read_group_size})")
        #instead of calling FullGrid.setup() set the grid_size manually
        #IF FullGrid.setup() changes or if a different Grid type is used, this will break
        grid_indices.grid_size=grid_points
            
    return grid_indices


def create_dataset(dataset, grid_indices, rollout=1, num_nodes=1, num_gpus_per_node=1, num_gpus_per_model=1):
    r=rollout
    p0print(f"Opening the dataset '{dataset}' (rollout {r}, {int(num_nodes)} nodes with {num_gpus_per_node} 'GPUs' per node, model split over {num_gpus_per_model} 'GPUs')...")
    start_time=time.time()
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
    p0print(f"Dataset opened in {time.time() - start_time:.2f}s.")
    return ds

def create_dataloader(ds, b=1, w=1, pf=1, pin_mem=True, worker_init_func=default_worker_init_func):
    seed=int(time.time())
    os.environ["ANEMOI_BASE_SEED"] = str(seed)
    start_time=time.time()
    p0print(f"Creating Dataloader (batch size: {b}, num_workers: {w}, prefetch_factor {pf}, {pin_mem=}, {seed=})...")
    #TODO only need to set this once per dataload
    #could ignore the mem monitor process
    #os.system("source /ec/res4/hpcperm/naco/aifs/anemoi-dataloader-microbenchmark/darshan/enable-darshan")
    #os.system("export PMIX_MCA_gds=hash") #needed to when creatin procs via
    #os.environ["LD_PRELOAD"]="/etc/ecmwf/nfs/dh1_home_a/naco/libs_built_from_source/darshan/darshan-3.4.5/gcc-8.5.0-ompi4.1.1.1-install/lib/libdarshan.so"
    #os.environ["DARSHAN_ENABLE_NONMPI"]=1
    #if os.environ.get('DARSHAN_DISABLE', 'unset') != "unset":
    #    del os.environ["DARSHAN_DISABLE"] #unset "DARSHAN_DISABLE" before spawning the dataloader processes
    #p0print(f"'DARSHAN_DISABLE' = {os.environ.get('DARSHAN_DISABLE', 'unset')}")
    #os.system("export DARSHAN_ENABLE_NONMPI=1")
    dataloader = DataLoader(
            ds,
            batch_size=b, #self.config.model_dump().dataloader.batch_size[stage],
            # number of worker processes
            num_workers=w, #self.config.model_dump().dataloader.num_workers[stage],
            #pin_memory=pin_mem, #self.config.dataloader.pin_memory,
            pin_memory=True, #self.config.dataloader.pin_memory,
            worker_init_fn=worker_init_func,
            # prefetch batches
            prefetch_factor=pf, #self.config.dataloader.prefetch_factor,
            persistent_workers=True,
            multiprocessing_context="fork" #fork is the default, spawn and forkserver crash with '"No permission" (-17) instead of "Success" '
    )
    dl_iter=iter(dataloader)
    #os.environ["DARSHAN_DISABLE"] = "1" 
    #os.system("unset DARSHAN_ENABLE_NONMPI")
    p0print(f"Dataloader created in {time.time() - start_time:.2f}s.")
    return dl_iter

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
        global_rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))

    #TODO change to MPI4PY, this is not portable
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", int(os.environ.get("SLURM_LOCALID", 0))))  # Rank within a node, between 0 and num_gpus
    procs_per_node = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", int(os.environ.get("SLURM_TASKS_PER_NODE", '1').split('(')[0]))) #number of processes on the current node
    num_nodes= world_size/procs_per_node
    p0print(f"Running in parallel across {world_size} processes over {int(num_nodes)} nodes")

    return global_rank, local_rank, world_size, procs_per_node, num_nodes

def p0print(str):
    #I dont love reading an env var for every print
    if MPI4PY_AVAILABLE:
        if comm.Get_rank() == 0:
            print(str)
    else:
        if int(os.environ.get("SLURM_PROCID", 0)) == 0:
            print(str)

def spawn_memory_monitor(test,res, dataset_index,count, r, bs, pf, nw, pm, rank, num_nodes, procs_per_node):
    if rank == 0:
        #generate a csv filename based on config
        setup=f"mem-usage-{res}-ds{dataset_index}-{count}loads-{r}r-{bs}bs-{pf}pf-{nw}nw-{pm}pm-{int(num_nodes)}N-{procs_per_node}ppn" 
        dir=f"out/{test}/{setup}"
        os.makedirs(dir, exist_ok=True)
        filename=f"{dir}/{setup}.csv"
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
        plot_mem_monitor(filename,show_plot=False,outdir=os.path.dirname(filename),filename_prefix=test)

def create_results_file(config,rank, filename="anemoi-dataloader-microbenchmark.csv",save_output=True):
    if save_output and rank==0:
        #write_header = not os.path.exists(filename)
        outdir=f"out/"
        output=f"{outdir}/{filename}"
        os.makedirs(outdir, exist_ok=True)
        f = open(output, "w")
        print(f"Creating output file: {output}")
        #if write_header:
        header="res,dataset,rollout,batch_size,num_workers,prefetch_factor,pin_memory,count,num_procs,elapsed(s),worker-throughput(byte/s),proc-throughput(byte/s),global-throughput(byte/s)\n"
        f.write(header)
        #f.flush()
        return f
    else:
        return None
    
    #wrapper to worker_init_func which enables darshan on the dataloader processes
    #It calls 'anemoi-dataloader-microbenchmark/darshan/enable-darshan' which sets the relevant env vars
def worker_init_func_w_darshan(worker_id: int):
    if MPI4PY_AVAILABLE:
        #print("Initalising MPI in dataloader process")
        #this is horrible, but since we use MPI in main.py, we have to launch darshan in MPI
        #Therefore the dataloaders must also call MPI.init to be picked up by Darshan
        print(f"{worker_id=} {MPI.Is_initialized()=}")
        #MPI.Init_thread(required=MPI.THREAD_FUNNELED)
        #print("MPI initalised in dataloader process")
        #exit()
    #os.system("./darshan/enable-darshan") #TODO replace with relative path
    #os.environ['DARSHAN_ENABLE_NONMPI']='1'
    #print(f"{worker_id=} {os.system('env | grep -i DARSHAN')}")
    
    #os.system("source /ec/res4/hpcperm/naco/aifs/anemoi-dataloader-microbenchmark/darshan/enable-darshan")
    default_worker_init_func(worker_id=worker_id)
    
def enable_darshan(log_path):
    #only works on Atos
    os.makedirs(log_path, exist_ok=True)
    p0print(f"Enabling Darshan. Darshan logs will be written to '{log_path}'.")

    #TODO rebuild with gnu to remove this intel compiler dependancy
    org_ld_library_path=os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"/usr/local/apps/intel/2021.4.0/compiler/latest/linux/compiler/lib/intel64:{org_ld_library_path}"

    org_ld_preload=os.environ.get("LD_PRELOAD", "")
    os.environ["LD_PRELOAD"] = f"/home/naco/libs_built_from_source/darshan/darshan-3.4.5/install/lib/libdarshan.so:{org_ld_preload}"
    os.environ["DARSHAN_EXCLUDE_DIRS"] = os.environ.get("VIRTUAL_ENV") #dont log loading the python libs into memory
    os.environ["DARSHAN_LOG_PATH"]=log_path
    #DARSHAN_ENABLE_NONMPI=1  
        
def save_results(f, results, res, count, ds, r, bs, pf, nw, pm, num_procs, save_output=True):
    #header="res,dataset,rollout,batch_size,num_workers,prefetch_factor,pin_memory,count,num_procs,elapsed(s),throughput(byte/s)\n"
    if save_output and f is not None:
        line=f"{res},{ds},{r},{bs},{nw},{pf},{pm},{count},{num_procs},{results[0]},{results[1]/nw},{results[1]},{results[1]*num_procs}\n"
        f.write(line)
        #f.flush()
    
    

#TODO distinguish between per node and multinode throughput
#TODO gather metrics from all procs, rather then naively extrapolating from p0
def run(dataloader_iterator, num_workers, count=5, simulate_compute=True, proc_count=1, rollout=1, sleep_time=0):
    #clear_page_cache() #permission denied on Atos

    #each process maintains a list of their times to load
    #this will be gathered on proc 0 at the end of the run and the mean, min, max etc will be calculated
    times = np.array(range(0,count), dtype=np.float32)
    size = None
    for i in range(0,count):
        p0print(f"Iteration {i}")
        batch, times[i] = load_batch(dataloader_iterator)
        #TODO ensure that batch is overwritten i.e. not a mem leak, cant have N * ~1G batches piling up
        #NEED TO KEEP CODE UNDER HERE TO A MINIMUM
        if size is None:
            size = batch.element_size() * batch.nelement()
            
        if simulate_compute:
            simulate_iter(rollout=rollout, sleep_time=sleep_time)
        #the following code syncs time across all procs and gets the average
   #if comm.Get_rank() == 0:
        
        
    global_times = None
    if MPI4PY_AVAILABLE:
        rank = comm.Get_rank()
    else:
        rank = 0
    if rank == 0:
        global_times = np.empty([proc_count, count], dtype=np.float32)
        
    if MPI4PY_AVAILABLE:
        comm.Gather(times, global_times, root=0)
    else:
        global_times=times
    
    if rank == 0:
        global_times=global_times.flatten() #from 2D [[count]*worldsize] to 1D [count*worldsize]
        global_times =np.flip(np.sort(global_times)) #largest first
        mean_time_per_process=global_times.mean()
        mean_time_per_worker=mean_time_per_process*num_workers #8s

        #The bottom (nw-1)/nw loads tend to be close to 0 since they're loaded in parallel, this tried to extract just the stalls
        stalls=global_times[0:int(len(global_times)/num_workers)]
        
        #TODO I'm sure there's a bug here
        min_time=global_times.min()
        max_time=global_times.max()
        total_time=times.sum()
        time_std_dev=global_times.std()
        #mean_time=mean_time_per_worker*num_workers
        av_throughput_per_worker=size/mean_time_per_worker #100
        #av_throughput_per_worker=size/mean_time_per_worker/num_workers #analytically im putting this here but i dont know why
        av_throughput_per_process=av_throughput_per_worker*num_workers #200
        av_throughput_global=av_throughput_per_process*proc_count #400
        p0print(f"Total time for {count} loads: {total_time:.2f}s")
        p0print(f"avg={mean_time_per_process:.2f}s, max={max_time:.2f}s, min={min_time:.2f}s, std-dev={time_std_dev:.4f}s")
        if num_workers > 1:
            p0print(f"From 'Stalls' (top 1/{num_workers}): avg={stalls.mean():.2f}s, max={stalls.max():.2f}s, min={stalls.min():.2f}s, std-dev={stalls.std():.4f}s")
        p0print(f"per-worker BW: {format(av_throughput_per_worker)}B/s,  per-process BW: {format(av_throughput_per_process)}B/s, global BW: {format(av_throughput_global)}B/s")
        #p0print(f"Av BW per worker = {format(av_throughput)}B/s, input batch size = {format(averages[1])}B => compute throughput should be >= {1.0/averages[0]:.3f}it/s to avoid starvation")
        
        #Is this fair since I'm loading batches one after another with no compute simulated in between?
        p0print(f"Estimated throughput upper-bound: {1.0/mean_time_per_process:.3f}it/s ({format(av_throughput_per_process)}B/s / {format(size)}B)")
        latency=mean_time_per_worker
        p0print(f"Est. latency to load the initial batch: {latency:.2f}s") 
        return [mean_time_per_process, av_throughput_per_process]
    else:
        return [None, None]
    
def get_bm_config(test="single-worker-bm"):
    #set defaults
    
    config = {}
    config["test"]=test
    config["resolutions"] = ["o1280"]
    config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o1280-2016-2023-6h-v1.zarr"]
    config["rollouts"] = [1]
    config["batch_sizes"]=[1]
    config["num_workers"]=[1]
    config["prefetch_factors"]=[1]
    config["pin_mem"]=[True]
    config["per_worker_count"]=True #if true, multiples count by nw
    config["monitor_memory"]=True
    config["use_darshan"]=True
    
    tests=["single-worker-bm", "rollout-bm", "multi-worker-bm", "4.4km", "compression"]
    
    if test == "single-worker-bm":
        pass
    elif test == "different-resolutions":
        #config["resolutions"] = ["o1280", "n320"]
        config["resolutions"] = ["n320", "o1280"]
        config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o1280-2016-2023-6h-v1.zarr", "/home/mlx/ai-ml/datasets/stable/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v4.zarr"]
    elif test == "4.4km":
        config["resolutions"] = ["o2560"]
        config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-rd-an-lwda-ifc3-mars-o2560-2023-2023-6h-v1-1week.zarr"]
    elif test == "compression":
        config["resolutions"] = ["n320"]
        config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-od-an-oper-0001-mars-n320-2016-2023-6h-v8.zarr", "/lus/h2resw01/scratch/mafp/public/cathal/aifs-od-an-oper-0001-mars-n320-2016-2023-6h-v8-with-no-compression.zarr"]
        config["rollouts"] = [12, 120] #large rollout to get large n320 batches
    elif test == "rollout-bm":
        config["rollouts"] = [1,12]
    elif test == "multi-worker-bm":
        config["num_workers"] = [1,2,4,8]
        config["per_worker_count"]=True
    elif test == "zarr-chunked-by-grid-dim":
        config["resolutions"] = ["o2560"]
        config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-rd-an-lwda-ifc3-mars-o2560-2023-2023-6h-v1-1week.zarr", "/home/mlx/ai-ml/datasets/aifs-rd-an-lwda-ifc3-mars-o2560-2023-2023-6h-v3-1week.zarr"]
    else:
        raise ValueError(f"Error. invalid benchmark. Please select one of '{tests}'")
    
    p0print(f"Running a benchmark with the config '{test}'")
    return config 
            
def manager():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--gpus-per-model', type=int, default=0)
    parser.add_argument('-r','--read-group-size',type=int, default=0)
    args = parser.parse_args()
    
    #TODO support looping over resolutions with different graphs and dataset lists
    #config = get_bm_config("single-worker-bm")
    #config = get_bm_config("different-resolutions")
    #config = get_bm_config("4.4km")
    config = get_bm_config("zarr-chunked-by-grid-dim")
    #config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-rd-an-lwda-ifc3-mars-o2560-2023-2023-6h-v1-1week.zarr"]
    #config["resolutions"] =["o2560"]
    count=1
    config["num_workers"]=[1]
    config["prefetch_factors"]=[1]
    config["rollouts"] = [1]
    config["test"]="debug"
    
    #config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o1280-2016-2023-6h-v1.zarr", "/lus/h2tcst01/ai-bm/datasets/aifs-od-an-oper-0001-mars-o1280-2016-2023-6h-v1.zarr"]
    
    #get parallel_info (#TODO refactor into a function which returns RGS, MCGS)
    global_rank, local_rank, world_size, procs_per_node, num_nodes = get_parallel_info()
    num_gpus_per_node=procs_per_node
    if args.gpus_per_model != 0:
        num_gpus_per_model=args.gpus_per_model
    else:
        num_gpus_per_model=world_size
    if args.read_group_size != 0:
        read_group_size=args.read_group_size
    else:
        read_group_size=num_gpus_per_model
    #check for invalid inputs
    if num_gpus_per_model < 1 or num_gpus_per_model > world_size or (not math.log2(num_gpus_per_model).is_integer()) :
        raise ValueError(f"Error! Model group size ({num_gpus_per_model}) must be a power of 2 between 1 and 'world-size' ({world_size}) ")
    if read_group_size < 1 or read_group_size > num_gpus_per_model or num_gpus_per_model % read_group_size != 0:
        raise ValueError(f"Error! read group size ({read_group_size}) must be a number between 1 and 'model-size' ({num_gpus_per_model}) which divides evenly into 'model-size'")
    
    
    save_output=False
    dir=f"out/{config['test']}"
    f = create_results_file(config,global_rank, save_output=save_output) #could do this in __init__ if it was a class

    worker_init_func=default_worker_init_func
    if config["use_darshan"]:
        p0print("Enabling Darshan on dataloader processes")
        worker_init_func=worker_init_func_w_darshan
   
    for res in config["resolutions"]:
        gi = setup_grid_indices(res, read_group_size=read_group_size)
        #filter the list of datasets to just the matching resolutions
        datasets=[dataset for dataset in config["datasets"] if res in dataset]
        for dataset in datasets:
            for r in config["rollouts"]:
                #os.system("source /ec/res4/hpcperm/naco/aifs/anemoi-dataloader-microbenchmark/darshan/enable-darshan")
                ds = create_dataset(dataset, grid_indices=gi, rollout=r, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, num_gpus_per_model=num_gpus_per_model)
                for bs in config["batch_sizes"]:
                    for pf in config["prefetch_factors"]:
                        for nw in config["num_workers"]:
                            for pm in config["pin_mem"]:
                                
                                if config["per_worker_count"]:
                                    base_count=count
                                    count=count*nw
                                
                                #TODO break these lines before run into a prelude function, could even make run a class with an __init__ and __del__
                                p0print(f"Proc {global_rank}: Starting {count} loads with {r=}, {bs=}, {pf=}, {nw=}, {pm=}")
                                if save_output and config["monitor_memory"]:
                                    mem_monitor_proc, csv = spawn_memory_monitor(config["test"],res, datasets.index(dataset), count, r, bs, pf, nw, pm, global_rank, num_nodes, procs_per_node)
                                dl_iter = create_dataloader(ds, b=bs, w=nw, pf=pf, pin_mem=pm, worker_init_func=worker_init_func)
                                
                                try:
                                    results = run(dl_iter, nw, count=count, simulate_compute=False, proc_count=world_size, rollout=r, sleep_time=25)
                                    save_results(f, results, res, count, dataset, r, bs, pf, nw, pm, world_size, save_output=save_output)
                                except MemoryError:
                                    print("!!!OUT OF MEMORY!!!")
                                    pass

                                del(dl_iter) #try to trigger __del__ on pytorch dataloader procs
                                
                                if save_output and config["monitor_memory"]:
                                    terminate_memory_monitor(mem_monitor_proc, csv, config["test"])
                                if config["per_worker_count"]:
                                    count=base_count
                                
    if save_output and f is not None:
        f.close()
        try:
            plot_anemoi_dataloader_benchmark(f.name, outdir=dir, outname=f"{config['test']}-j{os.environ.get('SLURM_JOBID','0')}-{int(time.time())}")
        except ValueError as err:
            p0print(f"Error plotting: {err}")
                                

if __name__ == "__main__":
    manager()
#def __main__():
#    manager()
        
#TODO   add gpu support
#       measure time spent in HtoD copies
#       Split the 'anemoi' and benchmarking code into different files
#       Gather time spent at barrier for each proc and analyse them, maybe there's some repeast offenders
#       distinguish between per node and global throughput (think atos node has 480MB/s max BW)
#       Add code which complains if graphs arent in 'graphs/'
#       Confirm that the graph i use does not impact performance
#       Understand why the pytorch insufficient cpu core message pops up even when my job has enough cores
#       Add requirements.txt


#Pre-reqs for a scaling run
#   Multinode support with a mix of model and data parallelism


