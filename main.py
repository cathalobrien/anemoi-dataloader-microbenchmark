import os
import time
import numpy as np
import multiprocessing as mp
import argparse
import math

from misc import maybe_import_mpi, format, p0print
from memory_monitor import run_mem_monitor
from plot import plot_mem_monitor, plot_anemoi_dataloader_benchmark
from setup_anemoi import setup_grid_indices, create_dataloader, create_dataset
 
MPI4PY_AVAILABLE, comm = maybe_import_mpi() 

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

def clear_page_cache():
    p0print("Clearing the page cache...")
    os.system("echo 1 > /proc/sys/vm/drop_caches")
    p0print("Page cache cleared.")
    
def get_parallel_info(args):
    """Reads Slurm env vars, if they exist, to determine if inference is running in parallel"""
    
    if MPI4PY_AVAILABLE:
        global_rank = comm.Get_rank()
        world_size= comm.Get_size()
    else:
        global_rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", int(os.environ.get("SLURM_LOCALID", 0))))  # Rank within a node, between 0 and num_gpus
    procs_per_node = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", int(os.environ.get("SLURM_TASKS_PER_NODE", '1').split('(')[0]))) #number of processes on the current node
    num_nodes= world_size//procs_per_node
    p0print(f"Running in parallel across {world_size} processes over {int(num_nodes)} nodes")
   
    #Optionally, get num_gpus_per_model and read_group_size from args
    num_gpus_per_model=world_size
    if args.gpus_per_model != 0:
        num_gpus_per_model=args.gpus_per_model
        
    read_group_size=num_gpus_per_model
    if args.read_group_size != 0:
        read_group_size=args.read_group_size
        
    #check for invalid inputs
    if num_gpus_per_model < 1 or num_gpus_per_model > world_size or (not math.log2(num_gpus_per_model).is_integer()) :
        raise ValueError(f"Error! Model group size ({num_gpus_per_model}) must be a power of 2 between 1 and 'world-size' ({world_size}) ")
    if read_group_size < 1 or read_group_size > num_gpus_per_model or num_gpus_per_model % read_group_size != 0:
        raise ValueError(f"Error! read group size ({read_group_size}) must be a number between 1 and 'model-size' ({num_gpus_per_model}) which divides evenly into 'model-size'")
    

    return global_rank, world_size, procs_per_node, num_nodes, num_gpus_per_model, read_group_size

def spawn_memory_monitor(test,res, dataset_index,count, r, bs, pf, nw, pm, rank, num_nodes, procs_per_node):
    if rank == 0:
        #generate a csv filename based on config
        setup=f"mem-usage-{res}-ds{dataset_index}-{count}loads-{r}r-{bs}bs-{pf}pf-{nw}nw-{pm}pm-{int(num_nodes)}N-{procs_per_node}ppn" 
        dir=f"out/{test}/{setup}"
        os.makedirs(dir, exist_ok=True)
        filename=f"{dir}/{setup}.csv"
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

def gather_timings(local_times, proc_count, run_count):
    #the following code gathers timing data across all procs and gets the average
    global_times = None
    if MPI4PY_AVAILABLE:
        rank = comm.Get_rank()
    else:
        rank = 0
        
    if MPI4PY_AVAILABLE:
        if rank == 0: #create output buffer on p0
            global_times = np.empty([proc_count, run_count], dtype=np.float32)
        comm.Gather(local_times, global_times, root=0)
    else:
        p0print("MPI not available -> stats can not be aggregated from all processes, they will be based on P0 instead.")
        global_times=local_times
        
    return global_times

#P0 computes stats for global timing data across all procs
def compute_run_stats(global_times, local_times, run_count, num_workers, proc_count, input_batch_size):
    if comm.Get_rank() == 0:
        global_times=global_times.flatten() #from 2D [[count]*worldsize] to 1D [count*worldsize]
        global_times =np.flip(np.sort(global_times)) #largest first
        mean_time_per_process=global_times.mean()
        mean_time_per_worker=mean_time_per_process*num_workers 

        #The bottom (nw-1)/nw loads tend to be close to 0 since they're loaded in parallel, this tried to extract just the stalls
        stalls=global_times[0:int(len(global_times)/num_workers)]
        
        min_time=global_times.min()
        max_time=global_times.max()
        total_time=local_times.sum()
        time_std_dev=global_times.std()
        av_throughput_per_worker=input_batch_size/mean_time_per_worker 
        av_throughput_per_process=av_throughput_per_worker*num_workers 
        av_throughput_global=av_throughput_per_process*proc_count
        
        p0print(f"Total time for {run_count} loads: {total_time:.2f}s")
        p0print(f"avg={mean_time_per_process:.2f}s, max={max_time:.2f}s, min={min_time:.2f}s, std-dev={time_std_dev:.4f}s")
        if num_workers > 1:
            p0print(f"From 'Stalls' (top 1/{num_workers}): avg={stalls.mean():.2f}s, max={stalls.max():.2f}s, min={stalls.min():.2f}s, std-dev={stalls.std():.4f}s")
        p0print(f"per-worker BW: {format(av_throughput_per_worker)}B/s,  per-process BW: {format(av_throughput_per_process)}B/s, global BW: {format(av_throughput_global)}B/s")
 
        p0print(f"Estimated throughput upper-bound: {1.0/mean_time_per_process:.3f}it/s ({format(av_throughput_per_process)}B/s / {format(input_batch_size)}B)")
        latency=mean_time_per_worker
        p0print(f"Est. latency to load the initial batch: {latency:.2f}s")
        return [mean_time_per_process, av_throughput_per_process, input_batch_size]
    else:
        return [None, None, None]

def create_results_file(config,rank, filename="anemoi-dataloader-microbenchmark.csv",save_output=True, outdir="out/"):
    if save_output and rank==0:
        #write_header = not os.path.exists(filename)
        output=f"{outdir}/{filename}"
        os.makedirs(outdir, exist_ok=True)
        f = open(output, "w")
        #if write_header:
        header="res,dataset,rollout,batch_size,num_workers,prefetch_factor,pin_memory,count,num_procs,elapsed(s),worker-throughput(byte/s),proc-throughput(byte/s),global-throughput(byte/s),input_batch_per_proc(bytes)\n"
        f.write(header)
        #f.flush()
        return f
    else:
        return None
    
def save_results(f, results, res, count, ds, r, bs, pf, nw, pm, num_procs, save_output=True):
    #header="res,dataset,rollout,batch_size,num_workers,prefetch_factor,pin_memory,count,num_procs,elapsed(s),throughput(byte/s),input_batch_per_proc(bytes)\n"
    if save_output and f is not None:
        line=f"{res},{ds},{r},{bs},{nw},{pf},{pm},{count},{num_procs},{results[0]},{results[1]/nw},{results[1]},{results[1]*num_procs},{results[2]}\n"
        f.write(line)
        #f.flush()
    
#TODO distinguish between per node and multinode throughput
def run(dataloader_iterator, num_workers, count=5, simulate_compute=True, proc_count=1, rollout=1, sleep_time=0):
    #clear_page_cache() #permission denied on Atos

    #each process maintains a list of their times to load
    #this will be gathered on proc 0 at the end of the run and the mean, min, max etc will be calculated
    times = np.array(range(0,count), dtype=np.float32)
    size = None
    for i in range(0,count):
        p0print(f"Iteration {i}")
        batch, times[i] = load_batch(dataloader_iterator)
        
        if size is None: #Compute size only once at the start
            size = batch.element_size() * batch.nelement()
            
        if simulate_compute:
            simulate_iter(rollout=rollout, sleep_time=sleep_time)

    global_times = gather_timings(times, proc_count, count)
    
    return compute_run_stats(global_times, times, count, num_workers, proc_count, size)
    
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
    
    tests=["single-worker-bm", "rollout-bm", "multi-worker-bm", "4.4km", "compression"]
    
    if test == "single-worker-bm":
        pass
    elif test == "different-resolutions":
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
        config["resolutions"] = ["o1280"]
        config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month.zarr", "/ec/res4/scratch/naco/aifs/inputs/custom/aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month-4gridchunks.zarr", "/ec/res4/scratch/naco/aifs/inputs/custom/aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month-8gridchunks.zarr", "/ec/res4/scratch/naco/aifs/inputs/custom/aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month-16gridchunks.zarr", "/ec/res4/scratch/naco/aifs/inputs/custom/aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month-32gridchunks.zarr"]
    elif test == "zarr-chunked-by-grid-dim-s2":
        config["resolutions"] = ["o1280"]
        config["datasets"] = ["/ec/res4/scratch/naco/aifs/inputs/custom/aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month-16gridchunks.zarr", "/lus/h2tcst01/ai-bm/datasets/aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month-16gridchunks.zarr"]
        #config["datasets"] = [ "aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month-4gridchunks.zarr"]
    elif test == "test-custom":
        config["resolutions"] = ["o1280"]
        config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month.zarr", "/ec/res4/scratch/naco/aifs/inputs/custom/aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month-8gridchunks.zarr"]

    elif test == "9km-at-scale":
        config["resolutions"] = ["o1280"]
        config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month.zarr", "/ec/res4/scratch/naco/aifs/inputs/custom/aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month-16gridchunks.zarr", "/lus/st2/ai-bm/datasets/aifs-od-an-oper-0001-mars-o1280-2023-2023-6h-v1-one-month-16gridchunks.zarr"]
        config["num_workers"]=[4]
    else:
        
        raise ValueError(f"Error. invalid benchmark. Please select one of '{tests}'")
    
    p0print(f"Running a benchmark with the config '{test}'")
    return config 
            
def manager():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--gpus-per-model', type=int, default=0)
    parser.add_argument('-r','--read-group-size',type=int, default=0)
    args = parser.parse_args()
    
    #config = get_bm_config("single-worker-bm")
    #config = get_bm_config("different-resolutions")
    #config = get_bm_config("4.4km")
    config = get_bm_config("9km-at-scale")
    #config = get_bm_config("test-custom")
    
    #config["datasets"] = ["/home/mlx/ai-ml/datasets/aifs-rd-an-lwda-ifc3-mars-o2560-2023-2023-6h-v1-1week.zarr"]
    #config["resolutions"] =["o2560"]
    count=10
    #config["num_workers"]=[1]
    #config["prefetch_factors"]=[1]
    #config["rollouts"] = [1]
    #config["test"]="debug"
    
    global_rank, world_size, procs_per_node, num_nodes, num_gpus_per_model, read_group_size = get_parallel_info(args)
    
    save_output=True
    dir=f"out/{config['test']}"
    #TODO give unique filename
    f = create_results_file(config,global_rank, filename=f"{config['test']}-J{os.environ.get('SLURM_JOBID','0')}.csv", outdir=dir, save_output=save_output) #could do this in __init__ if it was a class

    for res in config["resolutions"]:
        gi = setup_grid_indices(res, read_group_size=read_group_size)
        #filter the list of datasets to just the matching resolutions
        datasets=[dataset for dataset in config["datasets"] if res in dataset]
        for dataset in datasets:
            for r in config["rollouts"]:
                #p0print(dataset)
                #os.system("source /ec/res4/hpcperm/naco/aifs/anemoi-dataloader-microbenchmark/darshan/enable-darshan")
                ds = create_dataset(dataset, grid_indices=gi, rollout=r, num_nodes=num_nodes, num_gpus_per_node=procs_per_node, num_gpus_per_model=num_gpus_per_model)
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
                                dl_iter = create_dataloader(ds, b=bs, w=nw, pf=pf, pin_mem=pm)
                                
                                try:
                                    results = run(dl_iter, nw, count=count, simulate_compute=False, proc_count=world_size, rollout=r, sleep_time=25)
                                    save_results(f, results, res, count, dataset, r, bs, pf, nw, pm, world_size, save_output=save_output)
                                except MemoryError:
                                    print("!!!OUT OF MEMORY!!!")
                                    pass

                                if save_output and config["monitor_memory"]:
                                    terminate_memory_monitor(mem_monitor_proc, csv, config["test"])
                                if config["per_worker_count"]:
                                    count=base_count
                                
    if save_output and f is not None:
        f.close()
        try:
            plot_anemoi_dataloader_benchmark(f.name, outdir=dir, outname=f"{config['test']}-j{os.environ.get('SLURM_JOBID','0')}-{int(time.time())}", header=f"({num_nodes}N, {procs_per_node}gpn, {num_gpus_per_model}gpm, {read_group_size}gpr)", plot_iter_per_s=True)
        except ValueError as err:
            p0print(f"Error plotting: {err}")
                                

if __name__ == "__main__":
    manager()
        
#TODO   add gpu support
#       measure time spent in HtoD copies
#       Split the 'anemoi' and benchmarking code into different files
#       Gather time spent at barrier for each proc and analyse them, maybe there's some repeast offenders
#       distinguish between per node and global throughput (think atos node has 480MB/s max BW)
#       Understand why the pytorch insufficient cpu core message pops up even when my job has enough cores allocated
#       Add requirements.txt
#       replace p0print with LOGGER
#       refactor plotting so all plots go under a dir for a run