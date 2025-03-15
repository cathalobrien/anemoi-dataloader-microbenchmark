# anemoi-dataloader-microbenchmark
Small code which extracts just the dataloader logic from anemoi training, and adds wrapper code to benchmark it

# Running the benchmark
Currently you need to have graphs in the format
```bash
graphs/$RES.graph
# e.g. graphs/n320.graph
```
```python
python main.py
```
there are a few different benchmarking tests defined in main.py ('single worker, carying rollout', 'increasing workers')
Currently the benchmark only supports a single driver process (equivalent to running on a single node with 1 GPU)
You can run the code over multiple processes/nodes by launching with mpirun. This will change the IO behaviour, as currently all processes are added to the same read group and the batch is split across processes.
```bash
srun --mem=0 --qos=np -N 1 --ntasks-per-node=4 --cpus-per-task=32 --time=2:00:00 --pty bash
mpirun -np 4 python main.py
```
By default, the memory monitor will run in the background for each benchmark, and plot will be produced under 'out/'.

## Example output

```
Creating Dataloader (batch size: 1, num_workers: 2, prefetch_factor 1, pin_mem=True, seed=1742057691)...
Dataloader created.
Proc 0: Starting 8 loads with r=1, bs=1, pf=1, nw=2, pm=True
Iteration 0
p0: 3.7GB batch loaded in 26.68s (0.00s in barrier). 
p0: dataloader throughput: 143.0MB/s
Iteration 1
p0: 3.7GB batch loaded in 0.00s (0.00s in barrier). 
Iteration 2
p0: 3.7GB batch loaded in 29.03s (0.00s in barrier). 
p0: dataloader throughput: 131.4MB/s
Iteration 3
p0: 3.7GB batch loaded in 8.32s (8.07s in barrier). 
p0: dataloader throughput: 458.6MB/s
Iteration 4
p0: 3.7GB batch loaded in 27.35s (4.75s in barrier). 
p0: dataloader throughput: 139.5MB/s
Iteration 5
p0: 3.7GB batch loaded in 9.60s (9.60s in barrier). 
p0: dataloader throughput: 397.5MB/s
Iteration 6
p0: 3.7GB batch loaded in 27.20s (11.18s in barrier). 
p0: dataloader throughput: 140.2MB/s
Iteration 7
p0: 3.7GB batch loaded in 9.55s (9.55s in barrier). 
p0: dataloader throughput: 399.4MB/s
Av time: 17.21s, Total time for 8 runs: 137.72s
per-worker BW: 110.8MB/s,  per-process BW: 221.6MB/s, global BW: 443.1MB/s
Compute throughput must be >= 0.058it/s (221.6MB/s / 3.7GB) to avoid starvation
Est. latency to load the initial batch: 34.43s
```

# Errors
If you encounter this error:
```
  ompi_mpi_init: ompi_rte_init failed
  --> Returned "No permission" (-17) instead of "Success" (0)
```
you can fix it with
```bash
export PMIX_MCA_gds=hash
```
