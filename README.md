# anemoi-dataloader-microbenchmark
Small code which extracts just the dataloader logic from anemoi training, and adds wrapper code to benchmark it

# Running the benchmark
```python
python main.py
```
there are a few different benchmarking tests defined in main.py ('single worker, carying rollout', 'increasing workers')
Currently the benchmark only supports a single driver process (equivalent to running on a single node with 1 GPU)
You can run the code over multiple processes/nodes by launching with mpirun. This will change the IO behaviour, as currently all processes are added to the same read group and the batch is split across processes.
```bash
mpirun -np 4 python main.py
```
By default, the memory monitor will run in the background for each benchmark, and plot will be produced under 'out/'.

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