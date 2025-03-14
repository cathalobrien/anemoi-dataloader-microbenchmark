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

# Monitoring dataloader memory
To monitor dataloader memory usage during the benchmark you can run the following on a different session on the same node
```python
python memory_monitor.py
python plot.py #to plot the resulting csv file
```
This will show a table of memory usage during the run, and will produce a csv file which can be plotted.
