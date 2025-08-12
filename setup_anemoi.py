#   Copyright 2024-2025 Anemoi Contributors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from anemoi.datasets.data import open_dataset
from anemoi.training.data.dataset import NativeGridDataset
from anemoi.training.utils.worker_init import worker_init_func as  default_worker_init_func
from anemoi.training.data.grid_indices import FullGrid
from anemoi.utils.dates import frequency_to_seconds

from torch_geometric.data import HeteroData
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from collections.abc import Callable
import os
import time

from misc import maybe_import_mpi, p0print

MPI4PY_AVAILABLE, comm = maybe_import_mpi()
            
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
 
######## ANEMOI CODE BEGINS #############
training_batch_size=1
multistep_input=2
frequency="6h"
timestep="6h"
frequency_s = frequency_to_seconds(frequency)
timestep_s = frequency_to_seconds(timestep)
time_inc = timestep_s // frequency_s

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
                open_dataset(dataset=dataset, start=None, end=None, frequency=frequency, drop=[]),
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
            #multiprocessing_context="fork" #fork is the default, spawn and forkserver crash with an MPI error: '"No permission" (-17) instead of "Success" '
    )
    dl_iter=iter(dataloader)
    p0print(f"Dataloader created in {time.time() - start_time:.2f}s.")
    return dl_iter

######## ANEMOI CODE OVER #############
