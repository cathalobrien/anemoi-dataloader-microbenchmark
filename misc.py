import os

def maybe_import_mpi():
    comm=None
    USE_MPI=True #If you want to disable MPI while having it installed e.g. to use Darshan
    if os.path.basename(os.getenv("LD_PRELOAD", "")) == "libdarshan.so" and os.getenv("DARSHAN_ENABLE_NONMPI", "") != "":
        print("Darshan detected, Disabling MPI... (This removes a barrier after each load, and an all-gather after the run to aggregate the statistics)")
        USE_MPI=False
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
        
    return MPI4PY_AVAILABLE, comm

MPI4PY_AVAILABLE, comm = maybe_import_mpi()

def format(size):
    for unit in ('', 'K', 'M', 'G'):
        if size < 1024:
            break
        size /= 1024.0
    return "%.1f%s" % (size, unit)


def p0print(str):
    #I dont love reading an env var for every print
    if MPI4PY_AVAILABLE:
        if comm.Get_rank() == 0:
            print(str)
    else:
        if int(os.environ.get("SLURM_PROCID", 0)) == 0:
            print(str)
