#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


buf = np.empty(3, dtype=[('a', '<u8'), ('b', '<f8')])  # use *numpy.empty*

mpitype = from_numpy_dtype(buf.dtype)  # Convert numpy.dtype -> MPI.Datatype()
mpitype.Commit()  # commit the datatype before using it in communication calls

itemsize = mpitype.Get_size()

N = comm.Get_size() + 1
win_size = N * itemsize if rank == 0 else 0
win = MPI.Win.Allocate(
    size=win_size,
    disp_unit=itemsize,
    comm=comm,
)

if rank == 0:
    win.Lock(rank=0)
    mem = np.frombuffer(win, dtype=buf.dtype)
    for i in range(N):
        mem[i] = (i, (10+i)/2)
    win.Unlock(rank=0)
comm.Barrier()

win.Lock(rank=0)
win.Get([buf, mpitype], target_rank=0)
win.Unlock(rank=0)

comm.Barrier()
print(f"{rank=}, {buf=}")

mpitype.Free()  # mpi4py does not free MPI handles, do it yourself or you'll have a leak.
