#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"{rank=}")

datatype = MPI.FLOAT
np_dtype = np.float32
itemsize = datatype.Get_size()

N = comm.Get_size() + 1
print(f"{N=}")
win_size = N * itemsize if rank == 0 else 0
win = MPI.Win.Allocate(
    size=win_size,
    disp_unit=itemsize,
    comm=comm,
)
if rank == 0:
    mem = np.frombuffer(win, dtype=np_dtype)
    mem[:] = np.arange(len(mem), dtype=np_dtype)
    print(f"{mem=}")
comm.Barrier()

buf = np.zeros(3, dtype=np_dtype)
target = (rank, 2, datatype)
win.Lock(rank=0)
win.Get(buf, target_rank=0, target=target)
win.Unlock(rank=0)

comm.Barrier()

print(f"{rank=}, {buf=}")

assert np.all(buf == [rank, rank+1, 0])
