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

N = 10
win_size = N * itemsize if rank == 0 else 0
win = MPI.Win.Allocate(win_size, comm=comm)

buf = np.empty(N, dtype=np_dtype)
buf.fill(-1)

if rank == 0:
    buf.fill(42)
    win.Lock(rank=0)
    win.Put([buf, MPI.FLOAT], target_rank=0)
    win.Unlock(rank=0)
    comm.Barrier()
else:
    comm.Barrier()
    incr = np.array(list(range(N)), dtype=np.float32)
    win.Lock(rank=0)
    print(f"{rank=} {buf=} {incr=}")
    win.Get_accumulate(incr, buf, target_rank=0)
    print(f"{rank=} {buf=} {incr=}")
    win.Unlock(rank=0)

comm.Barrier()
if rank == 0:
    mem = np.frombuffer(win, dtype=np.float32)
    print(f"{rank=} {mem=}")
