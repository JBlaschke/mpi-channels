#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype


def define_struct_type(field_names, field_dtypes):
    dtypes = list(zip(field_names, field_dtypes))
    a = np.array([0], dtype=dtypes)

    struct_size = a.nbytes
    offsets = [ a.dtype.fields[field][1] for field in field_names ]

    field_mpi_types = [from_numpy_dtype(dtype) for dtype in field_dtypes]

    print(f"{field_names=} {offsets=} {field_mpi_types=}")
    struct_type = MPI.Datatype.Create_struct(
        [1]*len(field_names), offsets, field_mpi_types
    )
    struct_type = struct_type.Create_resized(0, struct_size)
    struct_type.Commit()

    return struct_type, a.dtype


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

datatype, np_dtype = define_struct_type(["a", "b"], [np.uint64, np.float32])
itemsize = datatype.Get_size()

N = comm.Get_size() + 1
win_size = N * itemsize if rank == 0 else 0
win = MPI.Win.Allocate(
    size=win_size,
    disp_unit=from_numpy_dtype(np.int16).Get_size(),
    comm=comm,
)
if rank == 0:
    win.Lock(rank=0)
    mem = np.frombuffer(win, dtype=np_dtype)
    for i in range(N):
        mem[i] = (i, 10+i)
    win.Unlock(rank=0)
    print(f"{mem=}")
comm.Barrier()

buf = np.zeros(1, dtype=np.float32)
target = (10, 2, from_numpy_dtype(np.uint16))
win.Lock(rank=0)
win.Get(buf, target_rank=0, target=target)
win.Unlock(rank=0)

comm.Barrier()

print(f"{rank=}, {buf=}")
