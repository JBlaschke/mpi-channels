#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy       as     np
from   mpi4py      import MPI
from   mpi4py.util import dtlib


class Producer(object):

    def __init__(self, n_buff, single=False):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.datatype = MPI.DOUBLE
        if single:
            self.datatype = MPI.FLOAT

        self.np_dtype = dtlib.to_numpy_dtype(datatype)
        self.itemsize = datatype.Get_size()

        self.win_size = n_buff * itemsize if rank == 0 else 0
        self.win      = MPI.Win.Allocate(self.win_size, comm=self.comm)
        self.buf      = np.empty(n_buff, dtype=self.np_dtype)

        self.ptr = MPI.Win.Allocate(MPI.INT, comm=self.comm)


    def fill(self, src):

        if self.rank == 0:
            self.buf.fill(0)

            self.ptr.Lock(rank=0)
            src_offset = np.array([0])
            self.ptr.Put(src_offset, target_rank=0)
            self.ptr.Unlock(rank=0)
            while True:

                self.ptr.Lock(rank=0)
                src_offset = np.array([0])
                self.ptr.Get(src_offset, target_rank=0)
                self.ptr.Unlock(rank=0)
     
                idx_max = self.win_size if len(src) - src_offset > self.win_size else len(src)
                self.buf[:idx_max] = src[src_offset:src_offset+idx_max]

                self.win.Lock(rank=0)
                self.win.Put(buf, target_rank=0)
                win.Unlock(rank=0)
                comm.Barrier()


    def take(self)

        if self.rank > 0:
            comm.Barrier()
            win.Lock(rank=0)
            win.Get(buf, target_rank=0)
            win.Unlock(rank=0)
            assert np.all(buf == 42)
