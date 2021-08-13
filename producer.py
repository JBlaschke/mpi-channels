#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy  as     np
from   mpi4py import MPI


def make_win(dtype, n_buf, comm, host):
    itemsize = dtype.Get_size()
    win_size = n_buf * itemsize if comm.Get_rank() == host else 0
    return MPI.Win.Allocate(
        size      = win_size,
        disp_unit = itemsize,
        comm      = comm
    )


def ptr_set(win, src, host):
    buf = np.array([src])
    win.Lock(rank = host)
    win.Put(buf, target_rank = host)
    win.Unlock(rank = host)


def ptr_peek(win, host):
    buf = np.array([0])
    win.Lock(rank = host)
    win.Get(buf, target_rank = host)
    win.Unlock(rank = host)
    return buf[0]


def ptr_incr(win, n, host):
    buf = np.array([0])
    win.Lock(rank = host)
    win.Get(buf, target_rank = host)
    ptr    = buf[0]
    buf[0] = buf[0] + n
    win.Put(buf, target_rank = host)
    win.Unlock(rank = host)
    return ptr


def fill_buffer(buf, src, offset):
    idx_max = len(buf) if len(src) - offset > len(buf) else len(src)
    buf[:idx_max] = src[offset:offset + idx_max]
    return idx_max


class Producer(object):

    def __init__(self, n_buf, single=False):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.mpi_dtype = MPI.DOUBLE
        self.np_dtype  = np.float64
        if single:
            self.mpi_dtype = MPI.FLOAT
            self.np_dtype = np.float32

        self.n_buf = n_buf

        self.win = make_win(
            dtype = self.mpi_dtype,
            n_buf = self.n_buf,
            comm  = self.comm,
            host  = 0
        )

        self.ptr = make_win(
            dtype = MPI.INT,
            n_buf = 1,
            comm  = self.comm,
            host  = 0
        )


    def fill(self, src):

        if self.rank == 0:

            ptr_set(win = self.ptr, src = 0, host = 0)

            self.win.Lock(rank = 0)
            mem = np.frombuffer(self.win, dtype=self.np_dtype)
            idx_max = fill_buffer(mem, src, offset = 0)
            self.win.Unlock(rank = 0)

            if idx_max < len(src):
                while True:

                    src_offset = ptr_peek(win = self.ptr, host = 0)
                    if src_offset < len(src):
                        continue
         
                    self.win.Lock(rank = 0)
                    idx_max = fill_buffer(mem, src, src_offset = src_offset)
                    self.win.Unlock(rank = 0)

                    if idx_max <= len(src):
                        break


    def take(self, N):

        if self.rank > 0:

            src_offset = ptr_incr(win = self.ptr, n = N, host = 0)

            buf = np.zeros(dtype=self.np_dtype)

            self.win.Lock(rank = 0)
            win.Get(
                buf,
                target_rank = 0,
                target = (src_offset, N, self.mpi_dtype)
            )
            self.win.Unlock(rank = 0)
