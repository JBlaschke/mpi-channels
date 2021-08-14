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
    idx_max = len(buf) if len(src) - offset > len(buf) else len(src) - offset
    print(f"{idx_max=}, {offset=}")
    buf[:idx_max] = src[offset:offset + idx_max]
    print(f"{buf=}")
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

        self.ptr_max = make_win(
            dtype = MPI.INT,
            n_buf = 1,
            comm  = self.comm,
            host  = 0
        )

        self.ptr_len = make_win(
            dtype = MPI.INT,
            n_buf = 1,
            comm  = self.comm,
            host  = 0
        )

        ptr_set(win = self.ptr,     src =  0, host = 0)
        ptr_set(win = self.ptr_max, src =  0, host = 0)
        ptr_set(win = self.ptr_len, src = -1, host = 0)


    def fill(self, src):

        if self.rank == 0:

            ptr_set(win = self.ptr,     src = 0,        host = 0)
            ptr_set(win = self.ptr_len, src = len(src), host = 0)
            print("ptr_len has been set: " + str(len(src)))

            self.win.Lock(rank = 0)
            mem = np.frombuffer(self.win, dtype = self.np_dtype)
            idx_max = fill_buffer(mem, src, offset = 0)
            self.win.Unlock(rank = 0)

            ptr_set(win = self.ptr_max, src = idx_max, host = 0)

            if idx_max < len(src):
                while True:

                    src_offset = ptr_peek(win = self.ptr, host = 0)
                    src_capacity = ptr_peek(win = self.ptr_max, host = 0)
                    if src_offset < src_capacity:
                        print(f"waiting {src_offset=}, {src_capacity=}")
                        continue

                    self.win.Lock(rank = 0)
                    mem = np.frombuffer(self.win, dtype = self.np_dtype)
                    idx_max = fill_buffer(mem, src, offset = src_offset)
                    self.win.Unlock(rank = 0)

                    ptr_set(
                        win  = self.ptr_max,
                        src  = src_offset + idx_max,
                        host = 0
                    )

                    print(f"refilled buffer: {src_offset=}, {idx_max=}")

                    if src_offset + idx_max >= len(src):
                        print("done!")
                        break


    def take(self, N):

        if self.rank > 0:

            src_len = -1
            while src_len < 0:
                print(f"waiting for data {self.rank=}, {src_len=}")
                src_len = ptr_peek(win = self.ptr_len, host = 0)
                sleep(0.1)

            src_offset = ptr_incr(win = self.ptr, n = N, host = 0)

            if src_offset >= src_len:
                print(f"Overrunning Src {src_offset=}, {src_len=}")
                return None

            src_capacity = ptr_peek(win = self.ptr_max,  host = 0)

            while src_offset > src_capacity:
                print(f"{self.rank=} peeking {src_offset=} {src_capacity=}")
                src_capacity = ptr_peek(win = self.ptr_max,  host = 0)

            buf = np.zeros(N, dtype=self.np_dtype)

            print(f"{self.rank=} taking {src_offset=}")
            self.win.Lock(rank = 0)
            self.win.Get(
                buf,
                target_rank = 0,
                target = (src_offset % self.n_buf, N, self.mpi_dtype)
            )
            self.win.Unlock(rank = 0)

            return buf
