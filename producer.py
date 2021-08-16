#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy             as     np
from   mpi4py            import MPI
from   mpi4py.util.dtlib import from_numpy_dtype


PTR_BUFF_SIZE = 3 # PTR, MAX, LEN


def make_win(dtype, n_buf, comm, host):
    itemsize = dtype.Get_size()
    win_size = n_buf * itemsize if comm.Get_rank() == host else 0
    return MPI.Win.Allocate(
        size      = win_size,
        disp_unit = itemsize,
        comm      = comm
    )



class FrameBuffer(object):

    def __init__(self, n_buf, dtype=np.float64, host=0):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.host = host

        self.np_dtype  = dtype
        self.mpi_dtype = from_numpy_dtype(dtype)

        self.n_buf = n_buf

        self.win = make_win(
            dtype = self.mpi_dtype,
            n_buf = self.n_buf,
            comm  = self.comm,
            host  = host
        )

        self.ptr = make_win(
            dtype = from_numpy_dtype(np.uint64),
            n_buf = PTR_BUFF_SIZE,
            comm  = self.comm,
            host  = host
        )


    def lock(self):
        self.win.Lock(rank=self.host)
        self.ptr.Lock(rank=self.host)


    def unlock(self):
        self.win.Unlock(rank=self.host)
        self.ptr.Unlock(rank=self.host)


    def ptr_set(self, src):
        buf = np.empty(PTR_BUFF_SIZE, dtype=np.uint64)

        buf[:] = src[:]

        self.ptr.Put(buf, target_rank=self.host)


    def ptr_peek(self):
        buf = np.empty(PTR_BUFF_SIZE, dtype=np.uint64)

        self.ptr.Get(buf, target_rank=self.host)

        return buf


    def buf_get(self, N, offset):
        buf = np.empty(N, dtype=self.np_dtype)

        # print(f"{self.rank=} taking {offset=}")
        self.win.Get(
            buf,
            target_rank = self.host,
            target      = (offset % self.n_buf, N, self.mpi_dtype)
        )

        return buf


    def fill_buffer(self, src, offset):
        idx_max = self.n_buf if len(src) - offset > self.n_buf else len(src) - offset
        # print(f"{idx_max=}, {offset=}")
        mem = np.frombuffer(self.win, dtype = self.np_dtype)
        # print(f"{mem=}")
        mem[:int(idx_max)] = src[int(offset):int(offset + idx_max)]
        # print(f"{mem=}")
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

        self.buf = FrameBuffer(n_buf, dtype=self.np_dtype, host=0)
        self.buf.lock()
        self.buf.ptr_set([0, 0, -1])
        self.buf.unlock()


    def fill(self, src):

        if self.rank == 0:

            self.buf.lock()
            idx_max = self.buf.fill_buffer(src, 0)
            self.buf.ptr_set([0, idx_max, len(src)])
            self.buf.unlock()

            # print("ptr_len has been set: " + str(len(src)))

            if idx_max < len(src):
                while True:

                    self.buf.lock()
                    [src_offset, src_capacity, src_len] = self.buf.ptr_peek()

                    if src_offset < src_capacity:
                        self.buf.unlock()
                        # print(f"waiting {src_offset=}, {src_capacity=}")
                        continue

                    idx_max = self.buf.fill_buffer(src, src_offset)
                    self.buf.ptr_set(
                        [src_offset, src_capacity + idx_max, src_len]
                    )
                    self.buf.unlock()

                    # print(f"refilled buffer: {src_offset=}, {idx_max=}")

                    if src_offset + idx_max >= len(src):
                        # print("done!")
                        break


    def take(self, N):

        if self.rank > 0:

            src_len = -1
            while src_len < 0:
                # print(f"waiting for data {self.rank=}, {src_len=}")

                self.buf.lock()
                [src_offset, src_capacity, src_len] = self.buf.ptr_peek()
                self.buf.unlock()

            self.buf.lock()
            [src_offset, src_capacity, src_len] = self.buf.ptr_peek()
            self.buf.unlock()

            if src_offset >= src_len:
                # print(f"Overrunning Src {src_offset=}, {src_len=}")
                return None

            while src_offset > src_capacity:
                # print(f"{self.rank=} peeking {src_offset=} {src_capacity=}")
                self.buf.lock()
                [src_offset, src_capacity, src_len] = self.buf.ptr_peek()
                self.buf.unlock()

            self.buf.lock()
            [src_offset, src_capacity, src_len] = self.buf.ptr_peek()
            buf = self.buf.buf_get(N, src_offset)
            self.buf.ptr_set([src_offset + N, src_capacity, src_len])
            self.buf.unlock()

            return buf
