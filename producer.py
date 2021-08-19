#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
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

    @staticmethod
    def logger_name(rank):
        return __name__ + f"::FrameBuffer.{rank}.log"


    def __init__(self, n_buf, dtype=np.float64, host=0):
        """
        """
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

        self.log = logging.getLogger(FrameBuffer.logger_name(self.rank))

        if self.rank == self.host:
            self.lock()
            self.ptr_set([0, 0, 0])
            self.unlock()


    def lock(self):
        """
        """
        self.win.Lock(rank=self.host, lock_type=MPI.LOCK_EXCLUSIVE)
        self.ptr.Lock(rank=self.host, lock_type=MPI.LOCK_EXCLUSIVE)

        self.log.debug("lock")


    def unlock(self):
        """
        """
        self.win.Unlock(rank=self.host)
        self.ptr.Unlock(rank=self.host)

        self.log.debug(f"unlock")


    def ptr_set(self, src):
        """
        """
        buf = np.empty(PTR_BUFF_SIZE, dtype=np.uint64)

        buf[:] = src[:]

        self.ptr.Put(buf, target_rank=self.host)

        self.log.debug(f"ptr_set {src=}")


    def ptr_incr(self, src):
        """
        """
        buf  = np.empty(PTR_BUFF_SIZE, dtype=np.uint64)
        incr = np.empty(PTR_BUFF_SIZE, dtype=np.uint64)

        incr[:] = src[:]
        self.ptr.Get_accumulate(incr, buf, target_rank=self.host)

        self.log.debug(f"ptr_incr {buf=} {incr=}")

        return buf


    def ptr_get(self):
        """
        """
        buf = np.empty(PTR_BUFF_SIZE, dtype=np.uint64)

        self.ptr.Get(buf, target_rank=self.host)

        self.log.debug(f"ptr_get {buf=}")

        return buf


    def buf_get(self, N, offset):
        """
        """
        buf = np.empty(N, dtype=self.np_dtype)

        # print(f"{self.rank=} taking {offset=}")
        self.win.Get(
            buf,
            target_rank = self.host,
            target      = (offset % self.n_buf, N, self.mpi_dtype)
        )

        self.log.debug(f"ptr_get {buf=}")

        return buf


    def buf_fill(self, src, offset):
        """
        """
        idx_remain = len(src) - offset
        idx_max = self.n_buf if idx_remain > self.n_buf else idx_remain
        self.log.debug(f"buf_fill {idx_max=} {offset=}")

        mem = np.frombuffer(self.win, dtype = self.np_dtype)
        mem[:idx_max] = src[offset:offset + idx_max]
        return idx_max


    def fence(self):
        """
        """
        self.win.Fence()
        self.ptr.Fence()


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


    def fill(self, src):

        if self.rank == 0:

            self.buf.lock()
            chunk = self.buf.buf_fill(src, 0)
            self.buf.ptr_set([0, chunk, len(src)])
            self.buf.unlock()

            # print("ptr_len has been set: " + str(len(src)))
            self.comm.Barrier()

            # self.buf.lock()
            # [src_offset, src_capacity, src_len] = self.buf.ptr_get()
            # # print(f"{self.rank=} peeking {src_offset=} {src_capacity=} {src_len=}")
            # self.buf.unlock()

            if chunk < len(src):
                while True:

                    self.buf.lock()
                    [src_offset, src_capacity, src_len] = self.buf.ptr_get()

                    if src_offset < src_capacity:
                        self.buf.unlock()
                        # print(f"waiting {src_offset=}, {src_capacity=}")
                        continue

                    idx_max = self.buf.buf_fill(src, chunk)
                    # self.buf.ptr_set(
                    #     [src_offset, src_capacity + idx_max, src_len]
                    # )
                    [src_offset, src_capacity, src_len] = self.buf.ptr_incr(
                        [0, idx_max, 0]
                    )
                    src_capacity += idx_max
                    chunk += idx_max
                    # print(f"refilled buffer: {src_offset=}, {src_capacity=}, {idx_max=}")
                    # [src_offset, src_capacity, src_len] = self.buf.ptr_get()
                    self.buf.unlock()

                    # print(f"refilled buffer: {src_offset=}, {src_capacity=}, {idx_max=}")

                    if chunk >= len(src):
                        # print("done!")
                        break
        else:
            self.comm.Barrier()


    def take(self, N):

        if self.rank > 0:

            # src_len = 0
            # while src_len == 0:
            #     # print(f"waiting for data {self.rank=}, {src_len=}")

            #     self.buf.lock()
            #     [src_offset, src_capacity, src_len] = self.buf.ptr_get()
            #     self.buf.unlock()

            while True:
                self.buf.lock()
                [src_offset, src_capacity, src_len] = self.buf.ptr_get()
                # self.buf.unlock()

                if src_offset >= src_len:
                    self.buf.unlock()
                    # print(f"Overrunning Src {src_offset=}, {src_len=}")
                    return None

                if src_offset >= src_capacity:
                    ## print(f"{self.rank=} peeking {src_offset=} {src_capacity=} {src_len=}")
                    self.buf.unlock()
                    continue
                    # self.buf.lock()
                    # [src_offset, src_capacity, src_len] = self.buf.ptr_get()
                    # self.buf.unlock()
                    # if src_offset >= src_len:
                    #     return None

                # self.buf.lock()
                # [src_offset, src_capacity, src_len] = self.buf.ptr_get()
                [src_offset, src_capacity, src_len] = self.buf.ptr_incr([N, 0, 0])
                buf = self.buf.buf_get(N, src_offset)
                # print(f"{self.rank=} taking {src_offset=}, {src_capacity=}, {src_len=}")
                # self.buf.ptr_set([src_offset + N, src_capacity, src_len])
                self.buf.unlock()

                return buf
