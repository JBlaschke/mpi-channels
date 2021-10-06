#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy              as     np
from   logging            import getLogger
from   mpi4py             import MPI
from   concurrent.futures import ThreadPoolExecutor

from . import FrameBuffer, Schedule



class RemoteChannel(object):

    def __init__(self,
            n_buf, n_mes, dtype=np.float64, host=0,
            s_threashold=4, t_wait=0, r_rwait=1):
        """
        RemoteChannel(n_buf, n_mes, dtype=np.float64, host=0,
                      s_threashold=4, t_wait=0, r_rwait=1)

        Create a RemoteChannel containing at most `n_buf` messages (with
        maximum message size `n_mes`). Messages are arrays of type `dtype`. The
        message buffers are hosted in MPI RMA windows on rank `host`.

        The RemoteChannel's internal state is held in a FrameBuffer object (in
        the `buf` attribute).

        The RemoteChannel uses a ThreadPoolExecutor (of size 1) to handle
        futures, The size of this pool needs to be limited to prevent multiple
        threads locking each other out of the MPI RMA buffer. For this reason
        we choose a size of 1. Future objects resulting from calls to `putf`
        are stored in `put_futures`, and Future instances resulting from calles
        to `takef` are stored in `take_futures`.
        """
        self.comm  = MPI.COMM_WORLD
        self.rank  = self.comm.Get_rank()
        self.dtype = dtype
        self.host  = host

        self.buf = FrameBuffer(n_buf, n_mes, dtype=self.dtype, host=self.host)

        self.pool         = ThreadPoolExecutor(1)
        self.put_futures  = list()
        self.take_futures = list()

        self.schedule = Schedule(threshold=128)


    def put(self, src):
        """
        put(src)

        Put a message `src` at the end of the RemoteChannel, where it can be
        retrieved using `take`. `src` must have a length. The size of `src`
        cannot exceed `n_mes`, but may be shorter (if it is shorter, padding to
        the size `n_msg` will be transmitted via MPI).

        Blocks if buffer is full. For non-blocking version see `putf`
        """
        while True:
            # print(f"{self.rank=} start putting 1 {self.buf.idx=} {self.buf.max=} {self.buf.len=}", flush=True)
            self.buf.lock()
            # print(f"{self.rank=} start putting 2", flush=True)
            self.buf.sync()
            # print(f"{self.rank=} start putting 3", flush=True)

            # Check if there is space in the buffer for new elements. If the
            # buffer is full, spin and watch for space
            # print(f"putting {self.buf.idx=} {self.buf.max=} {self.buf.len=}")
            if self.buf.max - self.buf.idx >= self.buf.n_buf:
                self.buf.unlock()
                # print(f"PUT {self.buf.max=} {self.buf.idx=} {self.buf.n_buf=}", flush=True)
                self.schedule()
                continue

            self.buf.put(src)
            self.buf.unlock()
            return


    def putf(self, src):
        """
        putf(src)
        returns Future(None)

        Submits the `put(src)` call to the Executor, and stores the future in
        `put_futures`.

        Non-blocking, returns a Future.
        """
        self.put_futures.append(
            self.pool.submit(self.put, src)
        )

        return self.put_futures[-1]


    def claim(self, N):
        """
        claim(N)

        Reserve `N` slots in the RemoteChannel's FrameBuffer.

        Note: `N` < `n_buf`. Only `host` can claim space in the buffer, calls
        to `claim` from ranks other than `host` are ignored.
        """
        if self.rank == self.host:
            self.buf.lock()
            self.buf.incr(0, 0, N)
            self.buf.sync()
            # print(f"claim: {self.buf.idx=} {self.buf.max=} {self.buf.len=}")
            self.buf.unlock()
        else:
            return


    def take(self):
        """
        take()
        returns src (a message -- i.e. `np.array` -- of length at most `n_mes`)
        returns None if the internal buffer contains no more elements

        Take a message `src` from the current location (`buf.idx`) of the
        RemoteChannel, where it had ben placed using `put`. `src` is of type
        `np.array` with variable length between 1 and `n_mes` (note that MPI
        communication is padded up to the size `n_mes`).

        `take` can return `None` if there are no more messages in the buffer.
        The RemoteChannel uses the total claimed by `claim` to determine if it
        should wait for more messages.

        Blocks if buffer is empty. For non-blocking version see `takef`
        """
        while True:
            # print(f"{self.rank=} start taking 1 {self.buf.idx=} {self.buf.max=} {self.buf.len=}", flush=True)
            self.buf.lock()
            # print(f"{self.rank=} start taking 2", flush=True)
            self.buf.sync()
            # print(f"{self.rank=} start taking 3", flush=True)

            if self.buf.idx >= self.buf.len:
                self.buf.unlock()
                # print(f"TAKE {self.buf.idx=}, {self.buf.len=}", flush=True)
                return None

            if self.buf.idx >= self.buf.max:
                self.buf.unlock()
                # print(f"TAKE {self.rank=} peeking {self.buf.idx=} {self.buf.max=} {self.buf.len=}", flush=True)
                self.schedule()
                continue

            buf = self.buf.take()
            # print(f"{self.rank=} taking", flush=True)
            # print(f"{self.rank=} taking {src_offset=}, {src_capacity=}, {src_len=}")
            self.buf.unlock()
            # print(f"{self.rank=} done taking", flush=True)

            return buf


    def takef(self):
        """
        takef()
        returns Future(src)

        Submits the `src = take()` call to the Executor, and stores the future
        in `take_futures`.
        
        Non-blocking, returns a Future.
        """
        self.take_futures.append(
            self.pool.submit(self.take)
        )

        return self.take_futures[-1]
