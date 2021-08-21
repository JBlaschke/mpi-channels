#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy              as     np
from   logging            import getLogger
from   mpi4py             import MPI
from   mpi4py.util.dtlib  import from_numpy_dtype
from   concurrent.futures import ThreadPoolExecutor


PTR_BUFF_SIZE = 3 # PTR, MAX, LEN


def make_win(dtype, n_buf, comm, host):
    """
    make_win(dtype, n_buf, comm, host)

    Create an MPI RMA window containing elements of MPI type `dtype`. The RMA
    window has length `n_buf`. It is accessible on MPI Communicator `comm`, and
    is hosted on rank `host`.
    """
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
        """
        FrameBuffer.logger_name(rank)

        Provide name of Logger for MPI rank `rank`.
        """
        return __name__ + f"::FrameBuffer.{rank}.log"


    def __init__(self, n_buf, n_mes, dtype=np.float64, host=0):
        """
        FrameBuffer(n_buf, n_mes, dtype=np.float64, host=0)

        Frame Buffers are a Message Queue. Each 'frame' refers to an
        ecapsulation of a message in the message queue (called a 'buffer').
        Each frame contains a counter representing the true message size, and a
        claim ID which ensures that the subsequent frames are claimed by the
        same MPI rank. This constructor preallocates MPI RMA window consisting
        of `n_buf` messages with a maximum size `n_mes`. Each message is
        `dtype`. The RMA buffer is hosted on rank `host`.

        A logger is available when the logging level is set to `logging.DEBUG`
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.host = host

        self.np_dtype  = np.dtype(
            [('end', np.uint64,), ('m', np.uint64,), ('f', dtype, (n_mes,))]
        )
        self.mpi_dtype = from_numpy_dtype(self.np_dtype)
        self.mpi_dtype.Commit()

        self.n_buf = n_buf
        self.n_mes = n_mes

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

        self.log = getLogger(FrameBuffer.logger_name(self.rank))

        if self.rank == self.host:
            self.lock()
            self._ptr_put((0, 0, 0))
            self.unlock()


    def __del__(self):
        self.mpi_dtype.Free()


    def lock(self):
        """
        lock()

        Lock the Frame Buffer's MPI RMA windows. Locks are MPI.LOCK_EXCLUSIVE.
        """
        self.win.Lock(rank=self.host, lock_type=MPI.LOCK_EXCLUSIVE)
        self.ptr.Lock(rank=self.host, lock_type=MPI.LOCK_EXCLUSIVE)

        self.log.debug("lock")


    def unlock(self):
        """
        unlock()

        Unlocks the Frame Buffer's MPI RMA windows.
        """
        self.win.Unlock(rank=self.host)
        self.ptr.Unlock(rank=self.host)

        self.log.debug(f"unlock")


    @property
    def idx(self):
        """
        Index to the current frame
        """
        return self._idx


    @property
    def max(self):
        """
        Maximum index in the current buffer
        """
        return self._max


    @property
    def len(self):
        """
        Maximum length of the all data entered into the buffer
        """
        return self._len


    def take(self):
        """
        take()

        Take the current frame (current means the frame at index `idx`) from
        the buffer, and increment `idx`.

        Requires a lock.
        """
        self.incr(1, 0, 0)
        # print(f"take {self.rank=} {self.idx=} {self.max=} {self.len=}")
        [buf] = self._buf_get(1, self.idx)
        # print(f"{buf=}")
        return buf['f'][:buf['end']]


    def put(self, src):
        """
        put(src)

        Place `src` at the end of the buffer (end means the frame at index
        `max`), and increment `max`.

        Requires a lock.
        """
        self.incr(0, 1, 0)
        # print(f"put {self.rank=} {self.idx=} {self.max=} {self.len=}")
        if self.rank == self.host:
            self._buf_set(src, self.max)
        else:
            buf = np.empty(1, dtype=self.np_dtype)
            buf[0]['end'] = len(src)
            buf[0]['f'][:len(src)] = src[:]

            # print(f"{buf=}")
            self._buf_put(buf, self.max)


    def sync(self):
        """
        sync()

        Syncronizes the local pointer states:
            * idx: index to the current frame
            * max: maximum index in the current buffer
            * len: maximum length of the all data entered into the buffer

        Requires a lock.
        """
        [self._idx, self._max, self._len] = self._ptr_get()


    def init(self, idx, idx_max, idx_len):
        """
        init(idx, idx_max, idx_len)

        Set the local pointer states and syncronize the MPI RMA windows.

        Requires a lock.
        """
        self._ptr_put((idx, idx_max, idx_len))


    def incr(self, idx, idx_max, idx_len):
        """
        incr()

        Increments the remote and local pointers.

        Requires a lock.
        """
        [self._idx, self._max, self._len] = self._ptr_incr(
            (idx, idx_max, idx_len)
        )


    def _ptr_put(self, src):
        """
        _ptr_put(src)

        Set the MPI RMA window to the state in src.
        """
        buf = np.empty(PTR_BUFF_SIZE, dtype=np.uint64)

        buf[:] = src[:]

        self.ptr.Put(buf, target_rank=self.host)

        self.log.debug(f"_ptr_put {src=}")


    def _ptr_incr(self, src):
        """
        _ptr_incr(src)

        Increment the state in the MPI RMA window by src.
        """
        buf  = np.empty(PTR_BUFF_SIZE, dtype=np.uint64)
        incr = np.empty(PTR_BUFF_SIZE, dtype=np.uint64)

        incr[:] = src[:]
        self.ptr.Get_accumulate(incr, buf, target_rank=self.host)

        self.log.debug(f"_ptr_incr {buf=} {incr=}")

        return buf


    def _ptr_get(self):
        """
        _ptr_get():

        Read the state in the MPI RMA window.
        """
        buf = np.empty(PTR_BUFF_SIZE, dtype=np.uint64)

        self.ptr.Get(buf, target_rank=self.host)

        self.log.debug(f"_ptr_get {buf=}")

        return buf


    def _buf_get(self, N, offset):
        """
        _buf_get(N, offset)

        Get `N` frames starting at index `offset`
        """
        buf = np.empty(N, dtype=self.np_dtype)

        self.win.Get(
            [buf, self.mpi_dtype],
            target_rank = self.host,
            target      = (offset % self.n_buf, N, self.mpi_dtype)
        )

        self.log.debug(f"_buf_get {offset=}")

        return buf


    def _buf_set(self, src, idx):
        """
        _buf_set(src, idx)

        Set `src` at the location at `idx`.
        """
        mem = np.frombuffer(self.win, dtype = self.np_dtype)
        # print(f"{mem=}, {src=}, {idx=}, {self.n_buf=} {int(idx) % self.n_buf}")
        mem[int(idx % self.n_buf)]['end'] = len(src)
        mem[int(idx % self.n_buf)]['f'][:len(src)] = src[:]

        self.log.debug(f"_buf_set {idx=}")


    def _buf_put(self, src, offset):
        """
        _buf_put(src)

        Put `src` into the MPI RMA window to the state at `offset`.
        """
        buf = np.empty(len(src), dtype=self.np_dtype)

        buf[:] = src[:]

        self.win.Put(
            [buf, self.mpi_dtype],
            target_rank = self.host,
            target      = (offset % self.n_buf, len(src), self.mpi_dtype)
        )

        self.log.debug(f"_buf_put {offset=}")


    def _buf_fill(self, src, offset):
        """
        _buf_fill(src, offset)
        returns idx_max = min(n_buf, len(src) - offset

        Sequentially fill the local buffer from the beginning to `idx_max`
        using elements from `src` starting at `offset`.
        """
        idx_remain = len(src) - offset
        idx_max = self.n_buf if idx_remain > self.n_buf else idx_remain

        mem = np.frombuffer(self.win, dtype = self.np_dtype)
        mem[:idx_max] = src[offset:offset + idx_max]

        self.log.debug(f"_buf_fill {idx_max=} {offset=}")
        return idx_max


    def fence(self):
        """
        fence()

        Place a Fence into the MPI RMA Windows.
        """
        self.win.Fence()
        self.ptr.Fence()


class RemoteChannel(object):

    def __init__(self, n_buf, n_mes, dtype=np.float64, host=0, n_fpool=4):
        """
        RemoteChannel(n_buf, n_mes, dtype=np.float64, host=0, n_fpool=4)

        Create a RemoteChannel containing at most `n_buf` messages (with
        maximum message size `n_mes`). Messages are arrays of type `dtype`. The
        message buffers are hosted in MPI RMA windows on rank `host`.

        The RemoteChannel's internal state is held in a FrameBuffer object (in
        the `buf` attribute).

        The RemoteChannel uses a ThreadPoolExecutor to handle futures,
        `n_fpool` sets the size of this pool. Future objects resulting from
        calls to `putf` are stored in `put_futures`, and Future instances
        resulting from calles to `takef` are stored in `take_futures`.
        """
        self.comm  = MPI.COMM_WORLD
        self.rank  = self.comm.Get_rank()
        self.dtype = np.float64
        self.host  = host

        self.buf = FrameBuffer(n_buf, n_mes, dtype=self.dtype, host=self.host)

        self.pool         = ThreadPoolExecutor(5)
        self.put_futures  = list()
        self.take_futures = list()

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
            self.buf.lock()
            self.buf.sync()

            # Check if there is space in the buffer for new elements. If the
            # buffer is full, spin and watch for space
            # print(f"putting {self.buf.idx=} {self.buf.max=} {self.buf.len=}")
            if self.buf.max - self.buf.idx >= self.buf.n_buf:
                self.buf.unlock()
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
            self.buf.lock()
            self.buf.sync()

            if self.buf.idx >= self.buf.len:
                self.buf.unlock()
                # print(f"Overrunning Src {self.buf.idx=}, {self.buf.len=}")
                return None

            if self.buf.idx >= self.buf.max:
                # print(f"{self.rank=} peeking {src_offset=} {src_capacity=} {src_len=}")
                self.buf.unlock()
                continue

            buf = self.buf.take()
            # print(f"{self.rank=} taking {src_offset=}, {src_capacity=}, {src_len=}")
            self.buf.unlock()

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

