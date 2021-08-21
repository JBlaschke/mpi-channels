#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy    as     np
from   mpi4py   import MPI
from   logging  import DEBUG, basicConfig
from   argparse import ArgumentParser
from   random   import random
from   time     import sleep

from producer import Producer, make_win


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = ArgumentParser()
parser.add_argument("--logging", type=bool, nargs=1, default=False)
args, _ = parser.parse_known_args()


buff_size = 10
data_size = 30
message_size = 10
vector_size  = 6

producer = Producer(buff_size, message_size)
result   = Producer(data_size, 1)
result.claim(data_size)

p_sum_win = make_win(
    dtype = MPI.DOUBLE,
    n_buf = 1,
    comm  = comm,
    host  = 0
)

if args.logging:
    producer.buf.log.setLevel(DEBUG)
    basicConfig(filename=f"{rank:0>3}.log", level=DEBUG)

if rank == 0:
    data = np.random.rand(data_size, vector_size)
    producer.claim(len(data))
    p_sum = 0.
    for elt in data:
        producer.put(elt)
        p_sum += np.sum(elt)
    print(f"{rank=} {p_sum=}")


if rank > 0:
    res = 0
    p_sum = 0.
    for i in range(data_size):
        p = producer.take(1)
        # sleep(random())
        if p is not None:
            # print(f"{rank=}, {i=}, {p=}")
            sp = np.sum(p)
            # print(f"{rank=} putting")
            result.put((sp,))
            # print(f"{rank=} done putting")
            p_sum += sp
            res += 1

    print(f"{rank=} {res=}")
    p_sum_win.Lock(rank=0)
    buf = np.empty(1, dtype=np.float64)
    incr = np.empty(1, dtype=np.float64)
    incr[0] = p_sum
    p_sum_win.Get_accumulate(incr, buf, target_rank=0)
    p_sum_win.Unlock(rank=0)

comm.Barrier()

if rank == 0:
    p_sum_win.Lock(rank=0)
    p_sum_buf = np.frombuffer(p_sum_win, dtype=np.float64)
    p_sum_win.Unlock(rank=0)

    p_sum_r = 0
    for i in range(data_size):
        sp = result.take(1)
        p_sum_r += sp[0]
        print(f"{rank=} {sp=}")

    print(f"{rank=} {p_sum_buf=} : {p_sum - p_sum_buf[0]}")
    print(f"{rank=} {p_sum_r=} : {p_sum - p_sum_r}")
