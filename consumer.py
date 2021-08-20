#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy    as     np
from   mpi4py   import MPI
from   logging  import DEBUG, basicConfig
from   argparse import ArgumentParser

from producer import Producer, make_win


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = ArgumentParser()
parser.add_argument("--logging", type=bool, nargs=1, default=False)
args, _ = parser.parse_known_args()


buff_size = 10
data_size = 30
message_size = 10

producer = Producer(buff_size, message_size)


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
    data = np.random.rand(data_size, 6)
    # producer.fill(data)
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
        if p is not None:
            print(f"{rank=}, {i=}, {p=}")
            p_sum += np.sum(p)
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
    print(f"{rank=} {p_sum_buf=} : {p_sum - p_sum_buf[0]}")
