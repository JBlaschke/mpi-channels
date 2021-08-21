#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy              as     np
from   mpi4py             import MPI
from   logging            import DEBUG, basicConfig
from   argparse           import ArgumentParser
from   random             import random
from   time               import sleep

from producer import RemoteChannel


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = ArgumentParser()
parser.add_argument("--logging", type=bool, nargs=1, default=False)
args, _ = parser.parse_known_args()


buff_size = 100
data_size = 300
message_size = 10000
vector_size  = 1024

producer = RemoteChannel(buff_size, message_size)
result   = RemoteChannel(data_size, 1)
producer.claim(data_size)
result.claim(data_size)


if args.logging:
    producer.buf.log.setLevel(DEBUG)
    basicConfig(filename=f"{rank:0>3}.log", level=DEBUG)

if rank == 0:
    data = np.random.rand(data_size, vector_size)
    
    p_sum = 0.
    for elt in data:
        producer.putf(elt)
        p_sum += np.sum(elt)
    print(f"{rank=} {p_sum=}")


if rank > 0:
    res = 0
    p_sum = 0.
    for i in range(data_size):
        p = producer.take()
        # sleep(random())
        if p is not None:
            sp = np.sum(p)
            result.putf((sp,))
            p_sum += sp
            res += 1

    print(f"{rank=} {res=}")

# Putf has no need for Barriers :)
# comm.Barrier()

if rank == 0:

    p_sum_r = 0
    for i in range(data_size):
        sp = result.take()
        p_sum_r += sp[0]
        # print(f"{rank=} {sp=}")

    print(f"{rank=} {p_sum_r=} : {p_sum - p_sum_r}")
