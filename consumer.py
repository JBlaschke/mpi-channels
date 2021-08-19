#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy   as     np
from   mpi4py  import MPI
from   logging import DEBUG, basicConfig

from producer import Producer


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

buff_size = 10
data_size = 30

producer = Producer(buff_size)
# producer.buf.log.setLevel(DEBUG)
# basicConfig(filename=f"{rank:0>3}.log", level=DEBUG)

data = np.random.rand(data_size)
producer.fill(data)

print(f"{rank=} {data=}")

res = 0
for i in range(data_size):
    p = producer.take(1)
    if p is not None:
        print(f"{rank=}, {i=}, {p=}")
        res += 1

print(f"{res=}")
