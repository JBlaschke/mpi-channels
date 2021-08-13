#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy    as     np
from   mpi4py   import MPI

from   producer import Producer


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

producer = Producer(20) 

data = np.random.rand(10)
producer.fill(data)

print(f"{rank=} {data=}")

for i in range(10):
    p = producer.take(1)
    print(f"{rank=}, {i=}, {p=}")
