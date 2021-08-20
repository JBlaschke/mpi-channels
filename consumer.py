#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy    as     np
from   mpi4py   import MPI
from   logging  import DEBUG, basicConfig
from   argparse import ArgumentParser

from producer import Producer


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = ArgumentParser()
parser.add_argument("--logging", type=bool, nargs=1, default=False)
args, _ = parser.parse_known_args()


buff_size = 10
data_size = 30
message_size = 10

producer = Producer(buff_size, message_size)

if args.logging:
    producer.buf.log.setLevel(DEBUG)
    basicConfig(filename=f"{rank:0>3}.log", level=DEBUG)

data = np.random.rand(data_size, 6)
# producer.fill(data)
producer.claim(len(data))
for elt in data:
    producer.put(elt)

# print(f"{rank=} {data=}")

res = 0
for i in range(data_size):
    p = producer.take(1)
    if p is not None:
        print(f"{rank=}, {i=}, {p=}")
        res += 1

print(f"{rank=} {res=}")
