#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy              as     np
from   mpi4py             import MPI
from   logging            import DEBUG, basicConfig
from   argparse           import ArgumentParser
from   random             import random
from   time               import sleep

from mpi_channels import RemoteChannel


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = ArgumentParser()
parser.add_argument("--logging", type=bool, nargs=1, default=False)
args, _ = parser.parse_known_args()


#_______________________________________________________________________________
# Define Work

buff_size = 100
data_size = 300
message_size = 10000
vector_size  = 1024

#_______________________________________________________________________________
# Set up remote channels
# NOTE: if you're using the async `putf` methods (rather than the blocking
# `put` methods), then you don't need to make sure that the `results`
# RemoteChannel is big enough to theretically accommodate all results. See the
# consumer.py for the non-async example.

inputs = RemoteChannel(buff_size, message_size)
result = RemoteChannel(buff_size, 1)
# Ensure that remote channels expect the right amount of data
inputs.claim(data_size)
result.claim(data_size)

#_______________________________________________________________________________
# (optional) Enable logging

if args.logging:
    inputs.buf.log.setLevel(DEBUG)
    basicConfig(filename=f"{rank:0>3}.log", level=DEBUG)

#_______________________________________________________________________________
# (on rank 0) Create data and put it into the `inputs` RemoteChannel. Also
# compute the total sum of all data.

if rank == 0:
    data = np.random.rand(data_size, vector_size)
    
    p_sum = 0.
    for elt in data:
        inputs.putf(elt)
        p_sum += np.sum(elt)
    print(f"{rank=} {p_sum=}", flush=True)

#_______________________________________________________________________________
# (on all other ranks) Take data, and compute the sum locally. The result is
# put into the `result` RemoteChannel.

if rank > 0:
    res = 0
    p_sum = 0.
    for i in range(data_size):
        # print(f"{rank=} taking", flush=True)
        p = inputs.take()
        # print(f"{rank=} {p=}", flush=True)
        # sleep(random())
        if p is not None:
            sp = np.sum(p)
            result.putf((sp,))
            p_sum += sp
            res += 1
        # else:
        #     print(f"{rank=} has quit", flush=True)

    print(f"{rank=} {res=}", flush=True)

#_______________________________________________________________________________
# (on rank 0) Take `result` elements (local partial sums), and finish the tally.

if rank == 0:

    p_sum_r = 0
    for i in range(data_size):
        sp = result.take()
        p_sum_r += sp[0]
        # print(f"{rank=} {sp=}")

    print(f"{rank=} {p_sum_r=} : {p_sum - p_sum_r}", flush=True)
