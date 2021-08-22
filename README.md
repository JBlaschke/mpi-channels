# Remote Channels via MPI RMA (One Sided Communication)

Use one-sided MPI communication (built on `mpi4py`) to coordinate task-based
parallelism via channels. The `RemoteChannel` class is a message queue.
"Producer" tasks can place messages into the queue using the (blocking) `put`
-- or the (non-blocking) `putf` -- methods. "Consumer" task can take messages
from the queue using the (blocking) `take` -- or the (non-blocking) `takef` --
methods.

## Installation

```
pip install mpi-channels
```

## Usage

* Create a `RemoteChannel` and ensure it knows how many messages to expect:

```python
from mpi_channels import RemoteChannel

# Make a Remote Channel (on MPI rank 0)
inputs = RemoteChannel(buff_size, message_size)
inputs.claim(data_size)
```

The `claim` method increments the expected number of messages (a counter used
to determine of a `take` call should wait for more data).

* Place data into the channel:

```python
# Put data into channel
if rank == 0:
    data = np.random.rand(data_size, vector_size)
    
    p_sum = 0.
    for elt in data:
        inputs.putf(elt)
```

The data `elt` must be a python object with a `len` (*hint:* scalars should be
wrapped in a single-item tuple: `inputs.putf((val,))` -- the trailing comman
ensures that `len((val,)) = 1`). The `putf` method returns a
`concurrent.Futures.Future` object and does not block execution if the channel
is full.

* Take data from the channel:

```python
# Take data from the channel
if rank > 0:
    res = 0
    p_sum = 0.
    for i in range(data_size):
        p = inputs.take()
```

The `take` method blocks until data can be taken from the channel. If the
channel is empty (that is more data have been taken from it than the sum of all
`inputs.claim(N)` calls until that point), then `take` returns `None`. Note
that `p` is an object with lenght. If `p` is a scalar with value `val` then
`p=[val]`.
