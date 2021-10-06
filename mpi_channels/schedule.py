#!/usr/bin/env python
# -*- coding: utf-8 -*-


from random import random
from enum   import Enum, auto
from time   import sleep



class Policy(Enum):
    WAIT = auto()
    KILL = auto()


class Action(Enum):
    CONTINUE = auto()
    WAIT = auto()
    KILL = auto()


class Schedule(object):

    def __init__(self, threshold, t_wait=0, r_wait=1, policy=Policy.WAIT):
        self._threshold = threshold
        self._t_wait = t_wait
        self._r_wait = r_wait
        self._policy = policy
        self._reset()

    @property
    def count(self):
        return self._count

    @property
    def wait(self):
        return self._t_wait + self._r_wait*random()

    def _reset(self):
        self._count = 0

    def __call__(self):
        self._count += 1

        if self._count > self._threshold:
            self._reset()
            if self._policy == Policy.WAIT:
                # print("wating", flush=True)
                sleep(self.wait)
                return Action.WAIT 
            if self._policy == Policy.KILL:
                return Action.KILL
        else:
            return Action.CONTINUE
