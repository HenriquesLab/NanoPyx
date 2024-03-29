# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

from ..utils.random cimport _random

import numpy as np
cimport numpy as np

from cython.parallel import prange
from tqdm import tqdm

def simple_state_transition_model(int n_particles, int n_ticks, double p_on, double p_transient_off, double p_permanent_off, int initial_state = 1) -> np.ndarray:
    """
    Simple photoswitching state transition model
    :param n_particles: number of particles
    :param n_ticks: number of time ticks
    :param p_on: probability of switching on
    :param p_transient_off: probability of switching off transiently
    :param p_permanent_off: probability of switching off permanently
    :param initial_state: initial state of the photoswitch
    :return: array of states with shape (n_particles, n_ticks)

    States:
        -1: bleached
        0: off
        1: on
    Transitions:
        -1 -> -1
        0 -> 0
        1 -> 1
        0 -> 1 with probability p_on
        1 -> 0 with probability p_transient_off
        1 -> -1 with probability p_permanent_off

    Example:
        >>> n_ticks = 1000
        >>> n_particles = 100
        >>> p_on = 0.1
        >>> p_transient_off = 0.8
        >>> p_permanent_off = 0.1
        >>> initial_state = 0
        >>> states = simple_state_transition_model(n_ticks, n_particles, p_on, p_transient_off, p_permanent_off, initial_state)
    """
    cdef int[:,:] states = np.zeros((n_particles, n_ticks), dtype=np.int32)
    cdef int i, b, b_stop
    cdef int _initial_state = initial_state

    # break it down to 100 tracks at a time
    with tqdm(total=n_particles, desc="Tracing", unit="particles") as progress_bar:
        for b in range(0, n_particles, 100):
            with nogil:
                b_stop = min(b + 100, n_particles)
                for i in prange(b, b_stop):
                    _simple_state_transition_model(states[i,:], p_on, p_transient_off, p_permanent_off, _initial_state)
            progress_bar.update(100)
    return np.asarray(states)


cdef void _simple_state_transition_model(int[:] states, double p_on, double p_transient_off, double p_permanent_off, int initial_state) nogil:
    """
    Simple photoswitching state transition model
    :param states: array of states
    :param p_on: probability of switching on
    :param p_transient_off: probability of switching off transiently
    :param p_permanent_off: probability of switching off permanently
    :param initial_state: initial state of the photoswitch

    States:
        -1: bleached
        0: off
        1: on
    Transitions:
        -1 -> -1
        0 -> 0
        1 -> 1
        0 -> 1 with probability p_on
        1 -> 0 with probability p_transient_off
        1 -> -1 with probability p_permanent_off
    """

    cdef int n_ticks = states.shape[0]
    cdef int current_state = initial_state

    for i in range(n_ticks):
        if current_state == -1:
            states[i] = -1
        elif current_state == 0:
            if _random() < p_on:
                current_state = 1
                states[i] = 1
            else:
                states[i] = 0
        elif current_state == 1:
            if _random() < p_permanent_off:
                current_state = -1
                states[i] = -1
            elif _random() < p_transient_off:
                current_state = 0
                states[i] = 0
            else:
                states[i] = 1
