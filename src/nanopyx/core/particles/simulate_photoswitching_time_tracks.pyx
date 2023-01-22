# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from ..utils.random cimport _random

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
