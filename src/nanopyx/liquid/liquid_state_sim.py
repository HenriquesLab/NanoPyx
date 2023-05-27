
import numpy as np


if __name__ == "__main__":

    # number of methods in a workflow
    n_methods = 4

    # states aka algorithms/implementations/liquid gears
    states = ['START', 'OPENCL_1', 'OPENCL_2', 'CYTHON_THREADED',
              'CYTHON_THREADED_DYNAMIC', 'CYTHON_THREADED_GUIDED', 'CYTHON_THREADED_STATIC',
              'CYTHON_UNTHREADED', 'NUMBA', 'PYTHON']
    
    # This matrix should depend on the method no?
    # (0,0) = probability to change from start to start
    # (0,1) = probability to change from start to opencl_1
    # (row, col) = probability to change from state row to state col
    # This means that ALL rows should sum to 1
    probability_matrix = np.zeros((len(states), len(states)))
    
    # Initialize all VALID transitions with uniform probablity values 
    # In practice maybe best to initialize based upon time of benchmarks
    probability_matrix[:,:] = 1/(len(states)-1)
    probability_matrix[:,0] = 0

    assert np.all(np.sum(probability_matrix, axis=1)==1), print("OOPS probabilities should sum to 1")

    