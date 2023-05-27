
import numpy as np


if __name__ == "__main__":

    # number of methods in a workflow
    n_methods = 4

    # states aka algorithms/implementations/liquid gears
    # Get this into a dict pls
    state_names = ['START', 'OPENCL_1', 'OPENCL_2', 'CYTHON_THREADED',
              'CYTHON_THREADED_DYNAMIC', 'CYTHON_THREADED_GUIDED', 'CYTHON_THREADED_STATIC',
              'CYTHON_UNTHREADED', 'NUMBA', 'PYTHON']
    states = dict(zip(state_names, list(range(len(state_names)))))

    # This matrix should depend on the method (so it should be (n_method, row, col) )
    # (0,0,0) = probability to change from start to start whilst running method one
    # (2,0,1) = probability to change from start to opencl_1 whilst running method two
    # (n_method, row, col) = probability to change from state row to state col whilst running method n_method
    # This means that ALL rows should sum to 1
    probability_matrix = np.zeros((n_methods,len(states), len(states)))
    
    # Initialize all VALID transitions with uniform probablity values 
    # In practice maybe best to initialize based upon time of previous benchmarks
    probability_matrix[:,:,:] = 1/(len(states)-1)
    # START is the initial state but we NEVER go back to it
    probability_matrix[:,:,0] = 0

    assert np.allclose(np.sum(probability_matrix, axis=2),np.ones((n_methods,len(states))))

    # Monte carlo
    rng = np.random.default_rng()
    iter_n = 100
    all_states = []
    for i in range(iter_n):
        # One pass on the workflow
        # ALWAYS start on start
        state_per_iter = [states['START'],]
        for n in range(n_methods):
            # Get probabilities from matrix 
            probs = probability_matrix[n,state_per_iter[-1],:].flatten()
            next_state = rng.choice(len(probs),1, p=probs)
            state_per_iter.append(next_state[0])
        all_states.append(state_per_iter)


    print(all_states)

            


