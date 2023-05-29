import numpy as np

ALL_GEARS = ['START', 'OPENCL_1', 'OPENCL_2',
            'CYTHON_THREADED','CYTHON_THREADED_DYNAMIC', 
            'CYTHON_THREADED_GUIDED', 'CYTHON_THREADED_STATIC',
            'CYTHON_UNTHREADED', 'NUMBA', 'PYTHON']

class SimMethod:
    """
    Class used to simulate a liquid engine method with several gears
    """

    def __init__(self) -> None:

        self.rng = np.random.default_rng()

        # (0,0) = probability to change from start to start whilst running method one
        # (0,1) = probability to change from start to opencl_1 whilst running method two
        # (row, col) = probability to change from state row to state col whilst running method n_method
        # This means that ALL rows should sum to 1
        self.probability_matrix = np.zeros((len(ALL_GEARS), len(ALL_GEARS)))
        
        # Initialize all VALID transitions with uniform probablity values 
        # This is kinda equivalent so far to having all gears running with the same avg time
        # In practice maybe best to initialize based upon time of previous benchmarks
        self.probability_matrix[:,:] = 1/(len(states)-1)
        # START is the initial state but we NEVER go back to it
        self.probability_matrix[:,0] = 0

        assert np.allclose(np.sum(self.probability_matrix, axis=1),np.ones(len(states)))

        self.avg_times = np.ones(len(ALL_GEARS)-1)
        self.std_times = np.zeros(len(ALL_GEARS)-1)

    def assign_times_to_gears(self,avg_times,std_times):

        self.avg_times = np.array(avg_times)
        self.std_times = np.array(std_times)

        # The transition probability depends ONLY on the destination not on the origin
        # In practice this means that for each transition
        probabilities = (1/self.avg_times)/np.sum(1/self.avg_times)

        assert np.sum(probabilities)==1

        self.probability_matrix[:,:] = probabilities
        self.

        



        

        



    def generate_time(self,mu=None,sigma=None):
        
        if not mu:
            mu = self.mean_time
        if not sigma:
            sigma = self.stddev_time

        return  self.rng.normal(mu,sigma)