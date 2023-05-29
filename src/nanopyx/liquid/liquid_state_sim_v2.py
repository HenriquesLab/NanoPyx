import numpy as np

ALL_GEARS = ['OPENCL_1', 'OPENCL_2',
            'CYTHON_THREADED','CYTHON_THREADED_DYNAMIC', 
            'CYTHON_THREADED_GUIDED', 'CYTHON_THREADED_STATIC',
            'CYTHON_UNTHREADED', 'NUMBA', 'PYTHON']

class SimMethod:
    """
    Class used to simulate a liquid engine method with several gears
    """

    def __init__(self, name) -> None:
        
        self.name = name

        self.rng = np.random.default_rng()

        self.probability_vector = np.ones(len(ALL_GEARS))
        
        # Initialize transitions with uniform probablity values 
        # This is kinda equivalent so far to having all gears running with the same avg time
        # In practice maybe best to initialize based upon time of previous benchmarks
        self.probability_vector *= 1/(len(ALL_GEARS))

        assert np.allclose(np.sum(self.probability_vector),1)

        self.avg_times = np.ones(len(ALL_GEARS))
        self.std_times = np.ones(len(ALL_GEARS))

        self.time_samples = np.zeros((100,len(ALL_GEARS)))
        for n in range(len(ALL_GEARS)):
            self.time_samples[:,n] = self.rng.normal(self.avg_times[n], self.std_times[n], 100) 
        

    def assign_times_to_gears(self,avg_times,std_times):

        self.avg_times = np.array(avg_times)
        self.std_times = np.array(std_times)

        # The transition probability depends ONLY on the destination
        # Start with a simple inversely proportional example
        probabilities = (1/self.avg_times)/np.sum(1/self.avg_times)

        assert np.sum(probabilities)==1

        self.probability_vector = probabilities

        self.time_samples = np.zeros((100,len(ALL_GEARS)))
        for n in range(len(ALL_GEARS)):
            self.time_samples[:,n] = self.rng.normal(self.avg_times[n], self.std_times[n], 100) 
        
    
if __name__ == "__main__":

    method_1 = SimMethod('1')
    method_2 = SimMethod('2')
    method_3 = SimMethod('3')
    method_4 = SimMethod('4')
    method_5 = SimMethod('5')
