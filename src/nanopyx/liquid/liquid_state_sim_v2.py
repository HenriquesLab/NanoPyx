import numpy as np

ALL_GEARS = ['OPENCL_1', 'OPENCL_2',
            'CYTHON_THREADED','CYTHON_THREADED_DYNAMIC', 
            'CYTHON_THREADED_GUIDED', 'CYTHON_THREADED_STATIC',
            'CYTHON_UNTHREADED', 'NUMBA', 'PYTHON']

class SimMethod:
    """
    Class used to simulate a liquid engine method with several gears
    """

    def __init__(self, name, sample_n=10000) -> None:
        
        self.name = name
        self.sample_n = sample_n

        self.rng = np.random.default_rng()

        self.probability_vector = np.ones(len(ALL_GEARS))
        
        # Initialize transitions with uniform probablity values 
        # This is kinda equivalent so far to having all gears running with the same avg time
        # In practice maybe best to initialize based upon time of previous benchmarks
        self.probability_vector *= 1/(len(ALL_GEARS))

        assert np.allclose(np.sum(self.probability_vector),1)

        self.avg_times = np.ones(len(ALL_GEARS))
        self.std_times = np.ones(len(ALL_GEARS))

        self.time_samples = np.zeros((self.sample_n,len(ALL_GEARS)))
        for n in range(len(ALL_GEARS)):
            self.time_samples[:,n] = self.rng.normal(self.avg_times[n], self.std_times[n], self.sample_n) 
        

    def assign_times_to_gears(self,avg_times,std_times):

        self.avg_times = np.array(avg_times)
        self.std_times = np.array(std_times)

        # The transition probability depends ONLY on the destination
        # How do the probabilties depend on the avg/std time?

        # Inversely proportional
        #probabilities = (1/self.avg_times)/np.sum(1/self.avg_times)

        # Inversely proportional to the square
        probabilities = (1/(self.avg_times**2))/np.sum(1/(self.avg_times**2))

        assert np.allclose(np.sum(self.probability_vector),1)

        self.probability_vector = probabilities

        self.time_samples = np.zeros((self.sample_n,len(ALL_GEARS)))
        for n in range(len(ALL_GEARS)):
            self.time_samples[:,n] = self.rng.normal(self.avg_times[n], self.std_times[n], self.sample_n) 

    def penalty(self,n_iter,gear_number):

        # sample time 
        time = self.time_samples[n_iter, gear_number]
        
        avg = self.avg_times[gear_number]
        std = self.std_times[gear_number]
        
        
        #penalty = self.penalty_function1(time,avg,std)
        penalty = self.penalty_function2(time,avg,std)

        self.probability_vector[gear_number] = self.probability_vector[gear_number] * penalty
        # renormalize 
        self.probability_vector = self.probability_vector / np.sum(self.probability_vector)

        assert np.allclose(np.sum(self.probability_vector), 1)


    @staticmethod
    def penalty_function1(sample,avg,std):
        
        if sample>avg+3*std or sample<avg-3*std:

            # How far from the average value?
            dist = (sample-avg)/avg

            # dist is positive if it takes MORE TIME. penalty_factor should be LOWER for higer dists
            # Example: avg=10, sample=15, dist = 0.5 => penalty = 0.5
            #          avg=10, sample=20, dist = 1.0 => penalty = 0.0
            
            # dist is negative if it takes LESS TIME. penalty_factor should be HIGHER for higher dists
            # Example: avg=20, sample=15, dist = -0.25 => penalty = 1.25
            #          avg=20, sample=10, dist = -0.50 => penalty = 1.5
            
            penalty = 1-dist 
        else:
            penalty = 1
        
        return max(penalty,0.001)

    @staticmethod
    def penalty_function2(sample,avg,std):

        dist = (sample-avg)/std
        
        if np.abs(dist)<3:
            return 1
        else:
            return 1/(1+np.exp(0.5*dist)) + 0.5



if __name__ == "__main__":

    """
    ALL_GEARS = ['OPENCL_1', 'OPENCL_2',
            'CYTHON_THREADED','CYTHON_THREADED_DYNAMIC', 
            'CYTHON_THREADED_GUIDED', 'CYTHON_THREADED_STATIC',
            'CYTHON_UNTHREADED', 'NUMBA', 'PYTHON']
    """

    # This one prefers gpus
    method_1 = SimMethod('1')
    method_1.assign_times_to_gears([10,15,20,21,22,21,21,23,100],[1,1,1,1,1,1,1,1,10])
    
    # This one prefers cpus
    method_2 = SimMethod('2')
    method_2.assign_times_to_gears([20,25,10,11,10,10,11,13,50],[1,1,1,1,1,1,1,1,10])

    # This one prefers nothing
    method_3 = SimMethod('3')
    method_3.assign_times_to_gears([10,10,10,10,10,11,10,10,25],[1,1,1,1,1,1,1,1,10])

    methods = [method_1,method_2,method_3]
    n_methods = len(methods)

    iter_n = 100
    all_history = np.empty((100,n_methods+1)).astype(str)

    rng = np.random.default_rng()

    for i in range(iter_n):
        state_per_iter = ['START',]
        for n in range(len(methods)):    
            gear = rng.choice(len(ALL_GEARS),1, p=methods[n].probability_vector)[0]
            state_per_iter.append(ALL_GEARS[gear])
        
        all_history[i,:] = state_per_iter


    print(all_history)