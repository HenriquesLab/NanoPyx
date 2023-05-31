import json
import pprint
import numpy as np

ALL_GEARS = ['OPENCL_1', 'OPENCL_2',
            'CYTHON_THREADED','CYTHON_THREADED_DYNAMIC', 
            'CYTHON_THREADED_GUIDED', 'CYTHON_THREADED_STATIC',
            'CYTHON_UNTHREADED', 'NUMBA', 'PYTHON']

class Method:

    def __init__(self, name, real_avg_times, real_std_times) -> None:
        
        self.name = name

        self.real_avg_times = real_avg_times
        self.real_std_times = real_std_times
        
        self.rng = np.random.default_rng()

        # Initialize a blank slate
        # No previous data
        self.observed_data = {g:[] for g in ALL_GEARS}
        self.observed_avg_times = {g:None for g in ALL_GEARS}
        self.observed_std_times = {g:None for g in ALL_GEARS}
        # Equal probabilities
        self.probabilities = {g:1/len(ALL_GEARS) for g in ALL_GEARS}

        self.time2run = []
        self.gear_history = []
        
        assert np.allclose(np.sum([self.probabilities[g] for g in ALL_GEARS]),1)


    def sample_real_time(self,gear):
        return self.rng.normal(self.real_avg_times[gear], self.real_std_times[gear], 1)[0]
        

    def run_benchmark(self, ntimes=1):

        for n in range(ntimes):
            for g in ALL_GEARS:
                self.observed_data[g].append(self.sample_real_time(g))

        self.observed_avg_times = {g:np.average(self.observed_data[g]) for g in ALL_GEARS}        
        self.observed_std_times = {g:np.std(self.observed_data[g]) for g in ALL_GEARS}

        probabilities = {g:(1/self.observed_avg_times[g])**2 for g in ALL_GEARS}
        sum_of_prob = np.sum([probabilities[g] for g in ALL_GEARS])
        self.probabilities = {g:probabilities[g]/sum_of_prob for g in ALL_GEARS}
        assert np.allclose(np.sum([self.probabilities[g] for g in ALL_GEARS]),1)

    def run(self):

        gear_chosen = self.rng.choice(ALL_GEARS,p=[self.probabilities[g] for g in ALL_GEARS])

        self.observed_data[gear_chosen].append(self.sample_real_time(gear_chosen))

        self.observed_avg_times[gear_chosen] = np.average(self.observed_data[gear_chosen])      
        self.observed_std_times[gear_chosen] = np.std(self.observed_data[gear_chosen])

        probabilities = {g:(1/self.observed_avg_times[g])**2 for g in ALL_GEARS}
        sum_of_prob = np.sum([probabilities[g] for g in ALL_GEARS])
        self.probabilities = {g:probabilities[g]/sum_of_prob for g in ALL_GEARS}
        assert np.allclose(np.sum([self.probabilities[g] for g in ALL_GEARS]),1)

        self.time2run.append(self.observed_data[gear_chosen][-1])
        self.gear_history.append(gear_chosen)

        return gear_chosen, self.observed_data[gear_chosen][-1]

class Simulator:
    """
    This class should encapsulate:
    1. Workflow 
    2. Each gear used
    3. Time for each gear
    """
    def __init__(self, *args) -> None:
        
        self.rng = np.random.default_rng()

        self.method_objects = args
        
        self.current_iter = 0

        self.gears = []
        self.time = []
        self.total_time = []

    def print_methods(self):

        print_string = f"{[m.name for m in self.method_objects]} \n"

        for g in ALL_GEARS:
            gear_string = f"{g} "
            for met in self.method_objects:
                gear_string += f"|{met.probabilities[g]:.4f}| "
            print_string += gear_string + '\n'
        print(print_string)

    def print_stats(self):

        print(f"Ran the entire workflow a total of {len(self.total_time)}")
        print(f"The average time to run all methods was {np.average(self.total_time):.2f} std_dev {np.std(self.total_time):.2f}")
        for met in self.method_objects:
            gear_used, counts = np.unique(met.gear_history, return_counts=True)
            print(f"Method {met.name} used {gear_used[np.argmax(counts)]} the most ({100*np.max(counts)/np.sum(counts):.1f}%)")

    def benchmark_all_methods(self,n_times=1):

        for met in self.method_objects:
            met.run_benchmark(n_times)

    def run_simulations(self, iter_n=100):

        
        for iter in range(iter_n):
            time = []
            gears = []
            for method in self.method_objects:
                g,t = method.run()
                time.append(t)
                gears.append(g)
            self.gears.append(gears)
            self.time.append(time)
            self.total_time.append(np.sum(time))


if __name__ == "__main__":
    
    avg_1 = dict(zip(ALL_GEARS,[10,13,25,25,25,25,25,25,100]))
    std_1 = dict(zip(ALL_GEARS,[1,1,3,3,3,3,3,3,10]))
    met_1 = Method('1',avg_1,std_1)

    avg_2 = dict(zip(ALL_GEARS,[5,7,2,2,2,2,2,2,10]))
    std_2 = dict(zip(ALL_GEARS,[.1,.1,.1,.1,.1,.1,.1,.1,1]))
    met_2 = Method('2',avg_2,std_2)
    
    avg_3 = dict(zip(ALL_GEARS,[30,30,30,30,30,30,30,30,30]))
    std_3 = dict(zip(ALL_GEARS,[10,10,10,10,10,10,10,10,10]))
    met_3 = Method('3',avg_3,std_3)

    sim = Simulator(met_1,met_2,met_3)
    sim.print_methods()
    