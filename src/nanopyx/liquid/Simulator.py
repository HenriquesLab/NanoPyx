import json
import pprint
import numpy as np
from scipy.stats import norm
import random

import copy

from sklearn.linear_model import LogisticRegression

"""
ALL_GEARS = ['OPENCL_1', 'OPENCL_2',
            'CYTHON_THREADED','CYTHON_THREADED_DYNAMIC', 
            'CYTHON_THREADED_GUIDED', 'CYTHON_THREADED_STATIC',
            'CYTHON_UNTHREADED', 'NUMBA', 'PYTHON']
"""

ALL_GEARS = ['OPENCL', 'CYTHON_THREADED', 'CYTHON_UNTHREADED', 'PYTHON']


class Method:

    def __init__(self, name, real_avg_times, real_std_times, exp=False) -> None:
        
        self.name = name

        self.real_avg_times = real_avg_times
        self.real_std_times = real_std_times

        self.exp = exp
        
        self.rng = np.random.default_rng()

        # Initialize a blank slate
        # No previous data
        self.observed_data = {g:[] for g in ALL_GEARS}
        self.observed_avg_times = {g:None for g in ALL_GEARS}
        self.observed_std_times = {g:None for g in ALL_GEARS}
        self.observed_exp_avg_times = {g:None for g in ALL_GEARS}
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

        exp_weights = {}
        for g in ALL_GEARS:
            # create truncated normal distribution
            data = self.observed_data[g]
            x = np.linspace(0, len(data)-1, len(data))
            mu = x[-1]
            sigma = 20
            weights = norm.pdf(x,mu,sigma)
            weights = np.abs(weights)  # take absolute value
            weights /= np.sum(weights)  # normalize to sum 1
            exp_weights[g] = weights

        self.observed_exp_avg_times = {g:np.sum(self.observed_data[g] * exp_weights[g]) for g in ALL_GEARS}
        self.observed_exp_std_times = {g:np.sqrt(np.sum(exp_weights[g] * (self.observed_data[g] - self.observed_exp_avg_times[g])**2)) for g in ALL_GEARS}    
        
        if self.exp:
            probabilities = {g:(1/self.observed_exp_avg_times[g])**2 for g in ALL_GEARS}
        else:
            probabilities = {g:(1/self.observed_avg_times[g])**2 for g in ALL_GEARS}

        sum_of_prob = np.sum([probabilities[g] for g in ALL_GEARS])
        self.probabilities = {g:probabilities[g]/sum_of_prob for g in ALL_GEARS}
        assert np.allclose(np.sum([self.probabilities[g] for g in ALL_GEARS]),1)

    def run(self, solid=False):
        
        if solid:
            gear_chosen = min(self.real_avg_times, key=self.real_avg_times.get)
        else:
            gear_chosen = self.rng.choice(ALL_GEARS,p=[self.probabilities[g] for g in ALL_GEARS])

        self.observed_data[gear_chosen].append(self.sample_real_time(gear_chosen))

        self.observed_avg_times[gear_chosen] = np.average(self.observed_data[gear_chosen])      
        self.observed_std_times[gear_chosen] = np.std(self.observed_data[gear_chosen])

        # create truncated normal distribution
        data = self.observed_data[gear_chosen]
        x = np.linspace(0, len(data)-1, len(data))
        mu = x[-1]
        sigma = 20
        weights = norm.pdf(x,mu,sigma)
        weights = np.abs(weights)  # take absolute value
        weights /= np.sum(weights)  # normalize to sum 1
        
        self.observed_exp_avg_times[gear_chosen] = np.sum(data * weights)
        self.observed_exp_std_times[gear_chosen] = np.sqrt(np.sum(weights * (data - self.observed_exp_avg_times[gear_chosen])**2))
        
        if self.exp:
            probabilities = {g:(1/self.observed_exp_avg_times[g])**2 for g in ALL_GEARS}
        else:
            probabilities = {g:(1/self.observed_avg_times[g])**2 for g in ALL_GEARS}
        
        sum_of_prob = np.sum([probabilities[g] for g in ALL_GEARS])
        self.probabilities = {g:probabilities[g]/sum_of_prob for g in ALL_GEARS}

        assert np.allclose(np.sum([self.probabilities[g] for g in ALL_GEARS]),1)

        self.time2run.append(self.observed_data[gear_chosen][-1])
        self.gear_history.append(gear_chosen)

        return gear_chosen, self.observed_data[gear_chosen][-1]

    def run_anomalous(self, affected_gear, new_avg, new_std, solid=False):

        if solid:
            gear_chosen = min(self.real_avg_times, key=self.real_avg_times.get)
        else:
            gear_chosen = self.rng.choice(ALL_GEARS,p=[self.probabilities[g] for g in ALL_GEARS])

        if gear_chosen == affected_gear:
            self.observed_data[gear_chosen].append(self.rng.normal(new_avg, new_std,1)[0])
        else:
            self.observed_data[gear_chosen].append(self.sample_real_time(gear_chosen))

        self.observed_avg_times[gear_chosen] = np.average(self.observed_data[gear_chosen])      
        self.observed_std_times[gear_chosen] = np.std(self.observed_data[gear_chosen])

        self.observed_exp_avg_times[gear_chosen] = np.average(self.observed_data[gear_chosen], weights=[np.exp(-i) for i in range(len(self.observed_data[gear_chosen]))][::-1])

        if self.exp:
            probabilities = {g:(1/self.observed_exp_avg_times[g])**2 for g in ALL_GEARS}
        else:
            probabilities = {g:(1/self.observed_avg_times[g])**2 for g in ALL_GEARS}

        sum_of_prob = np.sum([probabilities[g] for g in ALL_GEARS])
        self.probabilities = {g:probabilities[g]/sum_of_prob for g in ALL_GEARS}

        assert np.allclose(np.sum([self.probabilities[g] for g in ALL_GEARS]),1)

        self.time2run.append(self.observed_data[gear_chosen][-1])
        self.gear_history.append(gear_chosen)

        return gear_chosen, self.observed_data[gear_chosen][-1]
    
    def run_with_agent(self, agent):
        
        # choose gear
        avg = copy.deepcopy(self.observed_exp_avg_times)
        std = copy.deepcopy(self.observed_exp_std_times)
        gear_chosen = agent.get_runtype(avg, std)

        self.probabilities = {g: agent.probabilities[g] for g in ALL_GEARS}

        # run
        self.observed_data[gear_chosen].append(self.sample_real_time(gear_chosen))

        data = np.array(self.observed_data[gear_chosen])
        n_points = 200
        if data.shape[0] < n_points:
            n_points = data.shape[0]
        lower_limit = data.shape[0] - n_points
        data = data[lower_limit:]

        x = np.linspace(0, len(data)-1, len(data))
        mu = x[-1]
        sigma = 20
        weights = norm.pdf(x,mu,sigma)
        weights = np.abs(weights)  # take absolute value
        weights /= np.sum(weights)  # normalize to sum 1
        
        self.observed_exp_avg_times[gear_chosen] = np.sum(data * weights)
        self.observed_exp_std_times[gear_chosen] = np.sqrt(np.sum(weights * (data - self.observed_exp_avg_times[gear_chosen])**2))
        
        agent.check_delay(gear_chosen, self.observed_data[gear_chosen][-1],self.observed_data[gear_chosen][:-1])

        self.time2run.append(self.observed_data[gear_chosen][-1])
        self.gear_history.append(gear_chosen)

        return gear_chosen, self.observed_data[gear_chosen][-1]

    def run_anomalous_with_agent(self, agent, affected_gear, new_avg, new_std):

        # choose gear
        avg = copy.deepcopy(self.observed_exp_avg_times)
        std = copy.deepcopy(self.observed_exp_std_times)
        gear_chosen = agent.get_runtype(avg, std)

        self.probabilities = {g: agent.probabilities[g] for g in ALL_GEARS}

        # run
        if gear_chosen == affected_gear:
            self.observed_data[gear_chosen].append(self.rng.normal(new_avg, new_std,1)[0])
        else:
            self.observed_data[gear_chosen].append(self.sample_real_time(gear_chosen))
        
        data = np.array(self.observed_data[gear_chosen])
        n_points = 200
        if data.shape[0] < n_points:
            n_points = data.shape[0]
        lower_limit = data.shape[0] - n_points
        data = data[lower_limit:]

        x = np.linspace(0, len(data)-1, len(data))
        mu = x[-1]
        sigma = 20
        weights = norm.pdf(x,mu,sigma)
        weights = np.abs(weights)  # take absolute value
        weights /= np.sum(weights)  # normalize to sum 1
        
        self.observed_exp_avg_times[gear_chosen] = np.sum(data * weights)
        self.observed_exp_std_times[gear_chosen] = np.sqrt(np.sum(weights * (data - self.observed_exp_avg_times[gear_chosen])**2))
        
        agent.check_delay(gear_chosen,self.observed_data[gear_chosen][-1],self.observed_data[gear_chosen][:-1])

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

        self.agent = FakeAgent()

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

        print(f"Ran the entire workflow a total of {len(self.total_time)} times")
        print(f"The average time to run all methods was {np.average(self.total_time):.2f} std_dev {np.std(self.total_time):.2f}")
        for met in self.method_objects:
            gear_used, counts = np.unique(met.gear_history, return_counts=True)
            print(f"Method {met.name} used {gear_used[np.argmax(counts)]} the most ({100*np.max(counts)/np.sum(counts):.1f}%)")

    def benchmark_all_methods(self,n_times=1):

        for met in self.method_objects:
            met.run_benchmark(n_times)

    def run_simulations(self, iter_n=100, solid=False):

        for iter in range(iter_n):
            time = []
            gears = []
            for method in self.method_objects:
                g,t = method.run(solid=solid)
                time.append(t)
                gears.append(g)
            self.gears.append(gears)
            self.time.append(time)
            self.total_time.append(np.sum(time))

    def run_anomalous_simulations(self, iter_n=1000, ano_start=300, ano_end=600, affected_gear=ALL_GEARS[0], new_avg=100, new_std=10, solid=False):

        for iter in range(iter_n):

            time = []
            gears = []
            for method in self.method_objects:
                
                if ano_start<iter<ano_end:
                    g,t = method.run_anomalous(affected_gear, new_avg, new_std, solid=solid)
                else: 
                    g,t = method.run(solid=solid)
                
                time.append(t)
                gears.append(g)

            self.gears.append(gears)
            self.time.append(time)
            self.total_time.append(np.sum(time))

    def run_simulations_agent(self, iter_n=100):

        for iter in range(iter_n):
            time = []
            gears = []
            for method in self.method_objects:
                g,t = method.run_with_agent(self.agent)
                time.append(t)
                gears.append(g)
            self.gears.append(gears)
            self.time.append(time)
            self.total_time.append(np.sum(time))

    def run_anomalous_simulations_agent(self, iter_n=1000, ano_start=300, ano_end=600, affected_gear=ALL_GEARS[0], new_avg=100, new_std=10):

        for iter in range(iter_n):

            time = []
            gears = []
            for method in self.method_objects:
                
                if ano_start<iter<ano_end:
                    g,t = method.run_anomalous_with_agent(self.agent, affected_gear, new_avg, new_std)
                else: 
                    g,t = method.run_with_agent(self.agent)
                
                time.append(t)
                gears.append(g)

            self.gears.append(gears)
            self.time.append(time)
            self.total_time.append(np.sum(time))

class FakeAgent:
    def __init__(self) -> None:
        
        self.delayed_runtypes = {}  # Store runtypes as keys and their values as (delay_factor, delay_prob)
        self.probabilities = {}
                
    def get_runtype(self,avg,std):
        
        if len(self.delayed_runtypes.keys()) > 0:
            for runtype in self.delayed_runtypes.keys():
                if runtype in avg.keys():
                    delay_factor, delay_prob = self.delayed_runtypes[runtype]
                    # Weighted avg by the probability the run_type is still delayed
                    # expected_time * P(~delay) + delayed_time * P(delay)
                    avg[runtype] = avg[runtype] * (1 - delay_prob) + avg[runtype] * delay_factor * delay_prob


        weights = [(1/avg[k])**2 for k in avg]
        if sum(weights) == 0:
            weights = [1 for k in avg] 
        s = sum(weights)
        weights = [w/s for w in weights]
        
        assert np.allclose(sum(weights), 1)

        for g in ALL_GEARS:
            self.probabilities[g] = weights[ALL_GEARS.index(g)]
        
        return random.choices(list(avg.keys()), weights=weights, k=1)[0]
        

    def check_delay(self, run_type, time2run, previous_times):

        threaded_runtypes = ['CYTHON_THREADED']
        
        avg = np.nanmean(previous_times) # standard average as opposed to weighted as a weighted average would throw false negatives if delays happen consecutively
        std = np.nanstd(previous_times)
        if time2run > avg + 2*std:
            previous_times.append(time2run)
            delay_factor = time2run / avg
            delay_prob = self.calculate_prob_of_delay(previous_times, avg, std)
            # print(f"Run type {run_type} was delayed in the previous run. Delay factor: {delay_factor}, Delay probability: {delay_prob}")
            if "THREADED" in run_type:
                for threaded_run_type in threaded_runtypes:
                    self.delayed_runtypes[threaded_run_type] = (delay_factor, delay_prob)
            else:
                self.delayed_runtypes[run_type] = (delay_factor, delay_prob)

    def calculate_prob_of_delay(self, runtimes_history, avg, std):
        """
        Calculates the probability that the given run_type is still delayed using historical data
        """
        # Boolean array, True if delay, False if not
        delays = runtimes_history > avg+2*std

        model = LogisticRegression()
        model.fit([[state] for state in delays[:-1]], delays[1:])
        return model.predict_proba([[True]])[:,model.classes_.tolist().index(True)][0]

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
    
    