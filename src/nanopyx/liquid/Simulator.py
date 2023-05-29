import numpy as np

from liquid_state_sim_v2 import ALL_GEARS, SimMethod

class Simulator:
    """
    This class should encapsulate:
    1. Workflow 
    2. Each gear used
    3. Time for each gear
    """
    def __init__(self, *args, max_iter = 1000, penalty=True) -> None:
        
        self.rng = np.random.default_rng()
        self.method_objects = args

        self.max_iter = max_iter
        self.current_iter = 0

        self.history = np.empty((self.max_iter,len(self.method_objects))).astype(str)
        self.history[:,:] = 'N/A'
        
        self.times = np.zeros(self.max_iter)

        self.penalty=penalty

    def next_iter(self,):

        states = []
        time = 0
        for met in self.method_objects:
            gear = self.rng.choice(len(ALL_GEARS),1,p=met.probability_vector)[0]
            states.append(ALL_GEARS[gear])
            time += met.time_samples[self.current_iter, gear]
            if self.penalty:
                met.penalty(self.current_iter,gear)

        self.history[self.current_iter,:] = states
        self.times[self.current_iter] = time
        self.current_iter += 1

    def run_iter(self, until_iter=-1):

        while self.current_iter<self.max_iter:
            if self.current_iter == until_iter:
                break
            self.next_iter()

    def print_methods(self):

        print("#######################")
        float_formatter = "{:.2f}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})
        for met in self.method_objects:
            
            print(f"{met.name} : {met.probability_vector}")

        print(f"AVERAGE TIME = {np.average(self.times[:self.current_iter])}")
        print("#######################")
        
    def add_anomaly(self,iter_start,iter_end,gear,newavg,newstd):

        for met in self.method_objects:
            met.time_samples[iter_start:iter_end,gear] = self.rng.normal(newavg, newstd, iter_end-iter_start) 


if __name__ == "__main__":
    
    # This one prefers gpus
    method_1 = SimMethod('1')
    method_1.assign_times_to_gears([10,10,30,31,32,31,31,40,100],[1,1,2,2,2,2,2,2,10])
    
    # This one prefers cpus
    method_2 = SimMethod('2')
    method_2.assign_times_to_gears([20,25,10,11,10,10,11,13,50],[1,1,1,1,1,1,1,1,10])

    # This one prefers nothing
    method_3 = SimMethod('3')
    method_3.assign_times_to_gears([10,10,10,10,10,11,10,10,25],[1,1,1,1,1,1,1,1,10])

    print("NO PENALTY")
    sim = Simulator(method_1,method_2,method_3,penalty=False)
    sim.add_anomaly(50,100,0,1000,10)

    sim.print_methods()
    sim.run_iter(100)

    sim.print_methods()
    sim.run_iter(200)

    sim.print_methods()

    print("#"*50)
    
    # This one prefers gpus
    method_1 = SimMethod('1')
    method_1.assign_times_to_gears([10,10,30,31,32,31,31,40,100],[1,1,2,2,2,2,2,2,10])
    
    # This one prefers cpus
    method_2 = SimMethod('2')
    method_2.assign_times_to_gears([20,25,10,11,10,10,11,13,50],[1,1,1,1,1,1,1,1,10])

    # This one prefers nothing
    method_3 = SimMethod('3')
    method_3.assign_times_to_gears([10,10,10,10,10,11,10,10,25],[1,1,1,1,1,1,1,1,10])

    print("PENALTY")
    sim = Simulator(method_1,method_2,method_3,penalty=True)
    sim.add_anomaly(50,100,0,1000,10)

    sim.print_methods()
    sim.run_iter(100)

    sim.print_methods()
    sim.run_iter(200)

    sim.print_methods()
