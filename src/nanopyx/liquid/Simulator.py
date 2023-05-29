import numpy as np

from liquid_state_sim_v2 import ALL_GEARS, SimMethod

class Simulator:
    """
    This class should encapsulate:
    1. Workflow 
    2. Each gear used
    3. Time for each gear
    """
    def __init__(self, *args, max_iter = 100) -> None:
        
        self.rng = np.random.default_rng()
        self.method_objects = args

        self.history = np.empty((100,len(self.method_objects))).astype(str)
        self.history[:,:] = 'N/A'
        
        self.max_iter = max_iter
        self.current_iter = 0

    def next_iter(self,):

        states = []
        for met in self.method_objects:
            gear = self.rng.choice(len(ALL_GEARS),1,p=met.probability_vector)[0]
            states.append(ALL_GEARS[gear])
            met.penalty(self.current_iter,gear)

        self.history[self.current_iter,:] = states
        self.current_iter += 1

    def run_iter(self):

        while self.current_iter<self.max_iter:
            self.next_iter()

    def print_methods(self):

        print("#######################")
        float_formatter = "{:.2f}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})
        for met in self.method_objects:
            
            print(f"{met.name} : {met.probability_vector}")

        print("#######################")
        


if __name__ == "__main__":
    
    # This one prefers gpus
    method_1 = SimMethod('1')
    method_1.assign_times_to_gears([10,15,20,21,22,21,21,23,100],[1,1,1,1,1,1,1,1,10])
    
    # This one prefers cpus
    method_2 = SimMethod('2')
    method_2.assign_times_to_gears([20,25,10,11,10,10,11,13,50],[1,1,1,1,1,1,1,1,10])

    # This one prefers nothing
    method_3 = SimMethod('3')
    method_3.assign_times_to_gears([10,10,10,10,10,11,10,10,25],[1,1,1,1,1,1,1,1,10])

    sim = Simulator(method_1,method_2,method_3)

    sim.print_methods()
    sim.run_iter()
    sim.print_methods()

    