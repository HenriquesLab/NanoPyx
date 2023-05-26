import os
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np

from nanopyx.liquid._le_interpolation_nearest_neighbor import ShiftAndMagnify

class Workflow:

    def __init__(self, tasks:list, task_input:list, order:tuple[int]) -> None:
        # What do i need to totally define a workflow?

        # Tasks is for now a simple list of id's 
        self.tasks = tasks

        # Task input is the list that stores the initial args for the workflow 
        self.task_input = task_input

        # Order is a tuple of ints that defines the order each task is supposed to be ran
        # When two tasks have the same order they can be run concurrently
        # A task with lower order MUST ALWAYS wait for the higher order task to finish
        self.order = order


class Agent:
    # James Pond
    def __init__(self) -> None:

        self.total_cpu_cores = os.cpu_count()

        # Methods that CANT be run concurrently (because in some way they run on the same on device)
        self.gpu = False
        self.threaded = False

    def run_workflow(self, w:Workflow) -> None:
        
        # Run a workflow
        inputs = w.task_input

        # Iterate through the order
        for step in np.unique(w.order):
            
            print("STARTING STEP ", step)
            # Check what to run
            tasks = [t for t,o in zip(w.tasks,w.order) if o == step]

            # Check run_type for each task
            task_objects = [t() for t in tasks]
            fastest_run_types = [t._get_fastest_run_type() for t in task_objects]

            # The fastest run type can sometimes not be the chosen one.
            final_run_types = []
            # How many cores does the run type need?
            cores = []
            
            # NAIVELY, the first method gets the fastest option, the next one gets the second fastest etc
            # But care, if one method gets one threaded gear there are no more cores for other threaded gears so these are removed
            for idx,r in enumerate(fastest_run_types):

                # if repeats
                if r in final_run_types:
                    # Get next one
                    while True:
                        task_objects[idx]._run_types.pop(r)    
                        r = task_objects[idx]._get_fastest_run_type()
                        
                        if r.startswith("Threaded") and self.threaded:
                            pass
                        elif r not in final_run_types: 
                            break

                if r.startswith("Threaded"):
                    c = -1
                    self.threaded = True
                else:
                    c = 1
                    
                cores.append(c)
                final_run_types.append(r)

            print(final_run_types, cores)

            futures = []
            for idx, r in enumerate(final_run_types):
                
                c = cores[idx]
                if c < 0:
                    c = self.total_cpu_cores-(len(final_run_types)-1)
                exec = ProcessPoolExecutor(max_workers=c)
                print(step,r,c)
                future_obj = exec.submit(task_objects[idx].run,*inputs,run_type=r)
                futures.append(future_obj)

            wait(futures)

            print("ENDING STEP", step)

if __name__=="__main__":

    dummy_img_1 = np.ones((1000,1000,100)).astype(np.float32)
    dummy_img_2 = dummy_img_1 * 2

    tasks = [ShiftAndMagnify,ShiftAndMagnify,ShiftAndMagnify,ShiftAndMagnify,ShiftAndMagnify,ShiftAndMagnify,ShiftAndMagnify]
    order = [0,1,1,1,1,2]
    inputs = [dummy_img_1,100,100,1,1]
    w = Workflow(tasks=tasks,task_input=inputs, order=order)
    seer = Agent()

    seer.run_workflow(w)

