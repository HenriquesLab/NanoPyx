import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np
import networkx as nx
from networkx import DiGraph

from nanopyx.liquid._le_interpolation_bicubic import ShiftAndMagnify as BCShiftAndMagnify
from nanopyx.liquid._le_interpolation_bicubic import ShiftScaleRotate as BCShiftScaleRotate
from nanopyx.liquid._le_interpolation_catmull_rom import ShiftAndMagnify as CRShiftAndMagnify
from nanopyx.liquid._le_interpolation_catmull_rom import ShiftScaleRotate as CRShiftScaleRotate
from nanopyx.liquid._le_interpolation_lanczos import ShiftAndMagnify as LZShiftAndMagnify
from nanopyx.liquid._le_interpolation_lanczos import ShiftScaleRotate as LZShiftScaleRotate
from nanopyx.liquid._le_interpolation_nearest_neighbor import ShiftAndMagnify as NNShiftAndMagnify
from nanopyx.liquid._le_interpolation_nearest_neighbor import ShiftScaleRotate as NNShiftScaleRotate
from nanopyx.liquid._le_mandelbrot_benchmark import MandelbrotBenchmark

STR_2_METHOD = {
                'BCShiftAndMagnify': BCShiftAndMagnify,
                'BCShiftScaleRotate':BCShiftScaleRotate,
                'CRShiftAndMagnify': CRShiftAndMagnify,
                'CRShiftScaleRotate':CRShiftScaleRotate,
                'LZShiftAndMagnify': LZShiftAndMagnify,
                'LZShiftScaleRotate': LZShiftScaleRotate,
                'NNShiftAndMagnify': NNShiftAndMagnify,
                'NNShiftScaleRotate': NNShiftScaleRotate,
                'MandelbrotBenchmark': MandelbrotBenchmark,
                }


class Workflow(DiGraph):

    def __init__(self, tasks, **attr):
        super().__init__(tasks, **attr)

        assert nx.is_directed_acyclic_graph(self), "Graph is not DAG"
        
        self.steps = list(nx.topological_generations(self))

    def get_callable(self, node_name:str):
        
        sanitized_name = node_name.split('_')[0]

        try: 
            return STR_2_METHOD[sanitized_name]
        except KeyError:
            print(f"No method named {sanitized_name}")
            return None


class LiquidAgent:
    
    ## James Pond ##

    def __init__(self, device_list:list) -> None:

        self.total_cpu_cores = os.cpu_count()
        self.available_cpu_cores = self.total_cpu_cores
 
    def run_workflow(self, w:Workflow) -> None:
        
        # Iterate through concurrent methods
        # source: https://dl.acm.org/doi/pdf/10.1145/321941.321951
        for jobs in w.steps:
            
            # jobs is a list with various methods to be ran with length n
            n = len(jobs)
            
            
            # Lets check allowed run types for each method
     
            # Lets take from the LE the estimated times for these methods

            
            
            
if __name__=="__main__":

    t = {'0':['1','2'], '1':['3'], '2':['3'], '3':['4','5','6'],'4':['7'],'5':['7'],'6':['7'],'7':['8']}
    w = Workflow(tasks=t)

    