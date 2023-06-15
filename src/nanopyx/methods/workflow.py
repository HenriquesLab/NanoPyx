import numpy as np

#from nanopyx.liquid import BCShiftAndMagnify, BCShiftScaleRotate, NNShiftScaleRotate
from .liquid import BCShiftAndMagnify, BCShiftScaleRotate, NNShiftScaleRotate


class Workflow:
    
    """
    Workflow class that aggregates all the steps of the analysis and the corresponding functions.
    """

    def __init__(self, *args, arguments:dict) -> None:
        """
        Initialize the Workflow
        args: list of functions that will be executed in the order they are passed
        kwargs: dict of arguments that will be passed to the functions, the keys must match the function names, the values must be list of len 2 [*args, **kwargs]
        
        ARGUMENTS OR KEYWORD ARGUMENTS ARE "METHOD_RETURN_VALUE" they are substituted by their return value.
        """

        self.methods = [method() for method in args]
        self.method_arguments = arguments

        if list(self.method_arguments.keys()) != [method._designation for method in self.methods]:
            print(list(self.method_arguments.keys()))
            print([method._designation for method in self.methods])
            raise ValueError("The keys of the arguments dict must match the designation of the methods")

        self.returns = {}
        self.finished = False
        self.running = False

    def run(self,):

        """
        Run the workflow
        """

        self.running = True

        for method in self.methods:

            print(f"Running {method._designation}")

            method_name = method._designation
            args, kwargs = self.method_arguments[method_name]

            for idx, arg in enumerate(args):
                if isinstance(arg, str):
                    if "RETURN_VALUE" in arg:
                        args[idx] = self.returns[arg.split("_RETURN_VALUE")[0]] 
            for key in kwargs:
                if isinstance(kwargs[key], str):
                    if "RETURN_VALUE" in kwargs[key]:
                        kwargs[key] = self.returns[kwargs[key].split("_RETURN_VALUE")[0]]

            return_value = method.run(*args,**kwargs)
            self.returns[method_name] = return_value
        
        self.finished = True


if __name__ == "__main__":

    dummy_image = np.ones((100,100))

    w = Workflow(BCShiftScaleRotate,BCShiftAndMagnify,NNShiftScaleRotate, 
                arguments={"ShiftScaleRotate_BC": [[dummy_image,10,10,1,1,0],{}], 
                "ShiftMagnify_BC": [["ShiftScaleRotate_BC_RETURN_VALUE",-10,-10,1,1],{}],
                "ShiftScaleRotate_NN": [["ShiftMagnify_BC_RETURN_VALUE", 0,0,1,1,-np.pi],{}]})
    
    w.run()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(w.returns["ShiftScaleRotate_BC"][0])
    ax[1].imshow(w.returns["ShiftMagnify_BC"][0])
    ax[2].imshow(w.returns["ShiftScaleRotate_NN"][0])
    plt.show()