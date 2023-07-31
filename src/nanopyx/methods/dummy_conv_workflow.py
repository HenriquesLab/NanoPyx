from .workflow import Workflow
from ..liquid import Convolution2D as Conv

import numpy as np

def ConvolutionWorkflow(image, kernel):

    _conv = Workflow((Conv(),(image, kernel),{}),)
    
    
    return _conv
    