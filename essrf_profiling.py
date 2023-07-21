from nanopyx.methods.eSRRF_workflow import eSRRF
from nanopyx.liquid import RadialGradientConvergence, CRShiftAndMagnify, GradientRobertsCross, CRShiftScaleRotate
from nanopyx.data.download import ExampleDataManager
import numpy as np
import matplotlib.pyplot as plt
import time

frames = [10, 100, 250, 500, 1000]
dims = [10, 100, 250, 500, 1000]

frames = [500]
dims = [128]

runtypes = ["Threaded", "OpenCL_Apple M1"]

for frame in frames:
    for dim in dims:
        im = np.random.random((frame, dim, dim)).astype(np.float32)
        for interp1 in runtypes:
            for interp2 in runtypes:
                for interp3 in runtypes:
                    for gradrc in ["Threaded"]:
                        for rgc in ["OpenCL_Apple M1"]:
                            line = ""
                            start = time.time()
                            eSRRF(im, interp_runtype1=interp1, interp_runtype2=interp2, interp_runtype3=interp3, gradrc_runtype=gradrc, rgc_runtype=rgc).run()
                            end = time.time()
                            line += str(frame) + ";" + str(dim) + ";" + str(interp1) + ";" + str(interp2) + ";" + str(interp3) + ";" + str(gradrc) + ";" + str(rgc) + ";" + str(end - start) + "\n"
                            print(line)
                            with open("esrrf.csv", "a") as fl:
                                fl.write(line)
