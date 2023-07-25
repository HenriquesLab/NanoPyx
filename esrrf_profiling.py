from nanopyx.methods.workflow import Workflow
from nanopyx.liquid import CRShiftAndMagnify, RadialGradientConvergence, GradientRobertsCross
from nanopyx.methods.eSRRF_workflow import eSRRF, eSRRF_2, eSRRF_3
from nanopyx.core.utils.timeit import timeit2

import time
import numpy as np
from tifffile import imread, imwrite
from matplotlib import pyplot as plt

frames = [1000]
dims = [100]
runtypes = ["Threaded", "OpenCL_NVIDIA GeForce RTX 4090"]

im = imread("im.tif")

# for f in frames:
#     for dim in dims:
#         for interp1 in runtypes:
#             for interp2 in runtypes:
#                 for interp3 in runtypes:
#                     for rgc in runtypes:
#                         img = np.random.random((f, dim, dim)).astype(np.float32)
#                         print(img.shape)
#                         start = time.time()
#                         eSRRF(img, interp_runtype1=interp1, interp_runtype2=interp2,
#                               interp_runtype3=interp3, gradrc_runtype="Threaded",
#                               rgc_runtype=rgc).run()
#                         end = time.time()
#                         runtime = end - start
#                         bench = str(f) + "," + str(dim) + ";" + interp1 + ";" + "Threaded" + ";" + interp2 + ";" + interp3 + ";" + rgc + ";" + str(runtime) + "\n"
#                         print(bench)
                        
#                         with open("eSRRF.csv", "a") as fl:
#                             fl.writelines(bench)

#img = np.random.random((1000,500,500)).astype(np.float32)
img = im

# @timeit2
# def run(img):
#     out = eSRRF(img, interp_runtype1="OpenCL_NVIDIA GeForce RTX 4090", interp_runtype2="OpenCL_NVIDIA GeForce RTX 4090",
#                 interp_runtype3="OpenCL_NVIDIA GeForce RTX 4090", gradrc_runtype="OpenCL_NVIDIA GeForce RTX 4090",
#                 rgc_runtype="OpenCL_NVIDIA GeForce RTX 4090").run()
#     return np.mean(out[0], axis=0)

# plt.imshow(run(img))
# plt.show()

# @timeit2
# def run2(img):
#     out = eSRRF_2(img, interp_runtype1="OpenCL_NVIDIA GeForce RTX 4090", interp_runtype2="OpenCL_NVIDIA GeForce RTX 4090",
#                 gradrc_runtype="OpenCL_NVIDIA GeForce RTX 4090",
#                 rgc_runtype="OpenCL_NVIDIA GeForce RTX 4090").run()
#     return np.mean(out[0], axis=0)

# plt.imshow(run2(img))
# plt.show()

@timeit2
def run3(img):
    out = eSRRF_3(img).run()
    return np.mean(out[0], axis=0)

run3(im)

# plt.imshow(run3(img))
# plt.show()

# for i in range(10):
#     run3(img)
