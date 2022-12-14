#!/bin/bash -c python3

import os

nb_path = os.path.join(os.getcwd(), "notebooks")

options = {
    "Build nanopyx extensions": "python setup.py build_ext --inplace",
}

# print the options
print("What do you want to do:")
for i, option in enumerate(options.keys()):
    print("{}. {}: [CMD]> {}".format(i+1, option, options[option]))

# get the user's selection
selection = int(input("Enter your selection: "))-1

# print the selected option
cmd = list(options.values())[selection]
print(f"- Running command: {repr(cmd)}")
os.system(cmd)
