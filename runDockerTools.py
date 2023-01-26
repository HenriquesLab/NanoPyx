#!/bin/bash -c python3

import os
import sys

nb_path = os.path.join(os.getcwd(), "notebooks")
nanopyx_path = os.getcwd()

options = {
    "Build nanopyx docker-image": "sudo docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile -t henriqueslab/nanopyx:latest .",
    "Run jupyterlab from nanopyx docker-image": f"sudo docker run --rm --name nanopyx1 -p 8888:8888 -v {nanopyx_path}:/nanopyx henriqueslab/nanopyx:latest",  # potentially add --rm
    "Run bash from from nanopyx docker-image": "sudo docker run -it henriqueslab/nanopyx:v1 bash",  # potentially add --rm
    "Start jupyterlab from nanopyx docker-image (needs run first)": "sudo docker start nanopyx1",
    "Stop jupyterlab from nanopyx docker-image": "sudo docker stop nanopyx1",
    "Remove nanopyx docker-image": "sudo docker rm nanopyx1 || sudo docker image rm nanopyx || sudo docker image prune -f §|| sudo docker container prune -f || sudo docker system df",
    "Fix issues on mac": "rm ~/.docker/config.json"
}

print(
        """
  _  _               ___          
 | \| |__ _ _ _  ___| _ \_  ___ __
 | .` / _` | ' \/ _ \  _/ || \ \ /
 |_|\_\__,_|_||_\___/_|  \_, /_\_\\
                         |__/  
    |-- NanoPyx --| Nanoscopy Library...
    """
)

if len(sys.argv) > 1:
    selection = int(sys.argv[1]) - 1

else:
    # print the options
    print("(⌐⊙_⊙) what do you want to do?")
    for i, option in enumerate(options.keys()):
        print("{}. {}: [CMD]> {}".format(i + 1, option, options[option]))

    # get the user's selection
    selection = int(input("Enter your selection: ")) - 1

# print the selected option
cmd = list(options.values())[selection]
print(f"- Running command: {repr(cmd)}")
os.system(cmd)
print("\n(•_•) ( •_•)>⌐■-■\n(⌐■_■) Ready to rock!!")
