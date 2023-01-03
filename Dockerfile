# - generic docker commands
# docker run    # starts a new container
# docker ps     # shows running containers
# docker exec   # hops or "ssh" into a container
# docker stop   # stops a container
# docker rm     # removes a stopped container
# docker start  # start back up a stopped container
# - docker registry commands
# docker build  # builds an image from a Dockerfile
# docker images # lists the pulled or built images
# docker push   # pushes the image to a docker registry

# some help here https://blog.boltops.com/2018/04/19/docker-introduction-tutorial/ and https://docs.docker.com/engine/reference/builder/
# sudo docker build -t nanopyx .
# sudo docker run --rm -it nanopyx bash
# sudo docker run --name nanopyx1 --rm -p 8888:8888 -v ./notebooks:/opt/app/data nanopyx

# sudo docker ps
# sudo docker stop nanopyx1
# sudo docker rmi nanopyx

######################################
# Install base resources and nanopyx #
######################################

FROM python:3.10-slim AS nanopyx

ENV TZ=Europe/London
ENV LANG=C.UTF-8

RUN apt-get update && \
    apt-get install -qqy  \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    setuptools \
    cython \
    numpy \
    scipy \
    numba

# Install Jupyter
RUN pip3 install --no-cache-dir jupyter ipywidgets && \
    jupyter notebook --generate-config && \
    jupyter nbextension enable --py widgetsnbextension

# Install JupyterLab
RUN pip3 install --no-cache-dir jupyterlab && \
    jupyter serverextension enable --py jupyterlab

# Install Jupyter extensions
RUN pip3 install --no-cache-dir \ 
    jupyterlab-github \
    jupyterlab-topbar \
    jupyterlab-system-monitor \
    jupyterlab_tabnine \
    jupyterlab-simpledark

# copy content of current directory to inside docker container
ENV BUILD_DIR=/nanopyx
COPY . ${BUILD_DIR}
RUN pip3 install -e ${BUILD_DIR}

# test that we built correctly by running a pytest
RUN pytest ${BUILD_DIR}/tests/test_random_noise.py

# Start Jupyterlab port & cmd
EXPOSE 8888
ENV SHELL=/bin/bash
ENV NB_DIR=${BUILD_DIR}/notebooks
# RUN mkdir -p /tmp/notebooks
# COPY ./notebooks ${NB_DIR}
RUN echo 'Docker!' | passwd --stdin root
CMD jupyter lab --ip=* --port=8888 --no-browser --notebook-dir=${BUILD_DIR} --allow-root --ResourceUseDisplay.track_cpu_percent=True
