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

FROM --platform=linux/amd64 ubuntu:22.04 AS nanopyx

ENV TZ=Europe/London
ENV LANG=C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -qqy  \
        build-essential \
        python3.9 \
        python3-pip \
        git \
        # for latex labels
        cm-super \
        dvipng \
        # for matplotlib anim
        ffmpeg \
        # for javascript
        nodejs npm \
        # below is for napari - ref: https://github.com/napari/napari/blob/main/dockerfile
        #mesa-utils \
        #libgl1-mesa-glx \
        #libglib2.0-0 \
        #libfontconfig1 \
        #libxrender1 \
        #libdbus-1-3 \
        #libxkbcommon-x11-0 \
        #libxi6 \
        #libxcb-icccm4 \
        #libxcb-image0 \
        #libxcb-keysyms1 \
        #libxcb-randr0 \
        #libxcb-render-util0 \
        #libxcb-xinerama0 \
        #libxcb-xinput0 \
        #libxcb-xfixes0 \
        #libxcb-shape0 \
        && apt-get clean

# Install yarn for handling npm packages
RUN npm install --global yarn
# Enable yarn global add:
ENV PATH="$PATH:$HOME/.yarn/bin"

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir cython

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
    jupyterlab_tabnine 
    #jupyter-resource-usage

# Set jupyter theme
RUN jupyter labextension install jupyterlab_onedarkpro

# see https://github.com/napari/napari/blob/main/dockerfile
# install napari release version
# RUN pip3 install --no-cache-dir napari[all]

# copy content of current directory to inside docker container
ENV BUILD_DIR=/tmp/build
COPY . ${BUILD_DIR}
RUN pip3 install ${BUILD_DIR}

# Start Jupyterlab port & cmd
EXPOSE 8888
ENV NB_DIR=/notebooks
# RUN mkdir -p /tmp/notebooks
# COPY ./notebooks ${NB_DIR}
CMD jupyter lab --ip=* --port=8888 --no-browser --notebook-dir=${NB_DIR} --allow-root --ResourceUseDisplay.track_cpu_percent=True
