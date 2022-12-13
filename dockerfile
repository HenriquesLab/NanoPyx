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

# some help here https://blog.boltops.com/2018/04/19/docker-introduction-tutorial/
# sudo docker build -t nanopyx .
# sudo docker stop nanopyx
# sudo docker ps
# sudo docker run --rm -it nanopyx bash
# sudo docker run --name nanopyx1 --rm -p 8888:8888 nanopyx --name nanopyx
# sudo docker rmi nanopyx

FROM --platform=linux/amd64 ubuntu:22.04 AS nanopyx

ENV TZ=Europe/London
ARG DEBIAN_FRONTEND=noninteractive

# install resources
RUN apt-get update && \
    apt-get install -qqy  \
        build-essential \
        python3.9 \
        python3-pip \
        git \
        && apt-get clean

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir cython

# copy content of current directory to inside docker container
COPY . . 
RUN pip3 install ./
#RUN python3 setup.py sdist bdist_wheel


# Install Jupyter
RUN pip3 install jupyter
RUN pip3 install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension

# Install JupyterLab
RUN pip3 install jupyterlab && jupyter serverextension enable --py jupyterlab
# Install extensions
RUN pip3 install \ 
    jupyterlab-github \
    jupyterlab-topbar \
    jupyterlab-system-monitor \
    jupyter-tabnine

ENV LANG=C.UTF-8

# Set jupyter theme
RUN pip3 install jupyterthemes && jt -t onedork

# Expose Jupyter port & cmd
EXPOSE 8888
RUN mkdir -p /opt/app/data
CMD jupyter lab --ip=* --port=8888 --no-browser --notebook-dir=/opt/app/data --allow-root
