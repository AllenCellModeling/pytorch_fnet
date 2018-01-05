FROM pytorch

RUN apt update && apt install -y \
vim \
libx11-6 \
imagemagick \ 
 && rm -rf /var/lib/apt/lists/* 

# Jupyter Notebook config
COPY docker/jupyter_notebook_config.py /root/.jupyter/
EXPOSE 9998

RUN conda config --add channels https://repo.continuum.io/pkgs/main

RUN conda install -y jupyter=1.0.0 \
    matplotlib=2.1.1 \
    pandas=0.21.1 \
    tifffile=0.12.1 \

WORKDIR "/root/projects"
COPY . /root/projects/pytorch_fnet
WORKDIR "/root/projects/pytorch_fnet"



RUN python setup.py develop
