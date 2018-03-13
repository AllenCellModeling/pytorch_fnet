FROM pytorch/pytorch:v0.2

RUN apt update && apt install -y \
vim \
libx11-6 \
imagemagick \ 
 && rm -rf /var/lib/apt/lists/* 

# Jupyter Notebook config
COPY docker/jupyter_notebook_config.py /root/.jupyter/
EXPOSE 9998

COPY . /root/projects/pytorch_fnet
WORKDIR "/root/projects/pytorch_fnet"
RUN pip install -e . 
