FROM pytorch

RUN apt update
RUN apt install -y vim
RUN apt install -y libx11-6
RUN apt install -y imagemagick

RUN pip install jupyter
RUN pip install natsort
RUN pip install pillow
RUN pip install matplotlib
RUN pip install torchvision
RUN pip install pandas
RUN pip install requests

# additional aicsimage reqs
RUN pip install scikit-image
RUN pip install nose
RUN pip install tifffile

# Jupyter Notebook config
COPY jupyter_notebook_config.py /root/.jupyter/
EXPOSE 9998

WORKDIR "/root/projects"
