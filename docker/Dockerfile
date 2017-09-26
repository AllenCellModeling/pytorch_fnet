from pytorch-cudnnv6

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

# Jupyter Notebook config
COPY jupyter_notebook_config.py /root/.jupyter/
EXPOSE 9998

# Install aicsimage reqs
# ADD ./aicsimage/requirements.txt /root/tmp/aicsimage_reqs.txt
# RUN pip install -r /root/tmp/aicsimage_reqs.txt

WORKDIR "/root/projects/ttf"
