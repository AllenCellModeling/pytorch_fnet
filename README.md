Three dimensional cross-modal image inference: label-free methods for subcellular structure prediction
===============================

![Model](doc/PredictingStructures-1.jpg?raw=true "Model Architecture")

## System Requirements
Installing on linux is recommended (we have used Ubuntu 16.04)

An nVidia graphics card with >10GB of ram (we have used a NVidia Titan X Pascal) with current drivers installed (we have used nVidia driver version 375.39)

### Installation
Instructions follow docker based workflow, non docker versions are likely possible but instructions are not provided

## Install requirements
Estimated time 1 hour
 - install docker (https://docs.docker.com/install/)
 - install nvidia-docker (https://github.com/NVIDIA/nvidia-docker)
 - install git

## Installation
Estimated time 15 minutes, plus X minutes for data download
 - clone repository and build docker image
```
git clone https://github.com/AllenCellModeling/pytorch_fnet
cd pytorch_fnet
docker build -t ${USER}/pytorch_fnet -f Dockerfile .
```
 - download the test image dataset: **todo**
 
```
commands to download data
```

This installation should take a few minutes on a standard computer.
### Start Docker container:  
```
./start_docker.sh
```
### Start example training run
```
./scripts/test_run.sh
```
This will split the dataset up into test and training images and run training on the test images. 
Should take ~XX hours, and the final output should be similar to this
```
EXAMPLE OUTPUT OF SUCCESSFUL RUN OF TEST DATA
```

### run predictions on the test set data
```
./scripts/test_predict.sh
```
example prediction outputs should be places in ./results/

## Citation
If you find this code useful in your research, please consider citing the following paper:

    @article {Ounkomol216606,
      author = {Ounkomol, Chek and Fernandes, Daniel A. and Seshamani, Sharmishtaa and Maleckar, Mary M. and Collman, Forrest and Johnson, Gregory R.},
      title = {Three dimensional cross-modal image inference: label-free methods for subcellular structure prediction},
      year = {2017},
      doi = {10.1101/216606},
      publisher = {Cold Spring Harbor Laboratory},
      URL = {https://www.biorxiv.org/content/early/2017/11/09/216606},
      eprint = {https://www.biorxiv.org/content/early/2017/11/09/216606.full.pdf},
      journal = {bioRxiv}
    }

## Contact
Gregory Johnson
E-mail: gregj@alleninstitute.org

## License
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
