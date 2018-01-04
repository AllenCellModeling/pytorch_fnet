Pytorch Fluorescence Net
===============================

![Model](doc/PredictingStructures-1.jpg?raw=true "Model Architecture")

## Setup
Installing on linux is recommended

### Prerequisites
Running on docker is recommended, though not required.
 - install pytorch on docker / nvidia-docker as in e.g. this guide: https://github.com/pytorch/pytorch/tree/v0.3.1
 - download the training images: **todo**

### Build the Docker image:  
```
docker build -t ${USER}/pytorch_fnet -f Dockerfile .
```
Note: You may need to edit the Dockerfile to point to the correct pytorch image.
### Start Docker container:  
```
./start_docker.sh
```
### Start example training run
```
./scripts/test_run.sh
```

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
