Three dimensional cross-modal image inference: label-free methods for subcellular structure prediction
===============================

![Model](doc/PredictingStructures-1.jpg?raw=true "Model Architecture")

## System Requirements
Installing on linux is recommended (we have used Ubuntu 16.04)

An nVidia graphics card with >10GB of ram (we have used a NVidia Titan X Pascal) with current drivers installed (we have used nVidia driver version 375.39)

### Installation
Instructions follow docker based workflow, non docker versions are possible but instructions are not provided

## Install requirements
Estimated time 1 hour
 - install docker (https://docs.docker.com/install/) (we used `Docker version 17.09.1-ce, build 19e2cf6` )
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
 - download the images: 
 
```
wget http://downloads.allencell.org/publication-data/label-free-prediction/data.tar
tar -C ./data -xvf yourfile.tar 
```

This installation should take a few minutes on a standard computer.
### Start Docker container:  
```
./start_docker.sh
```
### Start example training run
```
./scripts/train_model.sh dna 0
```
This will split the dna dataset up into 25% test and 75% training images and run training on the test images. 
Should take ~16 hours, and the final output should be similar to this.  Your example output should be a saved model in `saved_models/dna` with a `run.log` file whose last entries look like this
```
$ tail run.log 
2018-02-06 16:40:24,520 - model saved to: saved_models/dna/model.p
2018-02-06 16:40:24,520 - elapsed time: 56481.3 s
2018-02-06 16:49:59,591 - BufferedPatchDataset buffer history: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
2018-02-06 16:49:59,592 - loss log saved to: saved_models/dna/losses.csv
2018-02-06 16:49:59,592 - model saved to: saved_models/dna/model.p
2018-02-06 16:49:59,592 - elapsed time: 57056.4 s
2018-02-06 16:59:31,301 - BufferedPatchDataset buffer history: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
2018-02-06 16:59:31,302 - loss log saved to: saved_models/dna/losses.csv
2018-02-06 16:59:31,302 - model saved to: saved_models/dna/model.p
2018-02-06 16:59:31,303 - elapsed time: 57628.1 s
```
and a `losses.csv` file whose last entries look like this

```
$ tail losses.csv 
49991,0.25850439071655273
49992,0.2647261321544647
49993,0.283742755651474
49994,0.311653733253479
49995,0.30210474133491516
49996,0.2369609922170639
49997,0.2907244861125946
49998,0.23749516904354095
49999,0.3207407295703888
50000,0.3556152284145355
```
You can train other models by replacing `dna` with the names of the structures dataset (alpha_tubulin, beta_actin, st6gal1, desmoplakin, dic_lamin_b1, dic_membrane, fibrillarin, lamin_b1, membrane, myosin_iib, sec61_beta, tom20, zo1). 

### run predictions on the test set data
```
./scripts/test_predict.sh dna 0
```
example prediction outputs should be places in `results/dna/test` and `results/dna/train`.  
```
pytorch_fnet/results/3d/dna/test$ ls
00  01	02  03	04  05	06  07	08  09	10  11	12  13	14  15	16  17	18  19	predictions.csv  predict_options.json
```

### recreating paper figures
If you want to run all the training and evaluation code necessary to reproduce the paper figures, run `./scripts/paper/run_all.sh`.  This should run training and predictions for all the models and datasets used in the paper. Note, this will take several days.

After running those scripts you should be able to run all the cells in the ipython notebook. First launch jupyter using `jupyter notebook` and then open `figures.ipynb` and run all the cells.

### recreate

## Instructions to run on your data
The most general solution is to implement a new PyTorch dataset object that is responsible for loading signal images (transmitted light) and target images (fluorescence) into a consistent format. See `fnet/data/tiffdataset.py` or `fnet/data/czidataset.py` as examples.  Our existing wrapper scripts will work if you make this dataset object have an `__init__` function can be correctly called with a simple keyword argument of path_csv, which points to a csv file (example: `data/csvs/mydata.csv`) that describes your dataset. You should implement `__get_item__(self,i)` to return a list of pytorch Tensor objects, where the first element is the signal data and the second element is the target image.  The Tensors should be of dimensions of `1,Z,Y,X`.  Place your new dataset object (example: MyDataSet.py) in `pytorch_fnet/fnet/data/`

If you have single channel tiff stacks for both input and target images, you can simply use our existing tiffdataset class with a csv that has columns labelled `path_target`, and `path_signal`, and whose elements are paths to where those images can be read.

Create a new training wrapper script that is a modification of `scripts/train_model.sh`, let's call it `train_mymodel.sh`.

```
#!/bin/bash -x

DATASET=${1:-dna}
N_ITER=50000
BUFFER_SIZE=30
BATCH_SIZE=24
RUN_DIR="saved_models/${DATASET}"
PATH_DATASET_ALL_CSV="data/csvs/${DATASET}.csv"
PATH_DATASET_TRAIN_CSV="data/csvs/${DATASET}/train.csv"
GPU_IDS=${2:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python scripts/python/split_dataset.py ${PATH_DATASET_ALL_CSV} "data/csvs" --train_size 0.75 -v
python train_model.py \
       --n_iter ${N_ITER} \
       --class_dataset MyDataSet
       --path_dataset_csv ${PATH_DATASET_TRAIN_CSV} \
       --buffer_size ${BUFFER_SIZE} \
       --buffer_switch_frequency 2000000 \
       --batch_size ${BATCH_SIZE} \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS}
```
Now to train your model on your dataset you would run (assuming you only have 1 GPU on slot 0)

```
./scripts/train_mymodel.sh mydata 0
```
This should save a trained model in `saved_models/mydata`, using 75/25 test/train split on your data, placing CSVs in `data/csvs/mydata/test.csv` and `data/csvs/mydata/train.csv` the reflect that split.  

You should modify `scripts/predict_model.sh` to reflect your new dataset object as well, saved as (`scripts/predict_mymodel.sh`)

```
#!/bin/bash -x

DATASET=${1:-dna}
MODEL_DIR=saved_models/${DATASET}
N_IMAGES=20
GPU_IDS=${2:-0}
TRANSFORM_TARGET=${3:-fnet.transforms.normalize}

SUFFIX=${4:-}

echo ${DATASET}${SUFFIX}

for TEST_OR_TRAIN in test train
do
  python predict.py \
	 --path_model_dir ${MODEL_DIR}${SUFFIX} \
         --class_dataset MyDataSet \
	 --path_dataset_csv data/csvs/${DATASET}/${TEST_OR_TRAIN}.csv \
	 --n_images ${N_IMAGES} \
	 --no_prediction_unpropped \
	 --path_save_dir results/${DATASET}${SUFFIX}/${TEST_OR_TRAIN} \
	 --gpu_ids ${GPU_IDS} \
	 --transform_target ${TRANSFORM_TARGET}
done
```
You can then run predictions on your dataset by running

```
scripts\predict_mymodel.sh mydata 0 
```
which will output into predictions in `results/mydata/test` and `results/mydata/train`.

## Citation
If you find this code useful in your research, please consider citing the following paper:

    @article {Ounkomol289504,
	author = {Ounkomol, Chawin and Seshamani, Sharmishtaa and Maleckar, Mary M and Collman, Forrest and Johnson, Gregory},
	title = {Label-free prediction of three-dimensional fluorescence images from transmitted light microscopy},
	year = {2018},
	doi = {10.1101/289504},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2018/03/28/289504},
	eprint = {https://www.biorxiv.org/content/early/2018/03/28/289504.full.pdf},
	journal = {bioRxiv}


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
