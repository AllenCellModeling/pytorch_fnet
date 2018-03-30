#!/bin/bash -x

DATASET=registration_2
PATH_DATASET_CSV=data/csvs/${DATASET}.csv
MODEL_DIR=saved_models/LMmed
N_IMAGES=-1
GPU_IDS=${1:-0}

PATH_SAVE_DIR=results/2d/${DATASET}
python predict.py \
       --class_dataset TiffDataset \
       --transform_signal fnet.transforms.normalize \
       --transform_target fnet.transforms.normalize \
       --path_model_dir ${MODEL_DIR} \
       --path_dataset_csv ${PATH_DATASET_CSV} \
       --n_images ${N_IMAGES} \
       --path_save_dir ${PATH_SAVE_DIR} \
       --propper_kwargs '{"action": "+", "mode": "reflect"}' \
       --no_prediction \
       --gpu_ids ${GPU_IDS}
