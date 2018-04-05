#!/bin/bash -x

DATASET=timelapse_wt2_s2
PATH_DATASET_CSV=data/csvs/${DATASET}.csv
MODEL_DIR="saved_models/dna saved_models/dna_extended saved_models/lamin_b1 saved_models/lamin_b1_extended saved_models/fibrillarin saved_models/tom20 saved_models/sec61_beta"
PATH_SAVE_DIR=results/timelapse/${DATASET}
N_IMAGES=-1
GPU_IDS=${1:-0}

python predict.py \
       --class_dataset TiffDataset \
       --path_model_dir ${MODEL_DIR} \
       --path_dataset_csv ${PATH_DATASET_CSV} \
       --n_images ${N_IMAGES} \
       --path_save_dir ${PATH_SAVE_DIR} \
       --no_prediction_unpropped \
       --gpu_ids ${GPU_IDS}
