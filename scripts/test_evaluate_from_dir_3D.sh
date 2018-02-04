#!/bin/bash -x

SAVE_DIR="saved_models/EVAL_DIR_3D"
PREDICTIONS_DIR="/root/allen/aics/modeling/cheko/projects/pytorch_fnet/results/3d/"
SAVE_ERROR_MAPS="TRUE"
OVERWRITE="FALSE"
GPU_IDS=${1:-0}

python evaluate_model.py \
       --predictions_dir ${PREDICTIONS_DIR} \
       --path_save_dir ${SAVE_DIR} \
       --save_error_maps ${SAVE_ERROR_MAPS} \
       --overwrite ${OVERWRITE} \
