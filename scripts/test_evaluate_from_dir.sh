#!/bin/bash -x

SAVE_DIR="saved_models/EVAL_DIR"
PREDICTIONS_DIR="/root/allen/aics/modeling/cheko/projects/pytorch_fnet/results/2d/"
SAVE_ERROR_MAPS="TRUE"
GPU_IDS=${1:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

rm -rf ${RUN_DIR}
python evaluate_model.py \
       --predictions_dir ${PREDICTIONS_DIR} \
       --path_save_dir ${SAVE_DIR} \
       --save_error_maps ${SAVE_ERROR_MAPS}
