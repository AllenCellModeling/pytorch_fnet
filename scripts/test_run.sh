#!/bin/bash

N_ITER=8
RUN_DIR=saved_models/TEST
PATH_DATASET_CSV=data/csvs/test_run.csv
GPU_IDS=${1:--1}
FNET_MODEL_KWARGS="{\
\"nn_module\": \"tests.data.nn_test\" \
}"

if [ -d ${RUN_DIR} ]; then
    rm -r ${RUN_DIR}
fi
python scripts/train_model.py \
       --n_iter ${N_ITER} \
       --path_dataset_csv ${PATH_DATASET_CSV} \
       --path_dataset_val_csv ${PATH_DATASET_CSV} \
       --fnet_model_kwargs "${FNET_MODEL_KWARGS}" \
       --interval_save 6 \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS}
retval=$?
if [ $retval -ne 0 ]; then
    exit $retval
fi
rm -r ${RUN_DIR}
