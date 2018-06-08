#!/bin/bash -x

BUFFER_SIZE=1
N_ITER=32
RUN_DIR="saved_models/TEST"
GPU_IDS=${1:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

rm -rf ${RUN_DIR}
python train_model.py \
       --n_iter ${N_ITER} \
       --path_dataset_val_csv /some/path/data.csv \
       --class_dataset DummyChunkDataset \
       --buffer_size ${BUFFER_SIZE} \
       --buffer_switch_frequency -1 \
       --path_run_dir ${RUN_DIR} \
       --interval_save 10 \
       --iter_checkpoint 10 12 31 \
       --gpu_ids ${GPU_IDS}


