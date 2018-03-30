#!/bin/bash -x

DATASET=membrane_caax_63x
N_ITER=50000
BUFFER_SIZE=30
BATCH_SIZE=24
RUN_DIR=saved_models/${DATASET}
PATH_DATASET_TRAIN_CSV=data/csvs/${DATASET}/train.csv
GPU_IDS=${1:-0}

# resize factor: images are 0.086 um/px, want 0.29 um/px => factors (1, 0.29655, 0.29655)

python train_model.py \
       --transform_signal fnet.transforms.normalize "fnet.transforms.Resizer((1, 0.29655, 0.29655))" \
       --transform_target fnet.transforms.normalize "fnet.transforms.Resizer((1, 0.29655, 0.29655))" \
       --n_iter ${N_ITER} \
       --path_dataset_csv ${PATH_DATASET_TRAIN_CSV} \
       --buffer_size ${BUFFER_SIZE} \
       --buffer_switch_frequency -1 \
       --batch_size ${BATCH_SIZE} \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS}
