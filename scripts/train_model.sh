#!/bin/bash -x

DATASET=${1:-dna}
N_ITER=50000
RUN_DIR="saved_models/${DATASET}"
PATH_DATASET_ALL_CSV="data/csvs/${DATASET}.csv"
PATH_DATASET_TRAIN_CSV="data/csvs/${DATASET}/train.csv"
GPU_IDS=${2:-0}

DATASET_KWARGS="{\
\"transform_signal\": [\"fnet.transforms.Normalize()\", \"fnet.transforms.Resizer((1, 0.37241, 0.37241))\"], \
\"transform_target\": [\"fnet.transforms.Normalize()\", \"fnet.transforms.Resizer((1, 0.37241, 0.37241))\"] \
}"
BPDS_KWARGS="{\
\"buffer_size\": 1 \
}"

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python scripts/python/split_dataset.py ${PATH_DATASET_ALL_CSV} "data/csvs" --train_size 0.75 -v
python train_model.py \
       --n_iter ${N_ITER} \
       --path_dataset_csv ${PATH_DATASET_TRAIN_CSV} \
       --dataset_kwargs "${DATASET_KWARGS}" \
       --bpds_kwargs "${BPDS_KWARGS}" \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS}
