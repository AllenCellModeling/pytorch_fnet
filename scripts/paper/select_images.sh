#!/bin/bash -x
PATH_PREDICTIONS_CSVS="\
/allen/aics/modeling/cheko/projects/pytorch_fnet/results/3d/dna/test/predictions.csv \
/allen/aics/modeling/cheko/projects/pytorch_fnet/results/3d/dna_extended/test/predictions.csv \
/allen/aics/modeling/cheko/projects/pytorch_fnet/results/3d/membrane_caax_63x/test/predictions.csv \
"
PATH_SAVE_DIR=/allen/aics/modeling/cheko/projects/fnet_paper/figure_s2_v2

python scripts/paper/python/select_s2_images.py \
       -i ${PATH_PREDICTIONS_CSVS} \
       -o ${PATH_SAVE_DIR} \
       --specify 10 10 13 \
       --include_signal \
       --crop 224 336 \
       --overwrite
