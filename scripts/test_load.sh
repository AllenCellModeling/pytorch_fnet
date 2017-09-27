#!/bin/bash

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python test_model.py \
	--path_source saved_models/TEST \
	--n_images 4

