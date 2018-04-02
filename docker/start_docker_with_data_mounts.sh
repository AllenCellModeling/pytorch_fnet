DATA_DIR_0='/allen/aics/modeling/data'
DATA_DIR_1='/allen/aics/microscopy'
DATA_DIR_2='/allen/aics/assay-dev'

nvidia-docker run --rm -it \
	      -v $(cd "$(dirname ${BASH_SOURCE})"/.. && pwd):/root/projects/pytorch_fnet \
	      -v ${DATA_DIR_0}:${DATA_DIR_0}:ro,rslave \
	      -v ${DATA_DIR_1}:${DATA_DIR_1}:ro,rslave \
	      -v ${DATA_DIR_2}:${DATA_DIR_2}:ro,rslave \
	      ${USER}/pytorch_fnet \
	      bash

