DATA_DIR='/allen/aics/microscopy'
DATA_DIR_2='/allen/aics/assay-dev'

nvidia-docker run --rm -it \
	      -v $(cd "$(dirname ${BASH_SOURCE})"/.. && pwd):/root/projects/pytorch_fnet \
	      -v ${DATA_DIR}:${DATA_DIR}:ro \
	      -v ${DATA_DIR_2}:${DATA_DIR_2}:ro \
	      ${USER}/pytorch_ttf \
	      bash

