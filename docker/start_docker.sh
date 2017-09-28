DATA_DIR='/allen/aics/microscopy'

nvidia-docker run --rm -it \
	      -v $(cd "$(dirname ${BASH_SOURCE})"/.. && pwd):/root/projects/pytorch_fnet \
	      -v ${DATA_DIR}:${DATA_DIR}:ro \
	      ${USER}/pytorch_ttf \
	      bash

