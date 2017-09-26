DATA_DIR='/allen/aics/microscopy'

nvidia-docker run --rm -it \
	      -v ${PWD%/*}:/root/projects/ttf \
	      -v ${DATA_DIR}:${DATA_DIR}:ro \
	      ${USER}/pytorch_ttf \
	      bash

