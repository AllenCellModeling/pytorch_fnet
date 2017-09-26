DATA_DIR='/allen/aics/microscopy'

nvidia-docker run --rm -it \
	      -v ${PWD%/*}:/root/projects/ttf \
	      -v ${DATA_DIR}:/root/projects/ttf/data/microscopy:ro \
	      ${USER}/pytorch_ttf \
	      bash

