nvidia-docker run --rm -it \
	      -v ${PWD%/*}:/root/projects/ttf \
	      ${USER}/pytorch_ttf \
	      bash

