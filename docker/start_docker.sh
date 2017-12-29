nvidia-docker run --rm -it \
	      -v $(cd "$(dirname ${BASH_SOURCE})"/.. && pwd):/root/projects/pytorch_fnet \
	      ${USER}/pytorch_fnet \
	      bash

