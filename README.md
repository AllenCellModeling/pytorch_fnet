Pytorch Fluorescence Net
===============================

![Model](doc/multi_pred_b.png?raw=true "Model Architecture")

## Setup
Build the Docker image:  
```
cd docker
./build_pytorch_ttf.sh
```
Start Docker container:  
```
./start_docker.sh
```
## Start example training run
```
./scripts/test_run.sh
```

## Citation
If you find this code useful in your research, please consider citing the following paper:

    @article {Ounkomol216606,
      author = {Ounkomol, Chek and Fernandes, Daniel A. and Seshamani, Sharmishtaa and Maleckar, Mary M. and Collman, Forrest and Johnson, Gregory R.},
      title = {Three dimensional cross-modal image inference: label-free methods for subcellular structure prediction},
      year = {2017},
      doi = {10.1101/216606},
      publisher = {Cold Spring Harbor Laboratory},
      URL = {https://www.biorxiv.org/content/early/2017/11/09/216606},
      eprint = {https://www.biorxiv.org/content/early/2017/11/09/216606.full.pdf},
      journal = {bioRxiv}
    }

## Contact
Gregory Johnson
E-mail: gregj@alleninstitute.org
