# Creating symlinks to files on network drive
**NOTE: this README is intended for Allen Institute employees only.**


Run the following python script to create symlinks to CZI files on the network drive:
```shell
python scripts/paper/python/make_symlinks.py
```
The above assumes you in the main `pytorch_fnet` directory, but (in theory) the python script should work from anywhere.

The script will create directories in the `data` directory, each of which corresponds to a cellular structure/cell line. e.g.:
- `data/alpha_tubulin`
- `data/beta_actin`
- `data/desmoplakin`
- etc

Note that there should be no `dna` directory because images to train DNA models are the same files in the other structures' directories.
