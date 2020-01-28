# Label-free prediction of three-dimensional fluorescence images from transmitted-light microscopy

[![Build Status](https://github.com/AllenCellModeling/pytorch_fnet/workflows/Build%20Master/badge.svg)](https://github.com/AllenCellModeling/pytorch_fnet/actions)
[![Documentation](https://github.com/AllenCellModeling/pytorch_fnet/workflows/Documentation/badge.svg)](https://allencellmodeling.github.io/pytorch_fnet/)
![Combined outputs](./resources/PredictingStructures-1.jpg?raw=true "Combined outputs")

## Support

This code is in active development and is used within our organization. We are currently not supporting this code for external use and are simply releasing the code to the community AS IS. The community is welcome to submit issues, but you should not expect an active response.

For the code corresponding to our Nature Methods paper, please use the `release_1` branch [here](https://github.com/AllenCellModeling/pytorch_fnet/tree/release_1).

## System requirements

We recommend installation on Linux and an NVIDIA graphics card with 12+ GB of RAM (e.g., NVIDIA Titan X Pascal) with the latest drivers installed.

## Installation

- We recommend an environment manager such as [Conda](https://docs.conda.io/en/latest/miniconda.html).
- Install Python 3.6+ if necessary.
- All commands listed below assume the bash shell.
- Clone and install the repo:

```shell
git clone https://github.com/AllenCellModeling/pytorch_fnet.git
cd pytorch_fnet
pip install .
```

- If you would like to instead install for development:

```shell
pip install -e .[dev]
```

- If you want to run the demos in the examples directory:

```shell
pip install .[examples]
```

## Demo on Canned AICS Data
This will download some images from our [Integrated Cell Quilt repository](https://open.quiltdata.com/b/allencell/tree/aics/pipeline_integrated_cell/) and start training a model 
```shell
cd examples
python download_and_train.py
```
When training is complete, you can predict on the held-out data with
```shell
python predict.py
```

## Command-line tool

Once the package is installed, users can train and use models through the `fnet` command-line tool. To see what commands are available, use the `-h` flag.

```shell
fnet -h
```

The `-h` flag is also available for all `fnet` commands. For example,

```shell
fnet train -h
```

## Train a model

Model training is done through the the `fnet train` command, which requires a json indicating various training parameters. e.g., what dataset to use, where to save the model, how the hyperparameters should be set, etc. To create a template json:

```shell
fnet train --json /path/to/train_options.json
```

Users are expected to modify this json to suit their needs. At a minimum, users should verify the following json fields and change them if necessary:

- `"dataset_train"`: The name of the training dataset.
- `"path_save_dir"`: The directory where the model will be saved. We recommend that the model be saved in the same directory as the training options json.

Once any modifications are complete, initiate training by repeating the above command:

```shell
fnet train --json /path/to/train_options.json
```

Since this time the json already exists, training should commence.

## Perform predictions with a trained model

User can perform predictions using a trained model with the `fnet predict` command. A path to a saved model and a data source must be specified. For example:

```shell
fnet predict --json path/to/predict_options.json
```

As above, users are expected to modify this json to suit their needs. At a minimum, populate the following fields and/or copy and paste corresponding dataset values from `/path/to/train_options.json`

e.g.:
```shell
    "dataset_kwargs": {
        "col_index": "Index",
        "col_signal": "signal",
        "col_target": "target",
        "path_csv": "path/to/my/train.csv",
    ...
    "path_model_dir": [
        "models/model_0"
    ],
    "path_save_dir": "path/to/predictions/dir",
```

This will use the model save `models/dna` to perform predictions on the `some.dataset` dataset. To see additional command options, use `fnet predict -h`.

Once any modifications are complete, initiate training by repeating the above command:

```shell
fnet predict --json path/to/predict_options.json
```

## Citation

If you find this code useful in your research, please consider citing our manuscript in Nature Methods:

```
@article{Ounkomol2018,
  doi = {10.1038/s41592-018-0111-2},
  url = {https://doi.org/10.1038/s41592-018-0111-2},
  year  = {2018},
  month = {sep},
  publisher = {Springer Nature America,  Inc},
  volume = {15},
  number = {11},
  pages = {917--920},
  author = {Chawin Ounkomol and Sharmishtaa Seshamani and Mary M. Maleckar and Forrest Collman and Gregory R. Johnson},
  title = {Label-free prediction of three-dimensional fluorescence images from transmitted-light microscopy},
  journal = {Nature Methods}
}
```

## Contact

Gregory Johnson  
E-mail: <gregj@alleninstitute.org>

## Allen Institute Software License

Allen Institute Software License – This software license is the 2-clause BSD license plus clause a third clause that prohibits redistribution and use for commercial purposes without further permission.   
Copyright © 2018. Allen Institute.  All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.  
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.  
3. Redistributions and use for commercial purposes are not permitted without the Allen Institute’s written permission. For purposes of this license, commercial purposes are the incorporation of the Allen Institute's software into anything for which you will charge fees or other compensation or use of the software to perform a commercial service for a third party. Contact terms@alleninstitute.org for commercial licensing opportunities.  

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
