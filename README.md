# Are Graph Neural Networks Miscalibrated?

In this repository we make available the Friendster dataset used in our
paper:

Teixeira, L., Jalaian, B., & Ribeiro, B., (2019). Are Graph Neural
Networks Miscalibrated? (arXiv preprint in preparation)

<!--If you use the data or code from this repo in your own code, please cite-->
<!--our paper: -->
<!--```tex-->
<!--@article{teixeira2019,-->
<!--    title={Are Graph Neural Networks Miscalibrated?},-->
<!--    author={Leonardo Teixeira and Brian Jalaian and Bruno Ribeiro},-->
<!--    journal={arXiv preprint arXiv:XXX.XXXXX},-->
<!--    year={2019}-->
<!--}-->
<!--```-->

## Friendster Dataset

The Friendster dataset used in our paper is available in the folder
`Friendster`. We also provide the Train, Validation and Test split used
in the paper, as well as a Python class to facilitate the usage with the
[PyTorch Geometric][1] library.

We provide the dataset in *HDF5* format and the data split as a *NumPy
NPY format* file. We also provide a python class that is compatible with
the PyTorch Geometric framework, which automatically downloads the data
and split.

### PyTorch Geometric version

If you use the [PyTorch Geometric][1] library, we provide a Python class
that can be used to access our Friendster dataset. It can automatically
download and provide access to the Friendster graph (and the data split
used in the paper) as an `Dataset` class from PyTorch Geometric.

The necessary libraries are:
- [NumPy](https://www.numpy.org/) (numpy)
- [PyTorch](https://pytorch.org/) (torch)
- [PyTorch Geometric](https://rusty1s.github.io/) (torch_geometric)
- [HDF5 for Python](https://www.h5py.org/) (h5py)

Please, refer to their documentation for installation instructions (in
particular for PyTorch and PyTorch Geometric). This code was tested with
PyTorch 1.0.1, PyTorch Geometric 1.0.2, NumPy 1.15 and h5py 2.9.

#### Usage:

Using the provided class is illustrated in the following snippet. The
class takes care of downloading the data automatically.

```python 
from friendster import Friendster

# Download the dataset to the folder: './Friendster-25K'
dataset = Friendster(root="./Friendster-25K/")

# This dataset has a single graph
graph = dataset[0]

print(f"Friendster dataset: {graph.num_nodes} nodes")

# The data splits can be accessed as:
train_mask = graph.train_mask
validation_mask = graph.validation_mask
test_mask = graph.test_mask
```

A full example is given in the file `example.py`, where we run a GCN
model on the Friendster dataset.

### HDF5 version

The dataset is available in the HDF5 format in the file
`friendster_25K.h5`.

This file has the following HDF5 Datasets:
- `adjacency`: The adjacency matrix, with `n_nodes` rows. Each entry `u`
  is an array with the neighbors of `u`.
- `features`: The feature matrix, of shape `(n_nodes, n_features)`. Each
  entry `u` has the features of node `u`.
- `target`: The target label of the ndoes, of shape `n_nodes`. Each
  entry `u` has the integer that represents the label of node `u`.
- `feature_names`: The names of each of the features. Has `n_features`
  entries.
- `target_names`: The names of each label.

Using the `h5py` library, the data can be loaded as: 
```python
import h5py

dataset = h5py.File("./friendster_25K.h5")
A = dataset["adjacency"][:]  # Adjacency list
X = dataset["features"][:]  # Feature matrix
y = dataset["target"][:]  # Node labels
```

The data split is available in the file `friendster_25K.split.npz`. This
can be loaded with:

```python
import numpy as np

data = np.load("friendster_25K.split.npz")
train_nodes = data["arr_0"][0]
validation_nodes = data["arr_0"][1]
test_nodes = data["arr_0"][2]
```

[1]: https://github.com/rusty1s/pytorch_geometric