# Copyright 2019 Leonardo V. Teixeira
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url

# ==================================================== #


class Friendster(InMemoryDataset):
    """The social network "Friendster".

    Nodes represent users, edges represent friendships. There are close
    to 700 node attributes, a combination of categorical (in one-hot-encoding)
    and numerical attributes, which include: age, gender, occupation, TV and
    music interests, college and location, among others. The label is the
    relationship status, which take the following values:
        0: Domestic Partner
        1: In A Relationship
        2: Married
        3: Single
        4: NA

    Training, validation and test splits are given by binary masks and they
    exclude nodes for which the target label is unknown (i.e. value of 4 = NA).

    The following versions are available:
        '25K': Subset of the graph with 25K labeled nodes.

    Parameters
    ----------
    root : string
        Root directory where the dataset should be saved.
    name : {'25K'}
        The name of the version of the dataset to use.
    transform : (callable, optional)
        A function/transform that takes in an :obj:`torch_geometric.data.Data`
        object and returns a transformed version. The data object will be
        transformed before every access. (default: None)
    pre_transform : (callable, optional)
        A function/transform that takes in an :obj:`torch_geometric.data.Data`
        object and returns a transformed version. The data object will be
        transformed before being saved to disk. (default: None)
    """

    url = "https://github.com/PurdueMINDS/GNNsMiscalibrated/raw/master/Friendster"
    name_map = {
        '25K': '25K',
    }

    def __init__(self, root, name="25K", transform=None, pre_transform=None):
        self.name = name
        if name in self.name_map:
            self.suffix = self.name_map[name]
        else:
            raise ValueError(f"Unsupported dataset name: '{name}'")

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        fnames = []
        suffix = self.name_map[self.name]
        fnames.append(f"friendster_{suffix}.h5")
        fnames.append(f"friendster_{suffix}.split.npz")
        return fnames

    @property
    def processed_file_names(self):
        return [f'data_{self.name}.pt']

    def download(self):
        # Download to `self.raw_dir`.
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        try:
            import h5py
        except ImportError:
            raise RuntimeError('To use Friendster dataset, the `h5py` library is required.')
        else:
            fname = self.raw_paths[0]
            h5 = h5py.File(fname, 'r')

            x = torch.from_numpy(h5['features'][:]).float()
            y = torch.from_numpy(h5['target'][:]).long()
            edge_index = [[], []]
            for u, nb_u in enumerate(h5["adjacency"]):
                for v in nb_u:
                    edge_index[0].append(int(u))
                    edge_index[1].append(int(v))
            edge_index = torch.LongTensor(edge_index)
            h5.close()

            data = Data(x=x, y=y, edge_index=edge_index)

            fname = self.raw_paths[1]
            splits = list(np.load(fname).values())
            train_mask = torch.zeros(len(x), dtype=torch.uint8)
            train_mask[splits[0][0]] = 1
            validation_mask = torch.zeros(len(x), dtype=torch.uint8)
            validation_mask[splits[0][1]] = 1
            test_mask = torch.zeros(len(x), dtype=torch.uint8)
            test_mask[splits[0][2]] = 1

            data["train_mask"] = train_mask
            data["validation_mask"] = validation_mask
            data["test_mask"] = test_mask

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data)]
            else:
                data_list = [data]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

