"""
Customized Dataset class for loading both the trajectories and the transition graph for time series-based data
"""

import os
import time
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class SimDatasetOneSubTS(Dataset):

    def __init__(
        self,
        ds_str,
        size,
        file_type = 'train',
        graph_key = 'graph',
        include_diag = True,
        seed = 0,
        subject_id = 0
    ):
        """
        argvs:
        - ds_str: dataset identifier
        - size: total sample size used in the loader
        - file_type: one of train, val or test
        - subject_id: which subject to use
        - seed: which seed for the dataset? default to 0
        - graph_key: which key to use for extracting the graphs?
        """
        self.data_prefix = f'data_sim/{ds_str}_seed{seed}/{file_type}/data_'

        self.sample_size = size
        self.list_of_ids = list(range(size))

        self.graph_path = os.path.join('data_sim', f'{ds_str}_seed{seed}', f'{graph_key}_by_subject.npy')
        self.subject_id = subject_id

        self.include_diag = include_diag
        self._load_graph()

    def _load_graph(self):

        graph = np.load(self.graph_path)
        self._get_indices_to_extract(graph.shape[-1])
        self.graph = graph[self.subject_id,self.indices_to_extract[0], self.indices_to_extract[1]]

    def __len__(self):
        return len(self.list_of_ids)

    def _get_indices_to_extract(self, num_nodes):
    
        if self.include_diag:
            valid_loc = np.ones([num_nodes, num_nodes])
        else:
            valid_loc = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
        self.indices_to_extract = np.where(valid_loc)

    def __getitem__(self, index):

        """ generate one sample of the {data, graph} dictionary """
        idx = self.list_of_ids[index]
        ## load trajectories
        data = np.load(self.data_prefix + str(idx) + '.npy')[self.subject_id]
        data = torch.from_numpy(data).to(dtype=torch.float32).unsqueeze(0)
        graph = torch.from_numpy(self.graph).to(dtype=torch.float32)

        return data, graph


class SimDatasetMultiSubTS(Dataset):

    """ a dataset class that also loads the oracle graphs for multiple subject"""

    def __init__(
        self,
        ds_str,
        size,
        file_type = 'train',
        graph_key = 'graph',
        include_diag = True,
        seed = 0
    ):
        """
        argvs:
        - ds_str: dataset identifier
        - size: total sample size used in the loader
        - undirected: flag for denoting whether the graph is directed or undirected
        - file_type: one of train, val or test
        - graph_key: what's the key that corresponds to the graph?
        - seed: which seed for the dataset? default to 0
        """
        self.data_prefix = f'data_sim/{ds_str}_seed{seed}/{file_type}/data_'

        self.list_of_ids = list(range(size))

        ## directly load graphs
        graph_by_subject = np.load(os.path.join('data_sim', f'{ds_str}_seed{seed}', f'{graph_key}_by_subject.npy'))
        graph_bar = np.load(os.path.join('data_sim', f'{ds_str}_seed{seed}', f'{graph_key}_bar.npy'))

        self.include_diag = include_diag
        self._get_indices_to_extract(graph_bar.shape[-1])

        self.graph_bar = graph_bar[self.indices_to_extract[0], self.indices_to_extract[1]]
        self.graph_by_subject = graph_by_subject[:, self.indices_to_extract[0], self.indices_to_extract[1]]

    def __len__(self):
        return len(self.list_of_ids)

    def _get_indices_to_extract(self, num_nodes):
    
        if self.include_diag:
            valid_loc = np.ones([num_nodes, num_nodes])
        else:
            valid_loc = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
        self.indices_to_extract = np.where(valid_loc)

    def __getitem__(self, index):

        """ generate one sample of the data """

        ## load trajectories
        idx = self.list_of_ids[index]
        data = np.load(self.data_prefix + str(idx) + '.npy')

        data = torch.from_numpy(data).to(dtype=torch.float32)
        graph_by_subject = torch.from_numpy(self.graph_by_subject).to(dtype=torch.float32)
        graph_bar = torch.from_numpy(self.graph_bar).to(dtype=torch.float32)

        return data, graph_by_subject, graph_bar
