import random

import torch
from torch.functional import Tensor

from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data


dataset = Planetoid(root='./datasets', name='Cora')

graph = dataset.data

print(graph.num_nodes)

class Walker:

    def __init__(self, G, walker='random_walk') -> None:
        super().__init__()

        self.G: Data = G
        self.walker = walker

    def _find_nbrs(self, src_node) -> Tensor:

        nbrs: Tensor = self.G.edge_index[1, torch.where(self.G.edge_index[0]==src_node)[0]]
        return nbrs.unique()

    def _simulate_walks(self, nodes, num_walks, walk_length):
        
        walks = []

        for _ in range(num_walks):
            # shuffle the nodes set
            nodes = nodes[torch.randperm(nodes.size(0))]
            for v in nodes:
                walks.append(self.deep_walk(walk_length, v))

        return walks

    def deep_walk(self, walk_length, start_node):

        walks = [start_node]

        while len(walks) < walk_length:

            cur_node = walks[-1]
            cur_nbrs = self._find_nbrs(cur_node)
            if cur_nbrs.size() == torch.Size([0]):
                break # no nbrs
            else:
                walks.append(int(random.choice(cur_nbrs)))

        return walks

walker = Walker(graph)
nodes = torch.IntTensor([0,1])
walks = walker._simulate_walks(nodes, 2, 5)
print(walks)

