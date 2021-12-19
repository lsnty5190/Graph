import itertools
import random
from typing import List
from joblib.parallel import delayed

import torch
from torch.functional import Tensor

from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

from joblib import Parallel


dataset = Planetoid(root='./datasets', name='Cora')

graph = dataset.data

print(graph.num_nodes)

class Walker:

    def __init__(self, G) -> None:

        self.G: Data = G

    def _find_nbrs(self, src_node) -> Tensor:

        nbrs: Tensor = self.G.edge_index[1, torch.where(self.G.edge_index[0]==src_node)[0]]
        return nbrs.unique()

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        nodes: List = torch.arange(self.G.num_nodes)

        partition_num = lambda num, workers: [num//workers]*workers if num % workers == 0 else [num//workers]*workers + [num % workers]

        out = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length)
            for num in partition_num(num_walks, workers)
        )

        walks = list(itertools.chain(*out))

        return torch.tensor(walks)

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
                walks.append(random.choice(cur_nbrs))

        return walks

walker = Walker(graph)
nodes = torch.IntTensor([0,1])
walks = walker.simulate_walks(num_walks=5, walk_length=5, workers=2)
print(walks)


