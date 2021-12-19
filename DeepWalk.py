import itertools
import random
from typing import List
import argparse
from warnings import WarningMessage

from gensim.models import Word2Vec

import torch
from torch.functional import Tensor
from torch.utils.data.dataset import DFIterDataPipe

from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

from joblib import Parallel, delayed


dataset = Planetoid(root='./datasets', name='Cora')

graph = dataset.data

print(graph.num_nodes)

class DeepWalker:

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

class DeepWalk:

    def __init__(self, G, num_walks, walk_length, workers, w2v_args) -> None:
        
        self.G: Data = G
        self.walker = DeepWalker(G)

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks,
            walk_length=walk_length,
            workers=workers
        ).tolist()
        print("DeepWalk Finish!")

        self.w2v_args = w2v_args
        self.w2v_model = None

        self._embeddings = {}

    def train(self):

        print("Learning Embedding Vectors...")
        self.w2v_model = Word2Vec(
            sentences=self.sentences,
            vector_size=self.w2v_args.embed_size,
            sg=1, # skip gram
            hs=1, # deepwalk use Hierarchical Softmax
            workers=self.w2v_args.workers,
            epochs=self.w2v_args.epochs,
            window=self.w2v_args.window_size
        )

        print("Done!")

    def get_embeddings(self, ):

        if self.w2v_model is None:
            raise NotImplementedError

        for word in range(self.G.num_nodes):
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

def args_register():

    w2v_parser = argparse.ArgumentParser()
    w2v_parser.add_argument('--embed_size', default=128)
    w2v_parser.add_argument('--workers', default=2)
    w2v_parser.add_argument('--epochs', default=5)
    w2v_parser.add_argument('--window_size', default=5)

    deepwalk_parser = argparse.ArgumentParser()
    deepwalk_parser.add_argument('--walk_length', default=10)
    deepwalk_parser.add_argument('--num_walks', default=80)
    deepwalk_parser.add_argument('--workers', default=5)

    w2v_args = w2v_parser.parse_args()
    deepwalk_args = deepwalk_parser.parse_args()

    return w2v_args, deepwalk_args

w2v_args, deepwalk_args = args_register()

model = DeepWalk(
    G=graph, 
    walk_length=deepwalk_args.walk_length,
    num_walks=deepwalk_args.num_walks,
    workers=deepwalk_args.workers,
    w2v_args=w2v_args
)
model.train()
embeddings = model.get_embeddings()
    

    

