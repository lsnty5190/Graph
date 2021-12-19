import random

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='./datasets', name='Cora')