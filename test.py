import random
import torch

# x = torch.IntTensor([[1,2,3,1],[1,2,3,4]])
# print(x[1,torch.where(x[0,:]==5)[0]].size()==torch.Size([0]))

x = torch.IntTensor([1,2,3,4])
print(x[torch.randperm(x.size(0))])