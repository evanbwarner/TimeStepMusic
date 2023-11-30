#taken from Permutation-Invariant Set Autoencoders with Fixed-Size
#Embeddings for Multi-Agent Learning
#associated code at github.com/Acciorocketships/SetAutoEncoder

import torch.nn as nn
import torch
import numpy as np

def BuildMLP(input_dim, output_dim, nlayers=1, batchnorm=False,
            layernorm=True, nonlinearity=nn.GELU):
    #Builds a MLP with given features. nlayers is number of hidden layers
    mlp_layers = GenerateLayerSizes(input_dim=input_dim, output_dim=output_dim,
                               nlayers=nlayers, midmult=midmult)
    mlp = MLP(layer_sizes=mlp_layers, batchnorm=batchnorm, layernorm=layernorm, 
              nonlinearity=nonlinearity)
    
    
def GenerateLayerSizes(input_dim, output_dim, nlayers=1):
    #gives some reasonable layer sizes for an MLP. nlayers is number of hidden layers
	midlayersize = midmult * (input_dim + output_dim) // 2
	midlayersize = max(midlayersize, 1)
	nlayers += 2
	layers1 = np.around(
		np.logspace(np.log10(input_dim), np.log10(midlayersize), num=(nlayers) // 2)
	).astype(int)
	layers2 = np.around(
		np.logspace(
			np.log10(midlayersize), np.log10(output_dim), num=(nlayers + 1) // 2
		)
	).astype(int)[1:]
	return list(np.concatenate([layers1, layers2]))


class BatchNorm(nn.Module):
    #needed since it looks like nn.BatchNorm1d expects weird dimensions
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.bn = nn.BatchNorm1d(*args, **kwargs)

	def forward(self, x):
		shape = x.shape
		x_r = x.reshape(np.prod(shape[:-1]), shape[-1])
		y_r = self.bn(x_r)
		y = y_r.reshape(shape)
		return y
    
    
class MLP(nn.Module):
    def __init__(self, layer_sizes, batchnorm=False, layernorm=False, nonlinearity=nn.ReLU):
        super().__init__()
        self.batchnorm = batchnorm
        layers = []
        for i in range(len(layer_sizes) - 1):
			layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
			if i != len(layer_sizes) - 2:
				if batchnorm:
					layers.append(BatchNorm(layer_sizes[i + 1]))
				if layernorm:
					layers.append(nn.LayerNorm(layer_sizes[i + 1]))
				layers.append(nonlinearity())
		self.net = nn.Sequential(*layers)

	def forward(self, X):
		return self.net(X)