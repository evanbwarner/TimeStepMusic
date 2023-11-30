#taken from Permutation-Invariant Set Autoencoders with Fixed-Size
#Embeddings for Multi-Agent Learning
#associated code at github.com/Acciorocketships/SetAutoEncoder
#(and heavily modified)

import torch.nn as nn
import torch
from torch import Tensor
import numpy as np
import math
from torch.nn import Module
import torch.nn.functional as F
import utils
    
    
    
class AutoEncoder(Module):
    def __init__(self, data_type, *args, **kwargs):
        super().__init__()
        self.data_type = data_type
        if data_type == 'OHE':
            dim = 94
        elif data_type == 'ponOHE':
            dim = 27
        elif data_type == 'pnOHE':
            dim = 20
        elif data_type == 'pon':
            dim = 6
        elif data_type == 'none':
            dim = 4
        
        self.encoder = Encoder(dim = dim, *args, **kwargs)
        self.decoder = Decoder(dim = dim, *args, **kwargs)
        
    def forward(self, x, n):
        z = self.encoder(x, n)
        xr, nr = self.decoder(z)
        return xr, nr

    def get_vars(self):
        self.vars = {"n_pred_approx": self.decoder.get_n_pred_approx(),
                     "n_pred": self.decoder.get_n_pred(),
                     "n": self.encoder.get_n(),
                     "x": self.encoder.get_x(),
                     "xr": self.decoder.get_x_pred()}
        return self.vars

    def loss(self, IDs=None):
        #Loss depends on datatype. See above for expected dimensions for IDs, which is included for
        #faster cross-entropy computation
        #Losses are of course only comparable within the same datatype
        vars = self.get_vars()
        losses = {}
        losses['size loss'] = nn.functional.mse_loss(vars['n_pred_approx'], vars['n'].unsqueeze(-1).detach().float())
        pred_idx, tgt_idx = get_loss_idxs(vars['n_pred'], vars['n'])
        x = vars['x']
        xr = vars['xr']
        
        if self.data_type == 'none':
            losses['note/pedal loss'] = F.mse_loss(xr[pred_idx,0], x[tgt_idx,0])
            losses['cont loss'] = F.mse_loss(xr[pred_idx,1:4], x[tgt_idx,1:4])
            losses['loss'] = losses['note/pedal loss'] + losses['cont loss'] + 0.001 * losses['size loss']
            
        elif self.data_type == 'pon':
            losses['pedal loss'] = F.mse_loss(xr[pred_idx,0], x[tgt_idx,0])
            losses['octave loss'] = F.mse_loss(xr[pred_idx,1], x[tgt_idx,1])
            losses['note loss'] = F.mse_loss(xr[pred_idx,2], x[tgt_idx,2])
            losses['cont loss'] = F.mse_loss(xr[pred_idx,3:6], x[tgt_idx,3:6])
            losses['loss'] = losses['pedal loss'] + losses['octave loss'] + losses['note loss'] + losses['cont loss'] + 0.001 * losses['size loss']
            
        elif self.data_type == 'pnOHE':
            losses['pedal loss'] = F.cross_entropy(xr[pred_idx,:4], IDs[tgt_idx,0].long())
            losses['octave loss'] = F.mse_loss(xr[pred_idx,4], x[tgt_idx,1])
            losses['note loss'] = F.cross_entropy(xr[pred_idx,5:17], IDs[tgt_idx,1].long())
            losses['cont loss'] = F.mse_loss(xr[pred_idx,17:20], x[tgt_idx,17:20])
            losses['loss'] = losses['pedal loss'] + losses['octave loss'] + losses['note loss'] + losses['cont loss'] + 0.001 * losses['size loss']
        
        elif self.data_type == 'ponOHE':
            losses['pedal loss'] = F.cross_entropy(xr[pred_idx,:4], IDs[tgt_idx,0].long())
            losses['octave loss'] = F.cross_entropy(xr[pred_idx,4:12], IDs[tgt_idx,1].long())
            losses['note loss'] = F.cross_entropy(xr[pred_idx,12:24], IDs[tgt_idx,2].long())
            losses['cont loss'] = F.mse_loss(xr[pred_idx,24:27], x[tgt_idx,24:27])
            losses['loss'] = losses['pedal loss'] + losses['octave loss'] + losses['note loss'] + losses['cont loss'] + 0.001 * losses['size loss']
        
        elif self.data_type == 'OHE':
            losses['note/pedal loss'] = F.cross_entropy(xr[pred_idx,:91], IDs[tgt_idx,0].long())
            losses['cont loss'] = F.mse_loss(xr[pred_idx,91:94], x[tgt_idx,91:94])
            losses['loss'] = losses['note/pedal loss'] + losses['cont loss'] + 0.001 * losses['size loss']
        
        return losses
    
    

def get_loss_idxs(set1_lens, set2_lens):
    """Given two data of different sizes, return the indices for the
    first and second datas denoting the elements to be used in the loss
    function. setn_lens represents the number of elements in each batch.

    E.g. 
    >>> pred_idx, target_idx = get_loss_idxs(pred, target, ....)
    >>> loss = functional.mse_loss(pred[pred_idx], target[target_idx])
    """
    assert set1_lens.shape == set2_lens.shape
    set_lens = torch.min(set1_lens, set2_lens)

    ptr1_start = set1_lens.cumsum(dim=0).roll(1)
    ptr1_start[0] = 0
    ptr1_end = ptr1_start + set_lens

    ptr2_start = set2_lens.cumsum(dim=0).roll(1)
    ptr2_start[0] = 0
    ptr2_end = ptr2_start + set_lens

    ptr1 = torch.cat(
        [
            torch.arange(ptr1_start[i], ptr1_end[i], device=set1_lens.device) 
            for i in range(ptr1_start.numel())
        ]
    )
    ptr2 = torch.cat(
        [
            torch.arange(ptr2_start[i], ptr2_end[i], device=set2_lens.device) 
            for i in range(ptr2_start.numel())
        ]
    )
    return ptr1, ptr2


class Encoder(Module):
    def __init__(self, dim, hidden_dim, max_n, **kwargs):
        super().__init__()
		# Params
        self.input_dim = dim #dim of an event, depending on datatype
        self.hidden_dim = hidden_dim #dim of latent variables
        self.max_n = max_n
        self.pos_mode = kwargs.get('pos_mode', 'onehot')
        self.depth = kwargs.get('depth', 2)
        # Modules
        self.pos_gen = PositionalEncoding(dim=self.max_n, mode=self.pos_mode)
        self.key_net = BuildMLP(input_dim=self.max_n, output_dim=self.hidden_dim, nlayers=self.depth, 
                                layernorm=True, nonlinearity=nn.Mish)
        self.val_net = BuildMLP(input_dim=self.input_dim, output_dim=self.hidden_dim, nlayers=self.depth,
                                layernorm=True, nonlinearity=nn.Mish)
        self.rank = torch.nn.Linear(self.input_dim, 1)
        self.cardinality = torch.nn.Linear(1, self.hidden_dim)

    def sort(self, x, batch):
        mag = self.rank(x)
        max_mag = torch.max(mag) + 0.0001
        batch_mag = batch * max_mag
        new_mag = mag.squeeze() + batch_mag
        _, idx_sorted = torch.sort(new_mag)
        x_sorted = x[idx_sorted]
        xs_idx = idx_sorted
        xs = x_sorted
        return xs, xs_idx
    
    def forward(self, x, n):
        _, input_dim = x.shape
        self.n = n
        batch = torch.repeat_interleave(torch.arange(n.shape[0], device=x.device), n, dim=0)
        
        # Sort
        xs, xs_idx = self.sort(x, batch)
        self.xs = xs
        self.xs_idx = xs_idx

        keys = torch.cat([torch.arange(ni, device=x.device) for ni in n], dim=0).int() #tot n of events
        pos = self.pos_gen(keys) # tot n of events * max n of events

		# Encoder
        y = self.val_net(xs) * self.key_net(pos)  # tot n of events * hidden_dim
        
        z_elements = torch.zeros((n.shape[0], self.hidden_dim), device=x.device)
        z_elements = z_elements.scatter_reduce(0, batch.unsqueeze(1).expand((batch.shape[0], self.hidden_dim)), y, reduce='sum')
        
        #z_elements = scatter(src=y, index=batch, dim=-2)  # batch_size * dim
        n_enc = self.cardinality(n.unsqueeze(-1).float())
        z = z_elements + n_enc
        self.z = z
        return z

    def get_x_perm(self):
        'Returns: the permutation applied to the inputs (shape: ninputs)'
        return self.xs_idx

    def get_z(self):
        'Returns: the latent state (shape: batch x hidden_dim)'
        return self.z
    
    def get_batch(self):
        'Returns: the batch idxs of the inputs (shape: ninputs)'
        return self.batch

    def get_x(self):
        'Returns: the sorted inputs, x[x_perm] (shape: ninputs x dim)'
        return self.xs

    def get_n(self):
        'Returns: the number of elements per batch (shape: batch)'
        return self.n

    def get_max_n(self):
        return self.max_n
    


class Decoder(Module):
    def __init__(self, dim, hidden_dim, max_n, **kwargs):
        super().__init__()
		# Params
        self.output_dim = dim
        self.hidden_dim = hidden_dim
        self.max_n = max_n
        self.pos_mode = kwargs.get("pos_mode", "onehot")
        self.depth = kwargs.get('depth', 2)
        # Modules
        self.pos_gen = PositionalEncoding(dim=self.max_n, mode=self.pos_mode)
        self.key_net = BuildMLP(input_dim=self.max_n, output_dim=self.hidden_dim, nlayers=self.depth,
                                layernorm=True, nonlinearity=nn.Mish)
        self.decoder = BuildMLP(input_dim=self.hidden_dim, output_dim=self.output_dim, nlayers=self.depth,
                                layernorm=False, nonlinearity=nn.Mish)
        self.size_pred = BuildMLP(input_dim=self.hidden_dim, output_dim=1, nlayers=self.depth, layernorm=True, 
                                  nonlinearity=nn.Mish)

    def forward(self, z):
        # z: batch_size x hidden_dim
        n_pred_approx = self.size_pred(z)
        n = torch.round(n_pred_approx, decimals=0).squeeze(-1).int()
        n = torch.minimum(n, torch.tensor(self.max_n-1))
        n = torch.maximum(n, torch.tensor(0))
        self.n_pred_approx = n_pred_approx
        self.n_pred = n

        k = torch.cat([torch.arange(n[i], device=z.device) for i in range(n.shape[0])], dim=0)
        pos = self.pos_gen(k) # total_nodes x max_n

        keys = self.key_net(pos)

        vals_rep = torch.repeat_interleave(z, n, dim=0)
        zp = vals_rep * keys

        x = self.decoder(zp)

        self.x = x
        
        #next two lines aren't really necessary; should do outside training
        batch = torch.repeat_interleave(torch.arange(n.shape[0], device=z.device), n, dim=0)
        self.batch = batch
        
        return x, n

    def get_batch_pred(self):
        'Returns: the batch idxs of the outputs x (shape: noutputs)'
        return self.batch

    def get_x_pred(self):
        'Returns: the outputs x (shape: noutputs x d)'
        return self.x

    def get_n_pred_approx(self):
        'Returns: the predicted n (approximation)'
        return self.n_pred_approx

    def get_n_pred(self):
        'Returns: the predicted n'
        return self.n_pred


class PositionalEncoding(Module):
    def __init__(self, dim: int, mode: str = 'onehot'):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.I = torch.eye(self.dim).byte()

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == 'onehot':
            return self.onehot(x.int()).float()
        elif self.mode == 'binary':
            return self.binary(x.int()).float()
        elif self.mode == 'sinusoid':
            return self.sinusoid(x)

    def onehot(self, x: Tensor) -> Tensor:
        out_shape = list(x.shape) + [self.dim]
        self.I = (self.I).to(x.device)
        return torch.index_select(input=self.I, dim=0, index=x.reshape(-1)).reshape(*out_shape)

    def binary(self, x: Tensor) -> Tensor:
        x = x + 1
        mask = 2 ** torch.arange(self.dim).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

    def binary_to_int(self, x: Tensor) -> Tensor:
        multiplier = 2 ** torch.arange(x.shape[-1]).float().view(-1,1)
        y = x.float() @ multiplier
        return (y-1).squeeze(1).int()

    def binary_logits_to_binary(self, x: Tensor) -> Tensor:
        xs = torch.softmax(x, dim=1)
        max_mag = torch.max(xs, dim=1)[0]
        xs_reg = xs / max_mag[:,None]
        binary = (xs_reg > 0.5).int()
        return binary

    def onehot_logits_to_int(self, x: Tensor) -> Tensor:
        return torch.argmax(x, dim=-1)

    def sinusoid(self, x: Tensor):
        max_n = torch.max(x)+1
        pe = torch.zeros(max_n, self.dim, device=x.device)  # like 10x4
        position = torch.arange(0, max_n, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=x.device).float() * (-math.log(10000.0) / self.dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe




def BuildMLP(input_dim, output_dim, nlayers=1, batchnorm=False,
            layernorm=True, nonlinearity=nn.Mish):
    #Builds a MLP with given features. nlayers is number of hidden layers
    mlp_layers = GenerateLayerSizes(input_dim=input_dim, output_dim=output_dim,
                                    nlayers=nlayers)
    mlp = MLP(layer_sizes=mlp_layers, batchnorm=batchnorm, layernorm=layernorm, 
              nonlinearity=nonlinearity)
    return mlp
    
    
def GenerateLayerSizes(input_dim, output_dim, nlayers=1):
    #gives some reasonable layer sizes for an MLP. nlayers is number of hidden layers
	midlayersize = (input_dim + output_dim) // 2
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


class BatchNorm(Module):
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
    def __init__(self, layer_sizes, batchnorm=False, layernorm=False, nonlinearity=nn.Mish):
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
    
    


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int=-1, dim_size=None):
    index = broadcast(index, src, dim)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index.long(), src)

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0,dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src