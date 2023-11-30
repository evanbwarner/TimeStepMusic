import torch
import torch.nn as nn
import numpy as np



class ConvGenerator(nn.Module):
    def __init__(self, tot_chunks, latent_dim, num_channels):
        #tot_chunks is length of music we're generating; will be rounded down to nearest
        #power of 2
        #latent_dim is size of generator input; try 100
        #num_channels will be equal to size of autoencoder's latent dim
        
        super().__init__()
        self.tot_chunks_exp = int(np.log2(tot_chunks))
        self.tot_chunks = 2**self.tot_chunks_exp
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        
        #leakyRU activation function is supposed to help with sparse gradients for GAN
        #bias=False because layer or batch norms already have a bias term
        #layer norm since gradient penalty is not valid with batch norm
        self.first_layer = nn.Sequential(nn.Linear(latent_dim, num_channels*self.tot_chunks),
                                         nn.InstanceNorm1d(num_channels*self.tot_chunks),
                                         nn.LeakyReLU(negative_slope=0.2))
        
        middle_layers = []
        for i in range(self.tot_chunks_exp-1,1,-1):
            inputdim = num_channels*(2**i)
            outputdim = num_channels*(2**(i-1))
            middle_layers.append(nn.ConvTranspose1d(inputdim, outputdim, 4, stride=2,
                                                     padding=1, bias=False))
            middle_layers.append(nn.InstanceNorm1d(outputdim))
            middle_layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.middle_layers = nn.Sequential(*middle_layers)
        
        self.last_layer = nn.Sequential(nn.ConvTranspose1d(num_channels*2, num_channels, 4, 
                                                           stride=2, padding=1, bias=False),
                                        nn.Tanh())
        
    def forward(self, z):
        N = z.shape[0]
        #let N be batch size, so z is dimensions (N,latent_dim)
        h0 = self.first_layer(z) #(N,num_channels*tot_chunks)
        h0 = h0.view(N, self.num_channels*self.tot_chunks//2, 2) #(N,num_channels*tot_chunks/2,2)
        h1 = self.middle_layers(h0) #(N,num_channels*2,tot_chunks/2)
        h2 = self.last_layer(h1) #(N,num_channels,tot_chunks)
        return h2
                                                       

class ConvDiscriminator(nn.Module):
    def __init__(self, tot_chunks, num_channels):
        #tot_chunks is length of music we're generating; will be rounded down to nearest
        #power of 2
        #num_channels will be equal to size of autoencoder's latent dim
        
        super().__init__()
        self.tot_chunks_exp = int(np.log2(tot_chunks))
        self.tot_chunks = 2**self.tot_chunks_exp
        self.num_channels = num_channels
        
        layers = []
        for i in range(self.tot_chunks_exp - 1):
            inputdim = num_channels*(2**i)
            outputdim = num_channels*(2**(i+1))
            layers.append(nn.Conv1d(inputdim, outputdim, 4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm1d(outputdim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            
        self.layers = nn.Sequential(*layers)
        
        #no instance norm needed when third dimension is 1
        inputdim = num_channels*self.tot_chunks//2
        outputdim = num_channels*self.tot_chunks
        self.penultimate_layer = nn.Sequential(nn.Conv1d(inputdim, outputdim, 4,
                                                         stride=2, padding=1, bias=False),
                                               nn.LeakyReLU(negative_slope=0.2))
        self.last_layer = nn.Linear(outputdim,1)        
        
    def forward(self, z):
        N = z.shape[0]
        #let N be batch size, so z is dimensions (N,num_channels,tot_chunks)
        h0 = self.layers(z) #(N,num_channels*tot_chunks/2,2)
        h1 = self.penultimate_layer(h0) #(N,num_channels*tot_chunks,1)
        h1 = h1.view(N, self.num_channels*self.tot_chunks) #(N,num_channels*tot_chunks)
        output = self.last_layer(h1).view(-1)
        
        return output, h1 #return h1 for feature matching