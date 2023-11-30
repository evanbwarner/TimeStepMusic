#defines AutoEncoderDataset and has script to train autoencoder
#calls autoencoder.py for the autoencoder itself and utils.py for various utilities

import autoencoder
import utils
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import copy



class AutoEncoderDataset(Dataset):
    def __init__(self, events, annotation, metadata, chunk_length, data_type, aug_type, split):
        #inputs events, annotation, metadata from run.ReadMIDI() routine
        #data_type is data encoding type; see run file for descriptions
        #aug_type is data augmentation type; not currently implemented
        #split is 'train' or 'valid' or 'test'
        
        super().__init__()
        
        print('Loading', split, 'data...')
        
        self.chunk_length = chunk_length
        self.data_type = data_type
        self.dim = utils.GetDim(data_type)
        self.split = split
        self.metadata = metadata #need to remember this to access ticks per second info
        
        #which events are relevant to the split?
        starts = np.array(metadata[metadata['split']==split]['start_event'])
        ends = np.array(metadata[metadata['split']==split]['end_event'])
        self.event_indices = np.concatenate([np.arange(start, end) for (start, end)
                                                      in zip(starts, ends)])
        
        #copy over only relevant data
        #this is important! we need all the data to be contiguous for performance reasons
        self.data = copy.deepcopy(events[self.event_indices,:])
        
        #which chunks are relevant to the split?
        starts = np.array(metadata[metadata['split']==split]['start_chunk'])
        ends = np.array(metadata[metadata['split']==split]['end_chunk'])
        self.chunk_indices = np.concatenate([np.arange(start, end) for (start, end)
                                                      in zip(starts, ends)])
        self.num_chunks = self.chunk_indices.shape[0]
        
        #internal annotation should reference internal events indices
        inverse_events_indices = {external:internal for internal, external in
                                  enumerate(self.event_indices)}
        
        
        self.annotation = {idx:(annotation[idx][0],
                                annotation[idx][1],
                                inverse_events_indices[annotation[idx][2]])
                                for idx in self.chunk_indices}
        
        #data augmentation: do later

        
    def __len__(self):
        return self.num_chunks
    
    def __getitem__(self, idx):
        #Loads a single chunk and normalizes according to datatype
        #Returns the chunk, the length of the chunk n, and a list of note/pedal IDs
        #Note that we normalize each time we get an item, so we are recomputing each epoch!
        #This is to keep memory requirements small, esp. for one hot encoding
        chunk_index = self.chunk_indices[idx]
        (track, n, start_event_index) = self.annotation[chunk_index]
        
        ticks_per_second = self.metadata['ticks_per_second'][track]
        
        #n*4 where n is number of events in chunk
        chunk = self.data[start_event_index:start_event_index+n,:4] 
        
        #norm_chunk is n*self.dim
        norm_chunk, IDs = utils.NormalizeChunk(chunk, ticks_per_second, self.chunk_length,
                                               self.data_type)
        
        return norm_chunk, n, IDs

    #GET RID OF THIS
    def GetTicksPerSecond(self, idx):
        chunk_index = self.chunk_indices[idx]
        (track, _, _) = self.annotation[idx]
        return self.metadata['ticks_per_second'][track]
    
#We need a custom collate_fn to group batches together - we don't want to make a new dimension, since
#then we'd pad out by unnecessary zeros, but only concatenate the given samples along the already-existing 
#dimension.
#Returns the data in an N*dim array (N = total number of events in batch), a vector n of chunk sizes of
#length batch size, and a vector of note/pedal IDs of length N
def collate(data_list):
    norm_chunks = torch.cat([d[0] for d in data_list])
    n = torch.Tensor([d[1] for d in data_list]).type(torch.int)
    IDs = torch.cat([d[2] for d in data_list])
    return norm_chunks, n, IDs  


def train(model, train_dataset, valid_dataset, data_type, aug_type, max_chunk_size, chunk_length,
          pedal_bins, epochs, lr, batch_size, model_path, model_file):

    train_DL = DataLoader(train_dataset, batch_size, shuffle=True, 
                          drop_last=True, num_workers=4, collate_fn=collate)
    valid_DL = DataLoader(valid_dataset, batch_size, shuffle=False, 
                          drop_last=True, num_workers=4, collate_fn=collate)
    optim = torch.optim.Adam(model.parameters(), lr)
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
    print('Using device:', device)
    model = model.to(device)

    train_losses = []
    valid_losses = []
        
    for epoch in range(epochs):
        model.train()
    
        epoch_loss = 0
        batch_counter = 0
        for norm_chunks, n, IDs in tqdm.tqdm(train_DL, desc=f'Training epoch {epoch+1}'):
            norm_chunks = norm_chunks.to(device)
            n = n.to(device)
            IDs = IDs.to(device)
                    
            _, _ = model(norm_chunks, n)
        
            loss_data = model.loss(IDs)
            loss = loss_data['loss']
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            epoch_loss += loss.detach().cpu()
            batch_counter += 1
        train_losses.append(epoch_loss/batch_counter) #avg losses per batch
        model.eval() #in evaluation mode
        with torch.set_grad_enabled(False): #validation
            epoch_loss = 0
            batch_counter = 0
            for norm_chunks, n, IDs in tqdm.tqdm(valid_DL, desc=f'Validation epoch {epoch+1}'): 
                norm_chunks = norm_chunks.to(device)
                n = n.to(device)
                IDs = IDs.to(device)
                _, _ = model(norm_chunks, n)
                loss_data = model.loss(IDs)
                loss = loss_data['loss']
                epoch_loss += loss.detach().cpu()
                batch_counter += 1
            valid_losses.append(epoch_loss/batch_counter) #avg loss per batch
        print(f'Epoch {epoch+1} Total training loss: {train_losses[epoch]:.8f}')
        print(f'Epoch {epoch+1} Total validation loss: {valid_losses[epoch]:.8f}')
                
        if epoch % 5 == 0: #save model to disk every 5 epochs
            os.chdir(model_path)
            torch.save(model.state_dict(), model_file)
    return train_losses, valid_losses