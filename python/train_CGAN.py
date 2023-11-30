import CGAN
import utils
import tqdm
import torch
import torch.nn as nn
import torch.distributions
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.autograd
import numpy as np
from midi2audio import FluidSynth
import os

class CGANDataset(Dataset):
    def __init__(self, ae_model, events, annotation, metadata, tot_chunks, 
                 chunk_length, data_type, chunk_dim):
        #we will use both train and validation sets for GAN training
        super().__init__()
        
        self.data_type = data_type
        self.chunk_dim = chunk_dim
        self.tot_chunks = tot_chunks
        
        relevant_metadata = metadata[(metadata['split']=='train') | (metadata['split']=='validation')]
        start_chunks = np.array(relevant_metadata['start_chunk'])
        end_chunks = np.array(relevant_metadata['end_chunk'])
        self.num_chunks = np.sum(end_chunks-start_chunks)
        
        self.data = torch.zeros(self.num_chunks, self.chunk_dim)
        curr_chunk = 0
        ae_model.eval()
        
        #total number of data points, keeping in mind that we should omit the last tot_chunks-1
        #chunks from each track so we don't load truncated passages at the end of tracks
        self.len = self.num_chunks - len(relevant_metadata.index)*(tot_chunks-1)
        
        #will index starting chunks of data points for __getitem__
        self.indices = np.zeros(self.len, dtype='int32')
        curr_index = 0
        
        for track in tqdm.tqdm(relevant_metadata.index, desc='Normalizing and encoding data'):
            start_event = metadata['start_event'][track]
            end_event = metadata['end_event'][track]
            start_chunk = metadata['start_chunk'][track]
            end_chunk = metadata['end_chunk'][track]
            chunks = events[start_event:end_event,:4]
            ticks_per_second = metadata['ticks_per_second'][track]
            ns = torch.tensor([annotation[chunk_index][1] for chunk_index in
                               range(start_chunk, end_chunk)], dtype=torch.int32)
            
            norm_chunks, _ = utils.NormalizeChunk(chunks, ticks_per_second, chunk_length,   
                                                  self.data_type)
            encoded = ae_model.encoder(norm_chunks, ns).detach()
            curr_num_chunks = encoded.shape[0]
            self.data[curr_chunk:curr_chunk+curr_num_chunks,:] = encoded
            self.indices[curr_index:curr_index+curr_num_chunks-(tot_chunks-1)] = np.arange(curr_chunk, curr_chunk+curr_num_chunks-(tot_chunks-1))
            curr_index += curr_num_chunks-(tot_chunks-1)
            curr_chunk += curr_num_chunks
            
            
    def __len__(self):
        return self.len
        
        
    def __getitem__(self, idx):
        chunk_index = self.indices[idx]
        #flip dimensions so `channels' is first
        return torch.transpose(self.data[chunk_index:chunk_index+self.tot_chunks,:], 0, 1)
    
    
    
def grad_penalty(disc_model, device, real_batch, fake_batch, batch_size):
    #for WGAN-GP
    
    #compute interpolation
    alpha = torch.rand((batch_size, 1, 1), requires_grad=True, device=device)
    
    interpolated = alpha*real_batch + (1-alpha)*fake_batch
    
    #validity of interpolated example
    interpolated_val, _ = disc_model(interpolated)
    
    #gradients of validities
    gradients = torch.autograd.grad(outputs=interpolated_val, inputs=interpolated,
                                    grad_outputs=torch.ones_like(interpolated_val, device=device),
                                    create_graph=True, retain_graph=True)[0]
    
    gradients = gradients.view(batch_size,-1)
    
    #regularize square root with added epsilon for stability
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

    return ((gradients_norm - 1)**2).mean()
    
    
def disc_train(disc_model, gen_model, disc_optim, distribution, device, 
               real_batch, batch_size, lambda_gp):
    disc_model.zero_grad()
            
    #first do an all-real batch
    real_batch = real_batch.to(device)
    d_real, features_real = disc_model(real_batch)
    
    #now do an all-fake batch
    fake_sample = distribution.sample([batch_size]).to(device)
    fake_batch = gen_model(fake_sample)
    d_fake, _ = disc_model(fake_batch) #d_fake is a vector of probabilities
    
    partial_disc_error = d_fake.mean() - d_real.mean()
    
    #error from gradient penalty
    grad_error = lambda_gp * grad_penalty(disc_model, device, real_batch, 
                                          fake_batch.detach(), batch_size)
                    
    disc_error = partial_disc_error + grad_error
    
    disc_error.backward()
    disc_optim.step() #update disc_model
    return disc_error.item(), features_real.detach(), partial_disc_error.item(), grad_error.item()


def gen_train(disc_model, gen_model, gen_optim, fm_criterion, distribution, 
              device, batch_size, lambda_fm, features_real):
    gen_model.zero_grad()
    
    fake_sample = distribution.sample([batch_size]).to(device)
    fake_batch = gen_model(fake_sample)
    d_fake, features_fake = disc_model(fake_batch)
    
    #usual generator error
    partial_gen_error = -d_fake.mean()
    
    #feature mapping error
    fm_error = lambda_fm * fm_criterion(features_real, features_fake)
    
    gen_error = partial_gen_error + fm_error
    
    gen_error.backward()
    gen_optim.step() #update gen_model
    return gen_error.item(), partial_gen_error.item(), fm_error.item()
    
    
def train(disc_model, gen_model, ae_model, train_dataset, tot_chunks, latent_dim, epochs, 
          lr, batch_size, num_channels, chunk_length, lambda_gp, lambda_fm, model_path, 
          disc_model_file, gen_model_file, data_type, examples_path):

    data_loader = DataLoader(train_dataset, batch_size, shuffle=True, 
                             drop_last=True, num_workers=4)
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
    print('Using device:', device)
    disc_model = disc_model.to(device)
    gen_model = gen_model.to(device)
    disc_model.train()
    gen_model.train()
    ae_model.eval() #not training the autoencoder!
    disc_optim = torch.optim.Adam(disc_model.parameters(), lr)
    gen_optim = torch.optim.Adam(gen_model.parameters(), lr)
    
    fm_criterion = nn.MSELoss()
    
    #sample from a ball rather than a cube
    #we'll use fixed_latent_sample to track the progress of the generator
    distribution = torch.distributions.MultivariateNormal(torch.zeros(latent_dim), 
                                                          torch.eye(latent_dim))
    fixed_latent_sample = distribution.sample([batch_size]).to(device)

    gen_losses = []
    disc_losses = []
    
    torch.set_printoptions(sci_mode=False, precision=3)
    
    #it might be reasonable to train the discriminator by itself for a while before
    #training the generator at all but skip for now
    
    for epoch in range(1, epochs+1):
        batch_counter = 1
        
        for real_batch in tqdm.tqdm(data_loader, desc=f'Training epoch {epoch}'):
            
            disc_error, features_real, partial_disc_error, grad_error = disc_train(disc_model, 
                                                                                   gen_model, 
                                                                                   disc_optim,
                                                                                   distribution, 
                                                                                   device, 
                                                                                   real_batch, 
                                                                                   batch_size, 
                                                                                   lambda_gp)
            
            disc_losses.append((disc_error, partial_disc_error, grad_error))
            
            gen_error=0
            partial_gen_error=0
            gen_error_fm=0
            
            if batch_counter % 5 == 0: #original paper recommends one gen training every 5 disc
                gen_error, partial_gen_error, gen_error_fm = gen_train(disc_model, gen_model, 
                                                                       gen_optim, fm_criterion, 
                                                                       distribution, device,
                                                                       batch_size, lambda_fm, 
                                                                       features_real)
                gen_losses.append((gen_error, partial_gen_error, gen_error_fm))
            
            if batch_counter % 10000 == 0:
                print(f'Epoch {epoch} batch {batch_counter}')
                print(f'Tot disc loss: {disc_error:.5f} partial_disc_error: {partial_disc_error} grad_error: {grad_error:.5f}')
                print(f'Tot gen loss: {gen_error:.5f} partial_gen_error: {partial_gen_error} gen_error_fm: {gen_error_fm}')
                
                #print decoded versions of real sample, fake sample, and interpolated sample
                gen_model.eval()
                with torch.no_grad():
                    #first take real batch
                    reshaped_real_batch = torch.transpose(real_batch,1,2).reshape(-1,num_channels)
                    real_norm_chunks, real_ns = ae_model.decoder(reshaped_real_batch)
                    
                    #take just first sample of batch
                    real_ns = real_ns[0:tot_chunks]
                    num_real_events = sum(real_ns)
                    real_norm_chunks = real_norm_chunks[0:num_real_events]
                    real_events = utils.NormalizedChunksToEvents(real_norm_chunks, real_ns,
                                                                 960, data_type, chunk_length)
                    
                    #now take a fake batch
                    fake_sample = distribution.sample([batch_size]).to(device)
                    fake_batch = gen_model(fake_sample).detach().cpu()
                    reshaped_fake_batch = torch.transpose(fake_batch,1,2).reshape(-1,num_channels)
                    fake_norm_chunks, fake_ns = ae_model.decoder(reshaped_fake_batch)
                    
                    #take just first sample of batch
                    fake_ns = fake_ns[0:tot_chunks]
                    num_fake_events = sum(fake_ns)
                    fake_norm_chunks = fake_norm_chunks[0:num_fake_events]
                    fake_events = utils.NormalizedChunksToEvents(fake_norm_chunks, fake_ns,
                                                                 960, data_type, chunk_length)
                    
                    #print('A real sample:', real_events)
                    real_prob, _ = disc_model(real_batch)
                    real_prob = real_prob[0].item()
                    print('Real sample length:', real_events.shape[0])
                    print('Disc validity of real sample:', real_prob)
                    
                    #print('A fake sample:', fake_events)
                    fake_prob, _ = disc_model(fake_batch)
                    fake_prob = fake_prob[0].item()
                    print('Fake sample length:', fake_events.shape[0])
                    print('Disc validity of fake sample:', fake_prob)
                    
                    alpha = torch.rand((batch_size, 1, 1), device=device)
                    interpolated_batch = alpha*real_batch + (1-alpha)*fake_batch
                    reshaped_interpolated_batch = torch.transpose(interpolated_batch,1,2).reshape(-1,num_channels)
                    interpolated_norm_chunks, interpolated_ns = ae_model.decoder(reshaped_interpolated_batch)
                    
                    interpolated_ns = interpolated_ns[0:tot_chunks]
                    num_interpolated_events = sum(interpolated_ns)
                    interpolated_norm_chunks = interpolated_norm_chunks[0:num_interpolated_events]
                    interpolated_events = utils.NormalizedChunksToEvents(interpolated_norm_chunks,
                                                                         interpolated_ns, 960,
                                                                         data_type, chunk_length)
                    
                    #print('An interpolated sample:', interpolated_events)
                    interpolated_prob, _ = disc_model(interpolated_batch)
                    interpolated_prob = interpolated_prob[0].item()
                    print('Interpolated sample length:', interpolated_events.shape[0])
                    print('Disc validity of interpolated sample:', interpolated_prob)            
                
            if batch_counter % 50000 == 0:
        
                #retain generator output on fixed latent samples
                gen_model.eval() #informs layers like batchnorm to work appropriately for eval
                with torch.no_grad(): #does not compute gradients
                    #each output is a batch of samples
                    output = gen_model(fixed_latent_sample).detach().cpu() #(batch_size,num_channels,64)
                output = torch.transpose(output,1,2).reshape(-1,num_channels) #(batch_size*64,num_channels)
                gen_model.train()

                #decode and save all our samples
                norm_chunks, ns = ae_model.decoder(output)
                norm_chunks = norm_chunks.detach().cpu()
                ns = ns.detach().cpu()
                curr_event_index = 0
                for i in range(batch_size):
                    curr_ns = ns[tot_chunks*i:tot_chunks*(i+1)]
                    curr_num_events = sum(curr_ns)
                    curr_sample = norm_chunks[curr_event_index:curr_event_index+curr_num_events]
                    curr_event_index += curr_num_events

                    #choice of ticks per second = 960 is arbitrary except for rounding
                    events = utils.NormalizedChunksToEvents(curr_sample, curr_ns, 
                                                            960, data_type, chunk_length)
                    midi_file = utils.EventsToMidi(events, 960)
                    file_name = 'epoch'+str(epoch)+'batch'+str(batch_counter)+'sample'+str(i)+'.midi'
                    os.chdir(examples_path)
                    midi_file.save(file_name)

                    #save models
                    os.chdir(model_path)
                    torch.save(disc_model.state_dict(), disc_model_file)
                    torch.save(gen_model.state_dict(), gen_model_file)
            batch_counter += 1
            
    return disc_losses, gen_losses



