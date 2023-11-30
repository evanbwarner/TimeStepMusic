import os
import train_autoencoder
import autoencoder
import train_CGAN
import CGAN
import torch
import utils
from midi2audio import FluidSynth

#folder names
HOME_PATH = '/Users/evanb/Documents/Python/TimeStepMusic/' #default for me
DATA_FOLDER = 'maestro-v3.0.0'
PYTHON_FOLDER = 'python'
MODELS_FOLDER = 'models'
EXAMPLES_FOLDER = 'examples'
AE_MODEL_FILE_NAME = 'small_ae_model_pon.pt'
CGAN_DISC_MODEL_FILE_NAME = 'cgan_disc_model.pt'
CGAN_GEN_MODEL_FILE_NAME = 'cgan_gen_model.pt'

#hyperparameters relating to data preprocessing
PEDAL_BINS = 4 #how coarsely to bin pedal intensities, 0 for no binning
MAX_CHUNK_SIZE = 15 #how many events per chunk will the autoencoder allow
CHUNK_LENGTH = 0.25 #how long are chunks, in seconds
DATA_TYPE = 'pon' #method of encoding events as vectors:
#datatype       dimension   IDs dim     description
#'OHE'          N*94        N*1         entire note/pedal ID one-hot encoded (91 + 3 dims)
#'ponOHE'       N*27        N*3         pedal, octave, note one-hot encoded separately (4 + 8 + 12 + 3 dims)
#'pnOHE'        N*20        N*2         pedal and note one-hot encoded separately (4 + 1 + 12 + 3 dims)
#'pon'          N*6         None        pedal, octave, note encoded separately (1 + 1 + 1 + 3 dims)
#'none'         N*4         None        no encoding (1 + 3 dims)
AUG_TYPE = 'none' #type of data augmentation:
#type           description
#'none'         no data augmentation
#'pitch'        add or subtract up to 5 semitones. pitches that are too low or too high are omitted
#'tempo'        speed up or slow down by up to 10%
#'pitchtempo'   both

#hyperparameters relating to autoencoder training
AE_LEARNING_RATE = 0.0001
AE_LATENT_DIM = 20
AE_BATCH_SIZE = 100 #number of *chunks* per batch
AE_EPOCHS = 75

#hyperparameters relating to CGAN training
CGAN_TOT_CHUNKS = 64 #a power of 2
CGAN_LEARNING_RATE = 0.0001
CGAN_LAMBDA_GP = 0.1
CGAN_LAMBDA_FM = 0.1
CGAN_LATENT_DIM = 50
CGAN_BATCH_SIZE = 10
CGAN_EPOCHS = 50




#visualization helper function? calls visualize.py routines with commentary



def ReadMIDI():
    #separate this routine because it takes a while so you only want to do it once
    #returns huge events array, chunk annotation dictionary, and metadata dataframe
    data_path = os.path.join(HOME_PATH, DATA_FOLDER)
    raw_metadata = utils.GetRawMetadata(data_path)
    
    return utils.GetAllEvents(raw_metadata, data_path, PEDAL_BINS,
                              CHUNK_LENGTH, MAX_CHUNK_SIZE)


def TrainAE(events, annotation, metadata):
    #prepares datasets and trains the autoencoder
    model_path = os.path.join(HOME_PATH, MODELS_FOLDER)
    
    
    ae_train_dataset = train_autoencoder.AutoEncoderDataset(events, annotation, metadata, 
                                                            CHUNK_LENGTH, DATA_TYPE, 
                                                            AUG_TYPE, 'train')
    ae_valid_dataset = train_autoencoder.AutoEncoderDataset(events, annotation, metadata, 
                                                            CHUNK_LENGTH, DATA_TYPE, 
                                                            AUG_TYPE, 'validation')
    
    ae_model = autoencoder.AutoEncoder(data_type=DATA_TYPE, hidden_dim=AE_LATENT_DIM, 
                                       max_n=MAX_CHUNK_SIZE)
    train_autoencoder.train(ae_model, ae_train_dataset, ae_valid_dataset, data_type=DATA_TYPE,
                            aug_type=AUG_TYPE, max_chunk_size=MAX_CHUNK_SIZE,
                            chunk_length=CHUNK_LENGTH, pedal_bins=PEDAL_BINS, epochs=AE_EPOCHS,
                            lr=AE_LEARNING_RATE, batch_size=AE_BATCH_SIZE, model_path=model_path,
                            model_file=AE_MODEL_FILE_NAME)
    
    
def ContinueTrainAE(events, annotation, metadata):
    #loads currently saved model and keeps training
    model_path = os.path.join(HOME_PATH, MODELS_FOLDER)
    
    ae_model = autoencoder.AutoEncoder(data_type=DATA_TYPE, hidden_dim=AE_LATENT_DIM,
                                       max_n=MAX_CHUNK_SIZE)
    ae_model_file = os.path.join(HOME_PATH, MODELS_FOLDER, AE_MODEL_FILE_NAME)
    ae_model.load_state_dict(torch.load(ae_model_file))
    
    ae_train_dataset = train_autoencoder.AutoEncoderDataset(events, annotation, metadata, 
                                                            CHUNK_LENGTH, DATA_TYPE, 
                                                            AUG_TYPE, 'train')
    ae_valid_dataset = train_autoencoder.AutoEncoderDataset(events, annotation, metadata, 
                                                            CHUNK_LENGTH, DATA_TYPE, 
                                                            AUG_TYPE, 'validation')
    
    train_autoencoder.train(ae_model, ae_train_dataset, ae_valid_dataset, data_type=DATA_TYPE,
                            aug_type=AUG_TYPE, max_chunk_size=MAX_CHUNK_SIZE,
                            chunk_length=CHUNK_LENGTH, pedal_bins=PEDAL_BINS, epochs=AE_EPOCHS,
                            lr=AE_LEARNING_RATE, batch_size=AE_BATCH_SIZE, model_path=model_path,
                            model_file=AE_MODEL_FILE_NAME)

def TrainCGAN(events, annotation, metadata):
    #assumes autoencoder has been trained and model saved as AE_MODEL_FILE_NAME
    #prepares dataset and trains CGAN
    ae_model = autoencoder.AutoEncoder(data_type=DATA_TYPE, hidden_dim=AE_LATENT_DIM,
                                       max_n=MAX_CHUNK_SIZE)
    ae_model_file = os.path.join(HOME_PATH, MODELS_FOLDER, AE_MODEL_FILE_NAME)
    ae_model.load_state_dict(torch.load(ae_model_file))
    cgan_dataset = train_CGAN.CGANDataset(ae_model, events, annotation, metadata, 
                                          tot_chunks=CGAN_TOT_CHUNKS, 
                                          chunk_length=CHUNK_LENGTH, data_type=DATA_TYPE, 
                                          chunk_dim=AE_LATENT_DIM)
    
    model_path = os.path.join(HOME_PATH, MODELS_FOLDER)
    examples_path = os.path.join(HOME_PATH, EXAMPLES_FOLDER)
    
    cgan_disc_model = CGAN.ConvDiscriminator(CGAN_TOT_CHUNKS, AE_LATENT_DIM)
    cgan_gen_model = CGAN.ConvGenerator(CGAN_TOT_CHUNKS, CGAN_LATENT_DIM, AE_LATENT_DIM)
    
    print(cgan_disc_model)
    print(cgan_gen_model)
    
    return train_CGAN.train(cgan_disc_model, cgan_gen_model, ae_model, cgan_dataset,
                     tot_chunks = CGAN_TOT_CHUNKS, latent_dim=CGAN_LATENT_DIM, 
                     epochs=CGAN_EPOCHS, lr=CGAN_LEARNING_RATE, 
                     batch_size=CGAN_BATCH_SIZE, num_channels=AE_LATENT_DIM, 
                     chunk_length=CHUNK_LENGTH, lambda_gp=CGAN_LAMBDA_GP, 
                     lambda_fm=CGAN_LAMBDA_FM, model_path=model_path,
                     disc_model_file=CGAN_DISC_MODEL_FILE_NAME, 
                     gen_model_file=CGAN_GEN_MODEL_FILE_NAME, data_type=DATA_TYPE,
                     examples_path=examples_path)
    
def ContinueTrainCGAN(events, annotation, metadata):
    #loads currently saved model and keeps training
    ae_model = autoencoder.AutoEncoder(data_type=DATA_TYPE, hidden_dim=AE_LATENT_DIM,
                                       max_n=MAX_CHUNK_SIZE)
    ae_model_file = os.path.join(HOME_PATH, MODELS_FOLDER, AE_MODEL_FILE_NAME)
    ae_model.load_state_dict(torch.load(ae_model_file))
    cgan_dataset = train_CGAN.CGANDataset(ae_model, events, annotation, metadata, 
                                          tot_chunks=CGAN_TOT_CHUNKS, 
                                          chunk_length=CHUNK_LENGTH, data_type=DATA_TYPE, 
                                          chunk_dim=AE_LATENT_DIM)
    
    model_path = os.path.join(HOME_PATH, MODELS_FOLDER)
    examples_path = os.path.join(HOME_PATH, EXAMPLES_FOLDER)
    
    disc_model_file = os.path.join(model_path, CGAN_DISC_MODEL_FILE_NAME)
    gen_model_file = os.path.join(model_path, CGAN_GEN_MODEL_FILE_NAME)
    cgan_disc_model = CGAN.ConvDiscriminator(AE_LATENT_DIM)
    cgan_gen_model = CGAN.ConvGenerator(CGAN_LATENT_DIM, AE_LATENT_DIM)
    cgan_disc_model.load_state_dict(torch.load(disc_model_file))
    cgan_gen_model.load_state_dict(torch.load(gen_model_file))
    
    train_CGAN.train(cgan_disc_model, cgan_gen_model, ae_model, cgan_dataset,
                     tot_chunks = CGAN_TOT_CHUNKS, latent_dim=CGAN_LATENT_DIM, 
                     epochs=CGAN_EPOCHS, lr=CGAN_LEARNING_RATE, 
                     batch_size=CGAN_BATCH_SIZE, num_channels=AE_LATENT_DIM, 
                     chunk_length=CHUNK_LENGTH, lambda_gp=CGAN_LAMBDA_GP, 
                     lambda_fm=CGAN_LAMBDA_FM, model_path=model_path,
                     disc_model_file=CGAN_DISC_MODEL_FILE_NAME, 
                     gen_model_file=CGAN_GEN_MODEL_FILE_NAME, data_type=DATA_TYPE,
                     examples_path=examples_path)
    
    
def TestAE(events, annotation, metadata):
    ae_valid_dataset = train_autoencoder.AutoEncoderDataset(events, annotation, metadata,
                                                            CHUNK_LENGTH, DATA_TYPE, 
                                                            AUG_TYPE, 'validation')
    ae_model = autoencoder.AutoEncoder(data_type=DATA_TYPE, hidden_dim=AE_LATENT_DIM,
                                       max_n=MAX_CHUNK_SIZE)
    model_file = os.path.join(HOME_PATH, MODELS_FOLDER, AE_MODEL_FILE_NAME)
    ae_model.load_state_dict(torch.load(model_file))
    examples_path = os.path.join(HOME_PATH, EXAMPLES_FOLDER)
    return utils.TestAutoEncoder(events, annotation, metadata, ae_model, 
                          DATA_TYPE, CHUNK_LENGTH, examples_path)
    
    
    
def draw_sample(ae_model, gen_model):
    #given autoencoder and GAN models, play a sample of music
    distribution = torch.distributions.MultivariateNormal(torch.zeros(CGAN_LATENT_DIM), 
                                                          torch.eye(CGAN_LATENT_DIM)*10.)
    latent_sample = distribution.sample([1])
    ae_model.eval()
    gen_model.eval()
    
    torch.set_printoptions(precision=3, sci_mode=False)
    
    output = gen_model(latent_sample).detach().cpu() #(1,num_channels,16)
    output = torch.transpose(torch.squeeze(output),0,1) #(16,num_channels)
    norm_chunks, ns = ae_model.decoder(output)
    norm_chunks = norm_chunks.detach().cpu()
    ns = ns.detach().cpu()
    print('Output of decoder:', norm_chunks)
    print(ns)
    events = utils.NormalizedChunksToEvents(norm_chunks, ns, 
                                            960, DATA_TYPE, CHUNK_LENGTH)
    print('Events:', events)
    midi_file = utils.EventsToMidi(events, 960)
    midi_file.save('sample.mid')
    FluidSynth().play_midi('sample.mid')