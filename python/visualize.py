#tools for visualizing the raw data
import utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def VisualizeNumBins(df):
    #Take some random tracks and plots histograms of how binning changes the number 
    #of events.
    #Used to help select NUM_PEDAL_BINS hyperparameter
    #Only relevant if the events files aren't already binned!
    num_bins = [1, 2, 4, 8, 16, 32, 64, 128]
    rands = np.random.choice(len(df), 4)
    fig, axs = plt.subplots(2, 2, sharey=False, sharex=True, tight_layout=True)
    for i in tqdm(range(4), desc='Preparing examples'):
        row = rands[i]
        events = utils.GetEvents(df, row)
        num_events = []
        for num in num_bins:
            num_events.append(utils.BinPedalEvents(events, num).shape[0])
        label = str(row) + ': ' + df['canonical_title'][row] + ', ' + df['canonical_composer'][row]
        axs[i//2,i%2].set_title(label, fontsize=6)
        axs[i//2,i%2].plot(num_bins, num_events)
        axs[i//2,i%2].set_xscale('log')
        axs[i//2,i%2].set_xlabel('Number of pedal bins', fontsize=5)
        axs[i//2,i%2].set_ylabel('Number of events in track', fontsize=5)
        axs[i//2,i%2].set_xticks(ticks=num_bins,labels=num_bins)
    plt.show()
    
def GetChunkSizes(events, ticks_per_second, seconds_per_chunk = utils.SECONDS_PER_CHUNK):
    #used for visualization routines
    ticks_per_chunk = round(ticks_per_second*seconds_per_chunk)
    total_ticks = events[-1,1]
    num_chunks = math.ceil(total_ticks/ticks_per_chunk)
    output = []
    event_start_index = 0
    tot_events = len(events)
    for chunk_index in range(num_chunks):
        while event_start_index < tot_events and events[event_start_index,1] < ticks_per_chunk*chunk_index:
            event_start_index += 1
        event_end_index = event_start_index
        while event_end_index < tot_events and events[event_end_index,1] < ticks_per_chunk*(chunk_index+1):
            event_end_index += 1
        output.append(event_end_index-event_start_index)
        event_start_index = event_end_index
    return output
    
def VisualizeChunkSizes(df):
    #Take some random tracks and plots histograms on how many events land in 
    #time intervals of length SECONDS_PER_CHUNK *after binning*
    #Used to help select MAX_EVENTS_PER_CHUNK hyperparameter
    rands = np.random.choice(len(df), 4)
    fig, axs = plt.subplots(2, 2, sharey=False, sharex=True, tight_layout=True)
    for i in tqdm(range(4), desc='Getting chunk size data'):
        row = rands[i]
        events = utils.BinPedalEvents(utils.GetEvents(df, row))
        ticks_per_second = df['ticks per second'][row]
        num_events = GetChunkSizes(events, ticks_per_second)
        label = str(row) + ': ' + df['canonical_title'][row] + ', ' + df['canonical_composer'][row]
        axs[i//2,i%2].set_title(label, fontsize=6)
        axs[i//2,i%2].hist(num_events)
        axs[i//2,i%2].set_xlabel('Number of events per chunk after binning pedal events', fontsize=5)
        axs[i//2,i%2].set_ylabel('Number of chunks', fontsize=5)
    plt.show()
    
def AllChunkSizes(df):
    #Plot a histogram of all chunk sizes
    #Used to help select EVENTS_PER_CHUNK hyperparameter
    chunk_sizes = []
    for row in tqdm(df.index):
        events = utils.BinPedalEvents(utils.GetEvents(df, row))
        ticks_per_second = df['ticks per second'][row]
        chunk_sizes = chunk_sizes + GetChunkSizes(events, ticks_per_second)
    plt.hist(chunk_sizes, bins=range(max(chunk_sizes)))
    plt.title('All chunk sizes')
    plt.xlabel('Nontrivial events in chunk')
    plt.ylabel('Number of chunks')
    plt.show()
    
    
def ChunkSizesSplit(df):
    #plot histograms and print averages of chunk sizes for each split
    train_sizes = []
    valid_sizes = []
    test_sizes = []
    for row in df.index:
        events = utils.GetEvents(df, row)
        tps = df['ticks per second'][row]
        if df['split'][row] == 'train':
            train_sizes += GetChunkSizes(events, tps)
        elif df['split'][row] == 'validation':
            valid_sizes += GetChunkSizes(events, tps)
        else:
            test_sizes += GetChunkSizes(events, tps)
    fig, axs = plt.subplots(3, 1, tight_layout=True)
    bins = range(1,60)
    axs[0].set_title('Training data')
    axs[0].hist(train_sizes, bins)
    axs[1].set_title('Validation data')
    axs[1].hist(valid_sizes, bins)
    axs[2].set_title('Testing data')
    axs[2].hist(test_sizes, bins)
    plt.show()
    print('Average training chunk size:', np.average(train_sizes))
    print('Average validation chunk size:', np.average(valid_sizes))
    print('Average testing chunk size:', np.average(test_sizes))

def MaxChunkSizes(df):
    #What is the largest chunk size for each track after pedal binning?
    #Used to help select EVENTS_PER_CHUNK hyperparameter
    max_sizes = []
    for row in tqdm(df.index, desc='Getting chunk size data'):
        events = utils.BinPedalEvents(utils.GetEvents(df, row))
        ticks_per_second = df['ticks per second'][row]
        num_events = GetChunkSizes(events, ticks_per_second)
        max_sizes.append(max(num_events))
    plt.hist(max_sizes, bins=range(max(max_sizes)))
    plt.title('Largest chunk sizes for each track')
    plt.xlabel('Largest chunk size necessary')
    plt.ylabel('Number of tracks')
    plt.show()
    