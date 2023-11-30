#various utility functions get shoved here

import mido
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import lognorm
from tqdm import tqdm
import torch
from collections import Counter



#damper pedal is MIDI control 64, sostenuto is 66, soft is 67
PedalIDToMidiDict = {88:64, 89:66, 90:67}
MidiToPedalIDDict = {v:k for k,v in PedalIDToMidiDict.items()}
NoteIDToMidiDict = {x:x+21 for x in range(88)} #lowest note on keyboard is MIDI 21
MidiToNoteIDDict = {v:k for k,v in NoteIDToMidiDict.items()}

#How much space to preallocate for all events?
MAX_TOTAL_EVENTS = 10000000

#parameters for log-normal distribution, estimated by scipy's lognorm.fit
LOGNORM_S = 1.125998957286227
LOGNORM_LOC = -0.022173461168885033
LOGNORM_SCALE = 84.32398063921644




#PUT THIS IN README:
#An event is a note press or pedal change, and is represented as a vector
#(note/pedal ID, start time in ticks, duration of event in ticks, intensity)
#The note/pedal ID is an integer between 0 and 90 (inclusive):
#   0-87  piano keys, lowest to highest
#   88    damper pedal
#   89    sostenuto pedal
#   90    soft pedal
#Start time means start time in ticks from the beginning of the chunk
#For a note event, intensity means velocity (1-127), whereas for pedals it means the 
#MIDI value, which is more or less how pressed down it is (1-127)         





def GetMidiFile(metadata, data_path, track):
    #Takes in track ID and metadata file and returns corresponding MidiFile object
    #data_path is the folder where the maestro database is
    path = os.path.join(data_path, metadata['midi_filename'][track])
    if not os.path.exists(path):
        print('Could not find MIDI file with index', track, 'at location', path)
        return None
    try:
        midi_file = mido.MidiFile(path)
    except:
        print('Could not read MIDI file with index', track, 'at location', path)
        return None
    return midi_file

def GetRawMetadata(data_path):
    #reads raw metadata file
    os.chdir(data_path)
    if os.path.isfile('maestro-v3.0.0.csv'):
        try:
            return pd.read_csv('maestro-v3.0.0.csv')
        except:
            print('Error: could not read metadata file')
            return None
    else:
        print('Error: metadata file maestro-v3.0.0.csv not found')
        return None

def MakeEvents(metadata, data_path, track):
    #Takes in metadata DataFrame and a track ID
    #Returns a numpy array of events, dimension N*4, and ticks per second for that track
    #data_path is the folder where the maestro database is
    midi_file = GetMidiFile(metadata, data_path, track)
    if not midi_file: #if we couldn't read it, return None
        return None
    ticks_per_second = midi_file.ticks_per_beat*2 #everything at 120 bpm
    note_tracker = [None]*91
    #note_tracker[i] is None means ith note/pedal is off; value (n,t) means note/pedal
    #ID is held over from nth event at overall tick t
    curr_tick = 0
    curr_event = 0
    
    events = []
    for message in midi_file:
        if message.is_meta or message.type == 'program_change':
            continue #ignore metamessages
        
        new_event = True #does this message create a new event of any kind?
        new_pedal_event = True #does this message create a new pedal event?
        if message.type == 'note_on' and message.velocity != 0: #note on message
            note_pedal_ID = MidiToNoteIDDict[message.note]
            intensity = message.velocity
            new_pedal_event = False
        elif message.type == 'note_on': #note on with zero velocity = note off
            note_pedal_ID = MidiToNoteIDDict[message.note]
            new_event = False
            new_pedal_event = False
        elif message.is_cc() and message.value != 0: #pedal on or change event
            note_pedal_ID = MidiToPedalIDDict[message.control]
            intensity = message.value
        elif message.is_cc(): #pedal off event
            note_pedal_ID = MidiToPedalIDDict[message.control]
            new_event = False
            new_pedal_event = False
        else:
            print('Unexpected MIDI input for message', message)
            continue
        
        curr_tick += round(message.time*ticks_per_second)
        
        if (not new_event or new_pedal_event) and note_tracker[note_pedal_ID]: 
            #note off event or pedal change event, so fill in previous event using note_tracker    
            duration = curr_tick - note_tracker[note_pedal_ID][1]
            events[note_tracker[note_pedal_ID][0]][2] = duration
            note_tracker[note_pedal_ID] = None
        if new_event: #add new event and update note_tracker
            next_event = [note_pedal_ID, curr_tick, 0, intensity]
            events.append(next_event)
            note_tracker[note_pedal_ID] = (curr_event, curr_tick)
            curr_event += 1
    return np.array(events, dtype='int32'), ticks_per_second

def BinPedalEvents(events, pedal_bins):
    #Takes a numpy array of events and consolidates pedal events into pedal_bins 
    #approximately equally spaced intensity bins (bin 0 is always 0 intensity by
    #itself)
    #If pedal_bins = 0, does nothing
    #For example, if pedal_bins is small enough and there is a damper pedal at 
    #intensity #100 for 40 ticks followed by intensity 110 for 10 ticks, we'd combine 
    #into intensity 102 for 50 ticks
    #This can reduce the number of events, hence the dimensionality of the data,
    #without sacrificing too much fidelity
    #Returns a numpy array of events after binning
    
    if pedal_bins == 0:
        return events
    
    curr_bin = [0,0,0] #can be 0, 1, 2, 3, 4 for damper, sostenuto, soft pedals
    curr_intensity_ticks = [0,0,0] #keep adding intensity*ticks to compute avg
    curr_ticks = [0,0,0] #how long has the pedal event been going on?
    new_events = []
    
    modulus = math.ceil(127/pedal_bins)
    def GetBin(intensity):
        if intensity == 0:
            return 0
        return (intensity-1)//modulus + 1
    
    for event in events:
        if event[0] == 88: #damper pedal
            pedal_index = 0
        elif event[0] == 89: #sostenuto pedal
            pedal_index = 1
        elif event[0] == 90: #soft pedal
            pedal_index = 2
        else: #not a pedal event
            new_events.append(event.tolist())
            continue
        intensity = event[3]
        new_bin = GetBin(intensity)
        if new_bin != 0 and curr_bin[pedal_index] == 0: #pedal on
            curr_bin[pedal_index] = new_bin
        elif new_bin != curr_bin[pedal_index]: #different bin (incl. pedal off), so add new averaged event
            if curr_ticks[pedal_index] == 0: #sometimes there are 0 duration pedal events
                avg_intensity = intensity #if this happens, just take last intensity as avg
            else:
                avg_intensity = round(curr_intensity_ticks[pedal_index]/curr_ticks[pedal_index])
            new_events.append([event[0], event[1]-curr_ticks[pedal_index], curr_ticks[pedal_index], avg_intensity])
            curr_bin[pedal_index] = new_bin
            curr_ticks[pedal_index] = 0
            curr_intensity_ticks[pedal_index] = 0
        curr_ticks[pedal_index] += event[2]
        curr_intensity_ticks[pedal_index] += event[2]*intensity
            
    new_events.sort(key = lambda x:x[1])
    return np.array(new_events)


def ChunkEvents(events, ticks_per_second, starting_chunk_ID, chunk_length):
    #Takes an array of events and assigns each to a chunk of duration chunk_length
    #Also start time is reassigned to measure from start of chunk, not beginning of track
    #Input is N*4 array, output is N*5 and number of chunks
    ticks_per_chunk = ticks_per_second * chunk_length
    bins = np.arange(0, events[-1,1], ticks_per_chunk)
    num_chunks = bins.shape[0]
    IDs_from_zero = (np.searchsorted(bins, events[:,1], side='right') - 1)
    IDs = IDs_from_zero + starting_chunk_ID
    chunked_events = np.concatenate([events, np.expand_dims(IDs,1)], axis=1)
    chunked_events[:,1] = chunked_events[:,1] - ticks_per_chunk*(IDs_from_zero)
    return chunked_events, num_chunks

def RemoveOverflows(chunked_events, max_chunk_size):
    #Takes in N*5 array of chunked events where last col is chunk ID and removes any events
    #in excess of max_chunk_size in any given chunk. Also returns number of removals (overflows)
    good_indices = []
    num_overflows = 0
    prev_index = chunked_events[0,4]
    curr_count = 0
    for i in range(chunked_events.shape[0]):
        if chunked_events[i,4] == prev_index:
            curr_count += 1
        else:
            curr_count = 0
        if curr_count < max_chunk_size:
            good_indices.append(i)
        else:
            num_overflows += 1
        prev_index = chunked_events[i,4]
    return chunked_events[good_indices], num_overflows

def GetAllEvents(raw_metadata, data_path, pedal_bins, chunk_length, max_chunk_size):
    #Reads all MIDI files, does pedal binning and chunking
    #Adds relevant columns to metadata file: start_event, end_event, start_chunk, end_chunk,
    #ticks_per_second, and num_overflows
    #Returns events N*5 events array, where last col is chunk ID, a chunk annotation dict, and
    #updated metadata dataframe
    #chunk annotation dict has keys chunk IDs and values tuples (track, number of events, starting event)
    
    #Preallocate space in events array
    events = np.zeros((MAX_TOTAL_EVENTS, 5),dtype='int32')
    curr_event = 0
    curr_chunk = 0
    total_overflows = 0
    chunk_annotation = {}
    raw_metadata['ticks_per_second'] = 0
    raw_metadata['start_event'] = 0
    raw_metadata['end_event'] = 0
    raw_metadata['start_chunk'] = 0
    raw_metadata['end_chunk'] = 0
    raw_metadata['num_overflows'] = 0
    
    for track in tqdm(raw_metadata.index, desc='Reading MIDI files and preparing data'):
        #read MIDI file
        
        output = MakeEvents(raw_metadata, data_path, track)
        
        if output is None: #could not find or read MIDI file
            continue
            
        (curr_events, ticks_per_second) = output
        
        #bin pedal events
        curr_events = BinPedalEvents(curr_events, pedal_bins)
        
        #chunk events
        chunked_events, num_chunks = ChunkEvents(curr_events, ticks_per_second, curr_chunk, chunk_length)
        
        #remove overflows
        chunked_events, num_overflows = RemoveOverflows(chunked_events, max_chunk_size)
        
        keys = range(curr_chunk, curr_chunk + num_chunks)
        counter = Counter(chunked_events[:,4]) #count number of events in each chunk
        ns = [counter[key] for key in keys] #for each chunk ID, how many ns?
        start_indices_array = np.concatenate(([0],np.cumsum(ns)))[:-1]
        temp_dict = {key:(track, ns[key-curr_chunk], curr_event+start_indices_array[key-curr_chunk])
                     for key in keys}
        
        chunk_annotation.update(temp_dict)
        
        curr_event_length = chunked_events.shape[0]
        
        #update metadata
        raw_metadata.at[track,'ticks_per_second'] = ticks_per_second
        raw_metadata.at[track,'start_event'] = curr_event
        raw_metadata.at[track,'end_event'] = curr_event + curr_event_length
        raw_metadata.at[track,'start_chunk'] = curr_chunk
        raw_metadata.at[track,'end_chunk'] = curr_chunk + num_chunks
        raw_metadata.at[track,'num_overflows'] = num_overflows

        events[curr_event:curr_event+curr_event_length,:] = chunked_events
        
        curr_event += curr_event_length
        curr_chunk += num_chunks
        total_overflows += num_overflows
        
        
    print('There were', total_overflows,'total chunk overflows')
    return events[:curr_event,:], chunk_annotation, raw_metadata




def EventsToMidi(events, ticks_per_second):
    #Takes in a numpy array of events (with start times measured in absolute ticks, N*4)
    #and returns a corresponding MidiFile object of the same format as the MAESTRO database
    #Note that converting midi to events and back may not be the identity, since
    #simultaneous messages may be shuffled and events that last for zero ticks may be
    #dropped.
    midi_file = mido.MidiFile(type=1, ticks_per_beat=int(ticks_per_second/2))
    prelimtrack = mido.MidiTrack()
    prelimtrack.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    prelimtrack.append(mido.MetaMessage('time_signature', numerator=4, denominator=4,
                                        clocks_per_click=24,
                                        notated_32nd_notes_per_beat=8, time=0))
    prelimtrack.append(mido.MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(prelimtrack)
    track = mido.MidiTrack()
    track.append(mido.Message('program_change', channel=0, program=0, time=0))
    
    #first unroll into separate on/off events
    #an on/off event is stored as (note/pedal ID, time in absolute ticks, intensity)
    #intensity = 0 means it's an off event, otherwise on
    #ignore any null events
    on_off_events = []
    for event in events:
        note_pedal_ID = event[0]
        if note_pedal_ID == 91: #Null event
            continue
        start_time = event[1]
        end_time = event[1] + event[2]
        intensity = event[3]
        on_off_events.append([note_pedal_ID, start_time, intensity])
        on_off_events.append([note_pedal_ID, end_time, 0])
    on_off_events.sort(key = lambda x: x[1]) #sort by time
    on_off_events = np.array(on_off_events)
    
    #now go through on/off events and find any pedal off events that are simultaneous
    #with a pedal on event (same pedal). Remove unnecessary off events.
    index = 0
    num_on_off_events = on_off_events.shape[0]
    unnecessary_rows = []
    while index < num_on_off_events - 1:
        same_time_as_next_event = (on_off_events[index][1] == on_off_events[index+1][1])
        same_ID = (on_off_events[index][0] == on_off_events[index+1][0])
        pedal_event = (on_off_events[index][0] >= 88)
        off_then_on = (on_off_events[index][2] == 0 and on_off_events[index+1][2] != 0)
        on_then_off = (on_off_events[index][1] != 0 and on_off_events[index+1][2] == 0)
        if same_time_as_next_event and same_ID and pedal_event:
            if off_then_on: #off event is first
                unnecessary_rows.append(index)
                index += 2
            elif on_then_off: #on event is first
                unnecessary_rows.append(index+1)
                index += 2
            else: #simultaneous on or off events; shouldn't really get here
                index += 1
        else:
            index += 1
    
    #remove all unnecessary rows
    on_off_events = np.delete(on_off_events, unnecessary_rows, axis=0)
    
    #convert to midi
    prev_time = 0
    for event in on_off_events:
        note_pedal_ID = event[0]
        curr_time = event[1]
        time_gap = curr_time - prev_time
        intensity = event[2]
        if note_pedal_ID >= 88: #pedal event
            track.append(mido.Message('control_change', channel=0, 
                                      control=PedalIDToMidiDict[note_pedal_ID], 
                                      value=intensity, time=time_gap))
        else: #note event
            track.append(mido.Message('note_on', channel=0, 
                                      note=NoteIDToMidiDict[note_pedal_ID],
                                      velocity=intensity, time=time_gap))
        prev_time = curr_time
        
    track.append(mido.MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track)
    return midi_file   


def GetDim(data_type):
    #Return how many dimensions each event takes up under various data_type options
    if data_type == 'OHE':
        return 94
    elif data_type == 'ponOHE':
        return 27
    elif data_type == 'pnOHE':
        return 20
    elif data_type == 'pon':
        return 6
    elif data_type == 'none':
        return 4
    

    

def NormalizeChunk(chunk, ticks_per_second, chunk_length, data_type):
    #Normalize chunked data according to data_type specification for use by autoencoder
    #Input is N*4 numpy array
    #Output is normalized N*dim pytorch tensor, where dim is determined according to data_type
    #In all cases, the last three indices are:
    #  start time, standardized to [0,1] by dividing by ticks_per_second
    #  duration divided by ticks_per_second
    #  intensity, standardized to [0,1]
    #Also returns N*? array of OHE indices, for faster evaluation of loss function,
    #where ? is determined by data_type (zero if no OHE used)
    #This is called in __getitem__() routine of the AutoEncoderDataset, so is not precomputed,
    #to save on memory for one hot encoding and such
    dim = GetDim(data_type)
    N = chunk.shape[0]
    norm_chunk = torch.zeros((N, dim))
    
    norm_lengths = lognorm.cdf(chunk[:,2], LOGNORM_S, LOGNORM_LOC, LOGNORM_SCALE)
    
    chunk = torch.from_numpy(chunk)
    
    half_ticks_per_chunk = ticks_per_second*chunk_length*.5
    
    #start times should be approximately uniform on [-1,1]
    norm_chunk[:, -3] = (chunk[:, 1]-half_ticks_per_chunk)/half_ticks_per_chunk
    
    #model event lengths with log normal distribution with estimated parameters
    #will then lie in [0,1]
    norm_chunk[:, -2] = torch.from_numpy(norm_lengths)
    
    #normalize so intensities lie in [-1,1]
    norm_chunk[:, -1] = (chunk[:, 3]-64.)/64. 
    
    if data_type == 'OHE':
        norm_chunk[np.arange(N), chunk[:, 0]] = 1.0
        IDs = chunk[:,0]
    elif data_type == 'ponOHE':
        #send all notes to 0 and pedals to 1, 2, 3
        pedIDs = torch.clamp(chunk[:,0] - 87, min=0)
        
        #this makes default octave for pedal events equal to 7, arbitrarily
        octIDs = chunk[:,0]//12
        
        #this makes default notes for pedal events equal to 4, 5, 6 (resp.), arbitrarily
        noteIDs = chunk[:,0]%12
        
        norm_chunk[np.arange(N), pedIDs] = 1.0 #OHE of pedal possibilities
        norm_chunk[np.arange(N), octIDs + 4] = 1.0 #OHE of octave possibilities
        norm_chunk[np.arange(N), noteIDs + 12] = 1.0 #OHE of note possibilities 
        IDs = torch.transpose(torch.vstack([pedIDs, octIDs, noteIDs]), 0, 1)
    elif data_type == 'pnOHE':
        pedIDs = torch.clamp(chunk[:,0] - 87, min=0)
        octIDs = chunk[:,0]//12
        noteIDs = chunk[:,0]%12
        
        norm_chunk[np.arange(N), pedIDs] = 1.0 #OHE of pedal possibilities
        
        #octave normalized to lie in [-1,1]
        norm_chunk[:, 4] = (octIDs - 3.5)/3.5
        
        norm_chunk[np.arange(N), noteIDs + 5] = 1.0 #OHE of note possibilities
        IDs = torch.transpose(torch.vstack([pedIDs, noteIDs]), 0, 1)
    elif data_type == 'pon':
        #send notes to 0 and pedals to 1/3, 2/3, 1
        norm_chunk[:, 0] = torch.clamp(chunk[:,0] - 87, min=0)/3.
        
        #octave normalized to lie in [-1,1]
        norm_chunk[:, 1] = ((chunk[:,0]//12 - 3.5)/3.5)
        
        #note normalized to lie in [-1,1]
        norm_chunk[:, 2] = ((chunk[:,0]%12 - 5.5)/5.5)
        
        IDs = torch.zeros((N,0))
    elif data_type == 'none':
        norm_chunk[:, 0] = chunk[:, 0]
        IDs = torch.zeros((N,0))

    return norm_chunk, IDs
    
    

def NormalizedChunksToEvents(norm_chunk, ns, ticks_per_second, data_type, chunk_length):
    #Convert normalized chunks (e.g. output of autoencoder, shape N*dim) into a numpy array 
    #of events, shape N*4. Needs data of ns to separate chunks
    #Also rejigger so start times are in absolute ticks
    N = norm_chunk.shape[0]
    output = np.zeros((N, 4), dtype='int32')
    norm_chunk = torch.Tensor.numpy(norm_chunk)
    
    half_ticks_per_chunk = ticks_per_second*chunk_length*.5

    #start time from beginning of chunk must be between 0 and ticks_per_second*chunk_length    
    output[:, 1] = np.clip(norm_chunk[:,-3]*half_ticks_per_chunk+half_ticks_per_chunk, 0, 
                           ticks_per_second*chunk_length)
    
    #adjust start times to be from beginning
    offsets = torch.repeat_interleave(torch.arange(ns.shape[0]), ns, 
                                      dim=0)*ticks_per_second*chunk_length
    output[:, 1] = output[:, 1] + offsets.numpy()
    
    #normalized duration must be between 0 and 1
    eps = 0.000001
    temp = np.clip(norm_chunk[:,-2], eps, 1-eps)
    output[:, 2] = lognorm.ppf(temp, LOGNORM_S, LOGNORM_LOC, LOGNORM_SCALE)
    
    #intensity must be between 0 and 127 inclusive
    output[:, 3] = np.clip(norm_chunk[:, -1]*64.+64., 0, 127)
    
    #now get the note/pedal ID, depending on data_type
    #some of this is super inefficient but it doesn't really matter since 
    #we don't need it for training
    if data_type == 'OHE':
        output[:, 0] = np.argmax(norm_chunk[:,:91], axis=1)
    elif data_type == 'ponOHE':
        pedIDs = np.argmax(norm_chunk[:,:4], axis=1)
        octIDs = np.argmax(norm_chunk[:,4:12], axis=1)
        noteIDs = np.argmax(norm_chunk[:,12:24], axis=1)
        for i in range(N):
            if pedIDs[i] == 1:
                output[i, 0] = 88 #damper
            elif pedIDs[i] == 2:
                output[i, 0] = 89 #sostenuto
            elif pedIDs[i] == 3:
                output[i, 0] = 90 #soft
            else:
                output[i, 0] = np.minimum(12*octIDs[i] + noteIDs[i], 87)    
    elif data_type == 'pnOHE':
        pedIDs = np.argmax(norm_chunk[:,:4], axis=1)
        #octave must be integer between 0 and 7 inclusive
        octIDs = np.maximum(np.minimum(np.round(norm_chunk[:,5]), 7), 0)
        noteIDs = np.argmax(norm_chunk[:,5:13], axis=1)
        for i in range(N):
            if pedIDs[i] == 1:
                output[i, 0] = 88 #damper
            elif pedIDs[i] == 2:
                output[i, 0] = 89 #sostenuto
            elif pedIDs[i] == 3:
                output[i, 0] = 90 #soft
            else:
                output[i, 0] = np.minimum(12*octIDs[i] + noteIDs[i], 87)
    elif data_type == 'pon':
        #pedal must be integer between 0 and 3 inclusive
        pedIDs = np.maximum(np.minimum(np.round(norm_chunk[:,0]*3.), 3), 0)
        #octave must be integer between 0 and 7 inclusive
        temp = norm_chunk[:,1]*3.5 + 3.5
        octIDs = np.maximum(np.minimum(np.round(temp), 7), 0)
        #note must be integer between 0 and 11 inclusive
        temp = norm_chunk[:,2]*5.5 + 5.5
        noteIDs = np.maximum(np.minimum(np.round(temp), 11), 0)
        for i in range(N):
            if pedIDs[i] == 1:
                output[i, 0] = 88 #damper
            elif pedIDs[i] == 2:
                output[i, 0] = 89 #sostenuto
            elif pedIDs[i] == 3:
                output[i, 0] = 90 #soft
            else:
                output[i, 0] = np.minimum(12*octIDs[i] + noteIDs[i], 87)
    elif data_type == 'none':
        #note/ped ID must be between 0 and 90 inclusive
        output[i, 0] = np.maximum(np.minimum(np.round(norm_chunk[i, 0]), 90), 0)
   
    return output




def TestAutoEncoder(events, annotation, metadata, ae_model, data_type, chunk_length, examples_path):
    #selects a random track, saves as midi, encodes * decodes then saves again
    ae_model.eval()
    os.chdir(examples_path)
    track = np.random.randint(len(metadata.index))
    print('Track', track)
    start_chunk = metadata['start_chunk'][track]
    end_chunk = metadata['end_chunk'][track]
    start_event = metadata['start_event'][track]
    end_event = annotation[end_chunk][2]
    chunks = events[start_event:end_event,:4].copy()
    ticks_per_second = metadata['ticks_per_second'][track]
    ns = torch.tensor([annotation[chunk_index][1] for chunk_index in
                        range(start_chunk, end_chunk)], dtype=torch.int32)  
    
    #adjust start times to be from beginning
    offsets = torch.repeat_interleave(torch.arange(ns.shape[0]), ns, dim=0)*ticks_per_second*chunk_length
    offset_chunks = chunks.copy()
    offset_chunks[:, 1] = chunks[:, 1] + offsets.numpy()
    real_midi = EventsToMidi(offset_chunks, ticks_per_second)
    real_midi.save('temp_real.mid')
    print(chunks[:100,:])
    norm_chunks, _ = NormalizeChunk(chunks, ticks_per_second, chunk_length, data_type)
    torch.set_printoptions(precision=3, sci_mode=False)
    print(norm_chunks[:100,:])
    #print(norm_chunks.mean())
    with torch.set_grad_enabled(False):
        encoded = ae_model.encoder(norm_chunks, ns).detach()
        decoded, ns = ae_model.decoder(encoded)
    print(decoded[:100,:])
    decoded_chunks = NormalizedChunksToEvents(decoded, ns, ticks_per_second, 
                                              data_type, chunk_length)
    print(decoded_chunks[:100,:])
    decoded_midi = EventsToMidi(decoded_chunks, ticks_per_second)
    decoded_midi.save('temp_fake.mid')
    return encoded