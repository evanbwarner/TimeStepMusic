#DEPRECATED since we no longer write events to a file
def WriteAllEvents(df, overwrite = False, binning = False):
    #Takes each track as recorded in the metadata file, converts to events, and saves
    #to EVENTS_FOLDER/train, EVENTS_FOLDER/validation, or EVENTS_FOLDER/test
    #Each file will be of the form row.npy, where row is the ID for the piece in
    #the metadata file
    #If binning is True, we apply BinPedalEvents to everything
    #Can take a few minutes to run
    os.chdir(FOLDER_PATH)
    if not os.path.exists(EVENTS_FOLDER):
        os.mkdir(EVENTS_FOLDER)
    os.chdir(os.path.join(FOLDER_PATH, EVENTS_FOLDER))
    if not os.path.exists('train'):
        os.mkdir('train')
    if not os.path.exists('validation'):
        os.mkdir('validation')
    if not os.path.exists('test'):
        os.mkdir('test')
    
    for row in tqdm(df.index, desc='Reading MIDI files to event arrays'):
        train_valid_test = df['split'][row]
        out_path = os.path.join(FOLDER_PATH, EVENTS_FOLDER, train_valid_test, str(row)+'.npy')
        if os.path.exists(out_path) and not overwrite: #don't overwrite
            continue
        if binning:
            events = BinPedalEvents(MakeEvents(df, row), NUM_PEDAL_BINS)
        else:
            events = MakeEvents(df, row)
        if len(events.shape) == 0:
            print('Could not convert MIDI file with index', row, 'to event format')
            continue
        os.chdir(os.path.join(FOLDER_PATH, EVENTS_FOLDER, train_valid_test))
        np.save(str(row)+'.npy', events)

#DEPRECATED since we no longer write events to a file
def GetEvents(df, row):
    #Loads and returns the events array corresponding to the row index
    os.chdir(os.path.join(FOLDER_PATH, EVENTS_FOLDER, df['split'][row]))
    return np.load(str(row)+'.npy')

#DEPRECATED: functionality now in __init__() method of AutoEncoderDataset
def GetMetadata():
    #Returns a DataFrame with metadata. If METADATA_FILE already exists, just load it.
    #Otherwise take raw metadata and augment with ticks per second column and loads into
    #METADATA_FILE
    #Can take a few minutes if we have to read all the MIDI files
    os.chdir(os.path.join(FOLDER_PATH, MAESTRO_FOLDER))
    if os.path.isfile(METADATA_FILE):
        try:
            return pd.read_csv(METADATA_FILE)
        except:
            print('Could not read metadata file at', os.path.join(FOLDER_PATH, MAESTRO_FOLDER, METADATA_FILE))
            return None
    elif os.path.isfile(RAW_METADATA_FILE):
        try:
            raw_df = pd.read_csv(RAW_METADATA_FILE)
            df = WriteAllTicksPerSecond(raw_df)
            df.to_csv(METADATA_FILE, index=False)
            return df
        except:
            print('Could not read raw metadata file at', os.path.join(FOLDER_PATH, MAESTRO_FOLDER, RAW_METADATA_FILE))
            return None
    else:
        print('Could not find any metadata file')
        return None
    
    


#DEPRECATED in favor of GetAutoencoderData which doesn't write file
def WriteAutoencoderData(df):
    #Makes train/validate/test files for autoencoder. WriteAllEvents (hopefully with binning)
    #needs to have been already called so we have events files.
    #Each file is a single pytorch tensor of integers of size N*5 where N is the total number
    #of events used. The first four columns are the event data and the last is the chunk ID
    #Also write annotation dfs of recording the track each chunk came from and where in the data
    #they lie
            
    curr_chunk_ID = [0,0,0] #train/valid/test
    tot_events = [0,0,0] #train/valid/test
    num_overflows = [0,0,0] #train/valid/test
    used_events = [0,0,0] #train/valid/test
    annotations = [[],[],[]]
        
    arrays = [np.zeros((8000000, 5), dtype='int32'), 
              np.zeros((1000000, 5), dtype='int32'), 
              np.zeros((1000000, 5), dtype='int32')] #pre-allocate here for safety    
    
    for row in tqdm(df.index, desc='Writing data for autoencoder'):
        if df['split'][row] == 'train':
            index = 0
        elif df['split'][row] == 'validation':
            index = 1
        elif df['split'][row] == 'test':
            index = 2
        ticks_per_second = df['ticks per second'][row]
        events = GetEvents(df, row)
        num_events = events.shape[0]
        (chunked_events, overflows), ns = ChunkEvents(events, ticks_per_second, curr_chunk_ID[index])
        num_events -= overflows
        
        arrays[index][used_events[index]:used_events[index]+num_events,:] = chunked_events
        num_overflows[index] += overflows
        for chunk_ID in range(curr_chunk_ID[index], chunked_events[-1,4] + 1): #write annotations
            curr_n = ns[chunk_ID - curr_chunk_ID[index]]
            annotations[index].append([row, curr_n, used_events[index]])
            used_events[index] += curr_n
        curr_chunk_ID[index] = chunked_events[-1,4] + 1

    os.chdir(FOLDER_PATH)
    train_annotation_df = pd.DataFrame(annotations[0], columns = ['track', 'num_events', 'start_row'])
    valid_annotation_df = pd.DataFrame(annotations[1], columns = ['track', 'num_events', 'start_row'])
    test_annotation_df = pd.DataFrame(annotations[2], columns = ['track', 'num_events', 'start_row'])
        
    train_annotation_df.to_csv('autoencoder_train_annotations.df', index=False)
    valid_annotation_df.to_csv('autoencoder_valid_annotations.df', index=False)
    test_annotation_df.to_csv('autoencoder_test_annotations.df', index=False)
    train_data = arrays[0][:used_events[0],:]
    train_data = torch.from_numpy(train_data)
    torch.save(train_data, 'autoencoder_train.pt')
    valid_data = arrays[1][:used_events[1],:]
    valid_data = torch.from_numpy(valid_data)
    torch.save(valid_data, 'autoencoder_valid.pt')
    test_data = arrays[2][:used_events[2],:]
    test_data = torch.from_numpy(test_data)
    torch.save(test_data, 'autoencoder_test.pt')
    
    print('There were', num_overflows[0], 'overflows (omitted events) in training data, out of', used_events[0]+num_overflows[0], 'events')
    print('There were', num_overflows[1], 'overflows (omitted events) in validation data, out of', used_events[1]+num_overflows[1], 'events')
    print('There were', num_overflows[2], 'overflows (omitted events) in testing data, out of', used_events[2]+num_overflows[2], 'events')
    
    

#DEPRECATED to write this data at the same time we read the MIDI files anyway, for efficiency
def WriteAllTicksPerSecond(metadata, data_path):
    #Takes in raw metadata DataFrame and returns it with a new column with ticks per second info
    #data_path is the folder where the maestro database is
    metadata['ticks per second'] = np.NaN
    for track in tqdm(metadata.index, desc='Reading ticks per second data'):
        midi_file = GetMidiFile(metadata, data_path, track)
        #tempo in all MAESTRO files set at 120 bpm:
        metadata.at[track,'ticks per second'] = midi_file.ticks_per_beat*2 
    return metadata


#DEPRECATED
def GetAutoencoderData(metadata, events, track_annotation, split, max_chunk_size, chunk_length):
    #Makes a `chunkified' pytorch tensor of events data, which is internally stored by
    #the AutoEncoderDataset
    #Inputs: usual metadata, huge N*4 events tensor, annotation telling us starting index
    #of each track, others self-explanatory
    #Output: N*5 pytorch tensor where fifth col is chunk index,
    #a chunk annotation dictionary which associates (track number, length of chunk, 
    #start of chunk in data) to each chunk index,
    #and updates track annotation dictionary which associates (starting event ID, ending event ID,
    #starting chunk ID, ending chunk ID) to each track
    
    curr_chunk_index = 0
    tot_events = 0
    num_overflows = 0
    used_events = 0
    chunk_annotation = {}
    
    #preallocate: we'll end up with slightly fewer due to overflows
    chunked_events = np.zeros((events.shape[0],5), dtype='int32')
    
    for track in tqdm(metadata.index, desc='Chunking ' + split + ' data'):
        if track!=41:
            continue
        if metadata['split'][track] != split:
            continue
        
        ticks_per_second = metadata['ticks_per_second'][track]
        
        (track_start_index, track_end_index) = track_annotation[track]
        num_events = track_end_index - track_start_index
        curr_events = events[track_start_index:track_end_index,:]
        (curr_chunked_events, overflows), ns = ChunkEvents(curr_events, ticks_per_second, curr_chunk_index, max_chunk_size, chunk_length)
        num_events -= overflows
        tot_events += num_events
        

        
        chunked_events[used_events:used_events+num_events,:] = curr_chunked_events
        num_overflows += overflows
        
        end_chunk_index = curr_chunked_events[-1,4]+1
        for chunk_index in range(curr_chunk_index, end_chunk_index):
            curr_n = ns[chunk_index - curr_chunk_index]
            chunk_annotation[chunk_index] = (track, curr_n, used_events)
            used_events += curr_n 
            
        track_annotation[track] = (track_start_index, track_end_index, curr_chunk_index, end_chunk_index)
        curr_chunk_index = end_chunk_index
        
    
    print('There were ' + str(num_overflows) + ' total chunk overflows in the ' + split + ' data')
    return chunked_events[:tot_events,:], chunk_annotation, track_annotation



#DEPRECATED because unneccessary
def GetNs(events, metadata, track):
    #Get array ns of chunk lengths for given track, for input into encoder
    #could also use annotation but whatever here's another way with numpy magic
    start_chunk = metadata['start_chunk'][track]
    end_chunk = metadata['end_chunk'][track]
    start_event = metadata['start_event'][track]
    end_event = metadata['end_event'][track]
    bins = np.arange(0, end_chunk - start_chunk)
    ns = np.bincount(events[start_event:end_event,4] - start_chunk)
    return ns









    
def GetNormalizedChunks(df, row, datatype):
    #Get events data for track represented by row and return normalized chunks for
    #autoencoder use
    #Also returns array of n's (number of events in each chunk) and IDs for loss fnctn
    ticks_per_second = df['ticks per second'][row]
    events = utils.GetEvents(df, row)
    (chunked_events, _), ns = utils.ChunkEvents(events, ticks_per_second, 0)
    chunked_events = torch.from_numpy(chunked_events[:,:-1])
    ns = torch.from_numpy(ns)
    normalized_chunks, IDs = NormalizeChunk(chunked_events, ticks_per_second, datatype)
    
    return normalized_chunks, ns, IDs
    

def GetEncodedTrack(df, row, aemodel):
    #Returns track encoded according to autoencoder aemodel
    chunks, ns = GetNormalizedChunks(df, row)
    encoded = aemodel.encoder(chunks, ns).detach()
    return encoded



