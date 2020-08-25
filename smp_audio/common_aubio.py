import aubio
import numpy as np

def data_load_aubio(**kwargs):
    if 'filename' in kwargs:
        filename = kwargs['filename']
    else:
        filename = '/home/src/QK/data/sound-arglaaa-2018-10-25/24.wav'

    samplerate = 0
    src = aubio.source(filename, samplerate, channels=1)
    src.seek(0)

    return src

def data_stream_aubio(**kwargs):
    if 'filename' in kwargs:
        filename = kwargs['filename']
    else:
        filename = '/home/src/QK/data/sound-arglaaa-2018-10-25/24.wav'

    src = aubio.source(filename, hop_size=kwargs['frame_length'], channels=kwargs['num_channels'])
    src.seek(0)

    return tuple((src, src.samplerate))

def data_stream_get_aubio(src, **kwargs):
    while True:
        samples, read = src()
        if read < src.hop_size:
            break

        yield tuple((samples, read))

def compute_onsets_aubio(y=None, sr=None, **kwargs):
    """Compute onset detection function with aubio
    """
    if 'method' in kwargs:
        method = kwargs['method']
    else:
        # use aubio default
        method = 'default'

    src = data_load_aubio(filename=kwargs['filename'])

    # list of onsets: sparse, event-based?
    onsets = []
    # onset detector aubio
    onset_detector = aubio.onset(method=method, samplerate=src.samplerate)
    # onset_detector.set_threshold(1.0)

    while True:
        samples, read = src()
        # print(samples.shape)
        if read < src.hop_size:
            break
        onset_detector(samples)
        onsets.append(onset_detector.get_thresholded_descriptor())

    return {'onsets': np.array(onsets), 'src': src}

# def get_file_bpm(path, params=None):
def compute_tempo_beats_aubio(y=None, sr=None, **kwargs):
    """ Calculate the beats per minute (bpm) of a given file.
        path: path to the file
        param: dictionary of parameters
    """
    # if params is None:
    #     params = {}

    params = kwargs
    # default:
    samplerate, win_s, hop_s = 44100, 1024, 512
    path = '/home/src/QK/data/sound-arglaaa-2018-10-25/24.wav'
    method = 'specdiff'
    
    if 'path' in params:
        path = params['path']

    if 'method' in params:
        method = params['method']

    if 'mode' in params:
        if params.mode in ['super-fast']:
            # super fast
            samplerate, win_s, hop_s = 4000, 128, 64
        elif params.mode in ['fast']:
            # fast
            samplerate, win_s, hop_s = 8000, 512, 128
        elif params.mode in ['default']:
            pass
        else:
            raise ValueError("unknown mode {:s}".format(params.mode))
    # manual settings
    if 'samplerate' in params:
        samplerate = params.samplerate
    if 'win_s' in params:
        win_s = params.win_s
    if 'hop_s' in params:
        hop_s = params.hop_s
    print('    samplerate = {0}'.format(samplerate))
    print('    win_s = {0}'.format(win_s))
    print('    hop_s = {0}'.format(hop_s))

    s = aubio.source(path, samplerate, hop_s)
    print('aubio src samplerate = {0}'.format(s.samplerate))
    
    samplerate = s.samplerate
    o = aubio.tempo(method, win_s, hop_s, samplerate)
    # List of beats, in samples
    beats = []
    # Total number of frames read
    total_frames = 0

    # read audio
    while True:
        samples, read = s()
        is_beat = o(samples)
        if is_beat:
            this_beat = o.get_last_s()
            beats.append(this_beat)
            #if o.get_confidence() > .2 and len(beats) > 2.:
            #    break
        total_frames += read
        if read < hop_s:
            break

    def beats_to_bpm(beats, path):
        # if enough beats are found, convert to periods then to bpm
        if len(beats) > 1:
            if len(beats) < 4:
                print("few beats found in {:s}".format(path))
            bpms = 60./np.diff(beats)
            return np.median(bpms)
        else:
            print("not enough beats found in {:s}".format(path))
            return 0

    bpm = beats_to_bpm(beats, path)

    return {'bpm': bpm, 'beats': np.array(beats)}
    
