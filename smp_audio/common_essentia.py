import numpy as np

# from essentia.standard import *
from essentia.standard import MonoLoader, RhythmExtractor2013, Danceability
from essentia.standard import MusicExtractor

from librosa import frames_to_time, frames_to_samples

from logging import DEBUG as LOGLEVEL
# from smp_base.common import get_module_logger
from slurp.common import get_module_logger
l = get_module_logger('slurp.common_essentia', LOGLEVEL)

def data_load_essentia(filename, duration=None, offset=0.0):
    """data_load_essentia

    Data load with essentia loader

    Args:
    - filename(str): audio filename
    - duration(float): fractional seconds of audio to load
    - offset(float): fractional seconds of reading offset

    Returns:
    - y(np.ndarray): audio data numpy array
    - sr(int): sample rate
    """
    assert type(filename) is str and filename is not None and filename != '', 'filename argument {0} / {1} is invalid'.format(filename, type(filename))
    # assert type(duration) in [float, int], 'duration argument {0} / {1} is invalid'.format(duration, type(duration))
    
    # if args is not None:
    #     # args = Namespace()
    #     # filename = file
    #     # args.duration = 10.0
    #     filename = data_get_filename(args)
    #     duration = args.duration

    l.debug('Loading ({2}) {0} s of audio file {1}'.format(duration, filename, 'essentia'))

    sr = 22050
    y = MonoLoader(filename=filename, sampleRate=sr).compute()
    l.debug('Loaded ({2}) audio file with {0} samples at rate {1}'.format(type(y), sr, 'essentia'))
    return y, sr

def compute_tempo_beats_essentia(y, sr=None):
    """compute_tempo_beats_essentia

    Compute beatiness features using essentia's RhythmExtractor2013
    and Danceability
    """
    # l.debug('compute_tempo_beats_essentia y = {0}'.format(y))
    audio = y

    # Compute beat positions and BPM
    rhythm_extractor = RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

    # print("    BPM:", bpm)
    # print("    beat positions (sec.):", len(beats))
    # print("    beat estimation confidence:", beats_confidence)

    danceability_extractor = Danceability()
    danceability_full = danceability_extractor(audio)
    danceability = danceability_full[0]
    # print("    danceability:", danceability[0])

    return {'bpm': bpm, 'beats': beats, 'beats_confidence': beats_confidence, 'beats_intervals': beats_intervals, 'danceability': danceability}

def compute_music_extractor_essentia(y, sr=None):
    """compute_music_extractor_essentia

    Compute features using essentia's MusicExtractor
    """
    # l.debug('compute_tempo_beats_essentia y = {0}'.format(y))
    filename = y
    features, features_frames = MusicExtractor(
        lowlevelStats=['mean', 'stdev'],
        rhythmStats=['mean', 'stdev'],
        tonalStats=['mean', 'stdev'])(filename)

    print(type(features))
    # See all feature names in the pool in a sorted order
    print(sorted(features.descriptorNames()))
    # MusicExtractor

    result = dict([(k, features[k]) for k in sorted(features.descriptorNames())])
    
    # # Compute beat positions and BPM
    # rhythm_extractor = RhythmExtractor2013(method="multifeature")
    # bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

    # # print("    BPM:", bpm)
    # # print("    beat positions (sec.):", len(beats))
    # # print("    beat estimation confidence:", beats_confidence)

    # danceability_extractor = Danceability()
    # danceability_full = danceability_extractor(audio)
    # danceability = danceability_full[0]
    # # print("    danceability:", danceability[0])

    # return {'bpm': bpm, 'beats': beats, 'beats_confidence': beats_confidence, 'beats_intervals': beats_intervals, 'danceability': danceability}
    return result # {'y': y, }

def compute_segments_essentia(Y, sr, numparts):
    """compute part segmentation estimate using essentia's information
    criterion approach

    Args:
    - Y(np.ndarray): input feature-gram (e.g. chroma)
    - sr(int): sample rate
    - numparts(int): desired number of segments

    Returns:
    - bounds(list): boundaries in frame index space
    - bound_times(np.ndarray): boundaries in time index space
    - bound_samples(np.ndarray): boundaries in sample index space
    """    
    from essentia.standard import SBic
    print('Y', type(Y))
    # Y = np.atleast_2d(Y)
    sbic = SBic(cpw=0.5, inc1=60, inc2=20, minLength=50, size1=200, size2=300)
    # myprint('chroma.T.tolist()', chroma.T.tolist())
    segs = sbic(Y.tolist())
    # segs = sbic(Y)
    print('segments sbic', frames_to_time(segs, sr=sr))
    
    # segments['sbic'] = {}
    # myprint('bounds', bd_)
    bounds_frames = segs.astype(int)
    bounds_times = frames_to_time(segs, sr=sr)
    bounds_samples = frames_to_samples(segs)
    
    # segments['sbic']['bounds_hist'] = np.histogram(np.diff(segments['sbic']['bound_times']), bins=20)
    # segments['sbic']['bounds_mode'] = mode(np.diff(segments[numparts]['bound_times']))
    # return bounds, bound_times, bound_samples
    
    return {'bounds_frames': bounds_frames, 'bounds_times': bounds_times, 'bounds_samples': bounds_samples}
