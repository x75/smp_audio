from argparse import Namespace

# caching librosa
import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'

import librosa
# import madmom
import numpy as np

from smp_audio.common import myprint

################################################################################
# data loading
def data_get_filename(args):
    if isinstance(args, Namespace):
        if args.file is None:
            filename = librosa.util.example_audio_file()
        else:
            # filename = '/home/x75/Downloads/BEAT1R-mono.wav'
            filename = args.file
        return filename
    else:
        return args[0]

def data_load_librosa(filename, duration=None, offset=0.0):
    """data_load_librosa

    Data load with librosa loader

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

    myprint('Loading %ss of audio file %s' % (duration, filename))
    
    y, sr = librosa.load(filename, offset=offset, duration=duration)
    myprint('Loaded audio file with %s samples at rate %d' % (y.shape, sr))
    return y, sr

def data_stream_librosa(**kwargs):
    # filename = librosa.util.example_audio_file()
    if 'filename' in kwargs:
        filename = kwargs['filename']
    else:
        filename = '/home/src/QK/data/sound-arglaaa-2018-10-25/24.wav'
    
    sr = librosa.get_samplerate(filename)
    stream = librosa.stream(filename,
                            block_length=1,
                            frame_length=512,
                            hop_length=512,
                            mono=True
    )
    
    return tuple((stream, sr))

def data_stream_get_librosa(src, **kwargs):
    print(src)
    for y_block in src:
        print(src, y_block.shape)
        # D_block = librosa.stft(y_block, center=False)

        yield tuple((y_block, len(y_block)))

################################################################################
# compute feature transforms
def compute_chroma_librosa(y, sr):
    """compute chromagram using librosa

    Args:
    - y(np.ndarray): input audio array
    - sr(int): sample rate

    Returns:
    - chroma(np.ndarray): chromagram array
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    myprint('computing chroma w/ {0}'.format(chroma.shape))
    return {'chromagram': chroma}

def compute_tempogram_librosa(y, sr, o_env, hop_length=512):
    """compute tempogram using librosa

    Args:
    - y(np.ndarray): input audio array
    - sr(int): sample rate
    - o_env(np.ndarray): onset strength envelope from onset_strength
    - hop_length(int): hop length in samples

    Returns:
    - tempogram(np.ndarray): tempogram array
    """
    tempogram = librosa.feature.tempogram(onset_envelope=o_env, sr=sr,
                                          hop_length=hop_length)
    return {'tempogram': tempogram}

def compute_onsets_librosa(y, sr):
    """compute onsets using librosa

    Args:
    - y(np.ndarray): input audio array
    - sr(int): sample rate

    Returns:
    - o_env(np.ndarray): onset strength envelope
    - o_frames(np.ndarray): onset times in frame index space
    - o_times_ref(np.ndarray): onset reference times for every frame
    """
    myprint('Computing onset strength envelope')
    o_env = librosa.onset.onset_strength(y, sr=sr)
    o_times_ref = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    o_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    myprint('Computing onset strength envelope o_env = %s' % (o_env.shape, ))
    return {'onsets_env': o_env, 'onsets_frames': o_frames, 'onsets_times_ref': o_times_ref}

def compute_beats_librosa(onset_env, onset_frames, start_bpm, sr):
    """compute beat estimate using librosa

    Args:
    - onset_env(np.ndarray): onset strength envelope
    - onset_frames(np.ndarray): onsets in frame index space
    - start_bpm(int): starting bpm value for search
    - sr(int): sample rate

    Returns:
    - tempo(float): global tempo estimate scalar
    - dtempo(np.ndarray): framewise tempo estimate
    - beats(np.ndarray): beat event occurrence in frame index space
    """    
    myprint('compute_beats_librosa: computing beat_track with start_bpm = {0}'.format(start_bpm))
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=float(start_bpm))
    # print('beats pre nan = {0}'.format(beats))
    beats = beats[np.logical_not(np.isnan(beats))]
    # print('beats post nan = {0}'.format(beats))
    dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None, start_bpm=float(start_bpm))
    return {'tempo': tempo, 'dtempo': dtempo, 'beats': beats}
    
def compute_beats_madmon(onset_env, onset_frames, start_bpm, sr, filename):
    """compute beat estimate using madmom

    Args:
    - onset_env(np.ndarray): onset strength envelope
    - onset_frames(np.ndarray): onsets in frame index space
    - start_bpm(int): starting bpm value for search
    - sr(int): sample rate

    Returns:
    - tempo(None): not implemented
    - dtempo(None): not implemented
    - beats(np.ndarray): beat event occurrence in frame index space
    """    
    myprint('Computing beat_track mm')
    mm_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    mm_act = madmom.features.beats.RNNBeatProcessor()(filename)
    
    mm_beat_times = mm_proc(mm_act)
    myprint('mm_beat_times', mm_beat_times)
    # return None, None, librosa.time_to_frames(mm_beat_times, sr)
    return {'tempo': None, 'dtempo': None, 'beats': librosa.time_to_frames(mm_beat_times, sr)}

def compute_beats_mix(y, sr, beats):
    clicks = librosa.clicks(librosa.frames_to_time(beats), sr=sr, length=len(y))
    mix = y + clicks
    librosa.output.write_wav('mix.wav', mix, sr)
    # ipd.Audio(x + clicks, rate=sr)

def compute_segments_librosa(Y, sr, numparts):
    """compute part segmentation estimate using librosa

    Args:
    - Y(np.ndarray): input feature-gram (e.g. chroma)
    - sr(int): sample rate
    - numparts(int): desired number of segments

    Returns:
    - bounds(list): boundaries in frame index space
    - bound_times(np.ndarray): boundaries in time index space
    - bound_samples(np.ndarray): boundaries in sample index space
    """    
    myprint('Computing parts segmentation')
    bounds_frames = librosa.segment.agglomerative(Y, numparts)
    bounds_times = librosa.frames_to_time(bounds_frames, sr=sr)
    bounds_samples = librosa.frames_to_samples(bounds_frames, hop_length=512, n_fft=2048)
    myprint('bounds_samples = %s / %s' % (bounds_samples.shape, bounds_samples))
    return {'bounds_frames': bounds_frames, 'bounds_times': bounds_times, 'bounds_samples': bounds_samples}

def myplot_specshow_librosa(ax, y):
    D = np.abs(librosa.stft(y))
    librosa.display.specshow(
        librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log',
        ax=ax
    )
    ax.set_title('Power spectrogram')

def myplot_onsets(ax, times, o_env, onset_frames):
    ax.plot(times, o_env, label='Onset strength')
    ax.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
              linestyle='--', label='Onsets')

def myplot_beats(ax, beattimes, ylow, yhigh, alpha, color, linestyle, label):
    ax.vlines(beattimes, ylow, yhigh, alpha=alpha, color=color, linestyle=linestyle, label=label)

def myplot_tempo(ax, times, tempo):
    # myprint('myplot_tempo times = %s, tempo = %s' % (times, tempo))
    ax.plot(times, tempo, alpha=0.5, color='b', linestyle='none', marker='o', label='Tempo')

def myplot_chroma(ax, chroma):
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    
def myplot_segments(ax, chroma, bound_times, color='g'):
    ax.vlines(bound_times, 0, chroma.shape[0], color=color, linestyle='-',
            linewidth=2, alpha=0.9, label='Segment boundaries')    

def myplot_segments_hist(ax, bound_hist, idx=None, color='k'):
    # myprint('segment bound hist', bound_hist)
    if idx is not None:
        ax.bar([idx], bound_hist.mode, alpha=0.4, color=color)
    else:
        ax.bar(bound_hist[1][:-1], bound_hist[0], alpha=0.4, color=color)

