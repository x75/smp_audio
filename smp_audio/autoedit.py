"""smp_audio.autoedit

autoedit function
"""
import os
from collections import OrderedDict

import joblib
import numpy as np
from librosa import samples_to_frames, time_to_frames, frames_to_time

from smp_audio.common import autocount
from smp_audio.common_essentia import data_load_essentia
from smp_audio.util import args_to_dict, ns2kw, kw2ns
from smp_audio.common_librosa import compute_segments_librosa, compute_chroma_librosa, compute_beats_librosa, compute_onsets_librosa
from smp_audio.common_essentia import compute_segments_essentia
from smp_audio.segments import compute_event_merge_combined
from smp_audio.assemble_pydub import track_assemble_from_segments, track_assemble_from_segments_sequential, track_assemble_from_segments_sequential_scale

# caching joblib
from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)
# memory = Memory(None)

# def timethis(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         # start = time.time()
#         result = func(*args, **kwargs)
#         # end = time.time()
#         # print(func.__name__, end-start)
#         return result
#     return wrapper

# memory = object()
# memory.cache = timethis

# TODO: create autoedit_conf_default
autoedit_conf = {
    'default': {
        'assemble_crossfade': 10,
        'assemble_mode': 'random',
        'duration': 60.0,
        'filenames': [],
        'mode': 'autoedit',
        'numsegs': 20,
        'rootdir': './',
        'seed': 1234,
        'seglen_max': 60,
        'seglen_min': 2,
        'sorter': 'features_mt_spectral_spread_mean',
        'sr_comp': 22050,
        'verbose': False,
        'write': False
    },
}
# floats
autoedit_conf_types_float = ['assemble_crossfade', 'duration']
# ints
autoedit_conf_types_int = ['numsegs', 'seed', 'seglen_max', 'seglen_min', 'sr_comp']
autoedit_conf_default = autoedit_conf['default']

def autoedit_args_check(args):
    """autoedit_args_check

    Type check arguments
    """
    for t in autoedit_conf_types_float:
        setattr(args, t, float(getattr(args, t)))
    for t in autoedit_conf_types_int:
        setattr(args, t, int(getattr(args, t)))
    return args
        
def main_autoedit(args):
    """main_autoedit

    Complete autoedit flow

    ..todo::
    - loop over chunks of input, batch is large single chunk
    - handle returned chunk data, integrate over time
    - chunk parallel processing (entire graph) vs. chunk serial processing (entire graph)
    - nodes with memory and nodes without

    - graph class
    - populate 'func' in graph w/ cached/non-cached funcs
    - step file / stream input: deal with chunking and collecting, refuse to work on files > maxlen (configurable)
    - openl3
    """
    # convert args to dict
    # kwargs = args_to_dict(args)

    # convert arguments to locals, TODO: config file for autoedit param dict
    # sr_comp = kwargs['sr_comp'] # 22050
    # numsegs = kwargs['numsegs'] # 10
    # duration = kwargs['duration'] # 10
    # verbose = kwargs['verbose']
    # seglen_min = time_to_frames(kwargs['seglen_min'])
    # seglen_max = time_to_frames(kwargs['seglen_max'])

    args = autoedit_args_check(args)
    
    seglen_min = time_to_frames(args.seglen_min)
    seglen_max = time_to_frames(args.seglen_max)
    timebase = "frames"
    spacer = '\n    '
    
    # caching
    # compute_music_extractor_essentia_cached = memory.cache(compute_music_extractor_essentia)
    
    # computation graph g
    g = OrderedDict()

    # populate graph with functions
    g['func'] = {}
    for func in [
            compute_beats_librosa,
            compute_chroma_librosa,
            compute_onsets_librosa,
            compute_segments_essentia,
            compute_segments_librosa,
            data_load_essentia,
    ]:
        g['func'][func] = memory.cache(func)

    for func in [
            compute_event_merge_combined,
            track_assemble_from_segments,
            track_assemble_from_segments_sequential_scale,
    ]:
        g['func'][func] = func

    # layer 1: file data
    g['l1_files'] = OrderedDict()
    for filename in args.filenames:
        # replace with basename
        filename_short = filename.split('/')[-1]
        if args.verbose:
            print(('main_autoedit{1}filename_short: {0}'.format(filename_short, spacer)))
        # files[filename_short] = compute_tempo_beats(filename)
        # load data
        # y, sr = data_load_essentia_cached(filename)
        # compute beatiness on data
        g['l1_files'][filename_short] = {}
        tmp_ = g['func'][data_load_essentia](filename)
        g['l1_files'][filename_short]['data'] = tmp_[0]
        g['l1_files'][filename_short]['numsamples'] = len(tmp_[0])
        g['l1_files'][filename_short]['numframes'] = samples_to_frames(len(tmp_[0]))
        g['l1_files'][filename_short]['sr'] = tmp_[1]
        if args.verbose:
            print('main_autoedit{5}loaded {0} with shape {1}, numsamples {2}, numframes {3}, sr {4}'.format(filename_short, g['l1_files'][filename_short]['data'].shape, g['l1_files'][filename_short]['numsamples'], g['l1_files'][filename_short]['numframes'], g['l1_files'][filename_short]['sr'], spacer))

    # layer 2: compute chromagram
    g['l2_chromagram'] = {}
    for file_ in g['l1_files']:
        # file_key = '{0}-{1}'.format(file_, 'chromagram')
        g['l2_chromagram'][file_] = {}
        g['l2_chromagram'][file_]['data'] = g['func'][compute_chroma_librosa](g['l1_files'][file_]['data'], args.sr_comp)['chromagram']

    # layer 3: compute segments based on chromagram
    g['l3_segments'] = OrderedDict()
    for file_ in g['l2_chromagram']:
        # file_key = '{0}-{1}'.format(file_, 'segments')
        bounds_frames = g['func'][compute_segments_essentia](g['l2_chromagram'][file_]['data'], args.sr_comp, args.numsegs)['bounds_frames']
        # print(('    file_: {0}, bounds_frames {1}, {2}'.format(file_, len(bounds_frames), pformat(bounds_frames))))
        g['l3_segments'][file_] = {}
        g['l3_segments'][file_]['seg_sbic'] = np.clip(bounds_frames, 0, [g['l1_files'][filename_short]['numframes'] for filename_short in g['l1_files']][0]-1)

        bounds_frames = g['func'][compute_segments_librosa](g['l2_chromagram'][file_]['data'], args.sr_comp, args.numsegs)['bounds_frames']
        # print(('    file_: {0}, bounds_frames {1}, {2}'.format(file_, len(bounds_frames), pformat(bounds_frames))))
        g['l3_segments'][file_]['seg_clust_1'] = bounds_frames

        bounds_frames = g['func'][compute_segments_librosa](g['l2_chromagram'][file_]['data'], args.sr_comp, args.numsegs + 5)['bounds_frames']
        # print(('    file_: {0}, bounds_frames {1}, {2}'.format(file_, len(bounds_frames), pformat(bounds_frames))))
        g['l3_segments'][file_]['seg_clust_2'] = bounds_frames
        
    # layer 4: compute onsets
    g['l4_onsets'] = OrderedDict()
    for file_ in g['l1_files']:
        onsets = g['func'][compute_onsets_librosa](g['l1_files'][file_]['data'], args.sr_comp)
        g['l4_onsets'][file_] = onsets

    # layer 5: compute beats based on onsets
    g['l5_beats'] = OrderedDict()
    for file_ in g['l4_onsets']:
        g['l5_beats'][file_] = {}
        for start_bpm in [60, 90, 120]:
            beats = g['func'][compute_beats_librosa](g['l4_onsets'][file_]['onsets_env'], g['l4_onsets'][file_]['onsets_frames'], start_bpm, args.sr_comp)
            # print('beats type = {0}'.format(type(beats['beats'])))
            # beats['beats'] = beats['beats'][np.logical_not(np.isnan(beats['beats']))]
            # beats = beats[~np.isnan(beats)]
            # print('    file_: {0}, bounds_frames {1}, {2}'.format(file_, len(bounds_frames), pformat(bounds_frames)))
            g['l5_beats'][file_]['beats_{0}'.format(start_bpm)] = beats['beats']
            g['l5_beats'][file_]['beats_{0}_16'.format(start_bpm)] = beats['beats'][::16]

    # layer 6: compute final segments from merging segments with beats
    g['l6_merge'] = OrderedDict()
    g['l6_merge']['files'] = []
    for file_ in g['l1_files']:
        # get basedir from filename
        dirname = os.path.dirname(filename)
        # return realpath absolute path
        # dirname = os.path.dirname(os.path.realpath(filename))
        if dirname == '':
            dirname = '.'
            
        if args.verbose:
            print(f'main_autoedit dirname {dirname}')
            print(f'main_autoedit{spacer}l6_merge file_ {file_}, dirname {dirname}, filename {filename}')
        beats_keys = ['beats_60', 'beats_90', 'beats_120'] + ['beats_60_16', 'beats_90_16', 'beats_120_16']
        # beats = [g['l5_beats'][file_][beat_type] for beat_type in beats_keys for file_ in g['l1_files']]
        beats = [g['l5_beats'][file_][beat_type] for beat_type in beats_keys]
        # segs = [g['l3_segments'][file_][seg_type_] for seg_type_ in ['seg_sbic', 'seg_clust_1', 'seg_clust_2'] for file_ in g['l1_files']]
        segs = [g['l3_segments'][file_][seg_type_] for seg_type_ in ['seg_sbic', 'seg_clust_1', 'seg_clust_2']]
        numframes = g['l1_files'][file_]['numframes']
        # compute
        if args.verbose:
            print(f'main_autoedit{spacer}l6_merge dirname {dirname}, filename {filename}')
        files = g['func'][compute_event_merge_combined](
            filename_48=dirname + '/' + file_,
            beats=beats,
            segs=segs,
            numframes=numframes,
            numsegs=args.numsegs,
            verbose=args.verbose,
        )
        
        g['l6_merge']['files'].extend(files['files'])
        if args.verbose:
            print('main_autoedit{2}l6_merge {0}, {1}'.format(file_, g['l6_merge']['files'], spacer))

    # layer 7: compute assembled song from segments and duration
    g['l7_assemble'] = OrderedDict()
    # compute duration
    g['l6_merge']['duration'] = args.duration
    # output filename
    g['l6_merge']['filename_export'] = args.filename_export
    # crossfade argument
    g['l6_merge']['assemble_crossfade'] = args.assemble_crossfade

    if args.assemble_mode == 'random':
        g['l7_assemble']['outfile'] = g['func'][track_assemble_from_segments](**(g['l6_merge']))
    elif args.assemble_mode == 'sequential':
        g['l7_assemble']['outfile'] = g['func'][track_assemble_from_segments_sequential_scale](**(g['l6_merge']))


    export_filename = g['l7_assemble']['outfile']['filename_export']
    export_duration = g['l7_assemble']['outfile']['final_duration']
    export_segs = g['l7_assemble']['outfile']['seg_s']
    export_numsegs = len(g['l7_assemble']['outfile']['seg_s'])
        
    # print((pformat(g)))
    # joblib.dump(g, './g.pkl')
    export_filename_graph = f'{export_filename[:-4]}.pkl'
    print(f'main_autoedit{spacer}exporting graph to {export_filename_graph}')
    joblib.dump(g, export_filename_graph)
    
    # # plot dictionary g as graph
    # autoedit_graph_from_dict(g=g, plot=False)
    ret = {
        'filename_': export_filename,
        'length': export_duration,
        'numsegs': export_numsegs,
        'filename_graph': export_filename_graph
    }
    # # yeah nice, should be obsolete
    # ret.update(g['l7_assemble']['outfile'])
    
    return ret
