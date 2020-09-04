"""sort audio files by some scalar feature: beatiness, density, etc

algorithm prototype
 1. input: list of files
 2. slice input files
 3. flat list of segments
 4. measure segments
 5. flat list of segment measures
 6. sort list by measure
 7. create envelope
 8. complete output file by concatenating segments selected by their index by sampling the envelope

main_autoedit:
 - fix graph for multiple input files
 - fix graph for parameter variations
 - fix graph for dynamic modification

 - 1.1. perform pre-classification for model choice later on
 - rename: autosnd, automusic
"""
import argparse, time, pickle, sys, os
from collections import OrderedDict
from pprint import pformat
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import wave
import audioread

from librosa import samples_to_frames, time_to_frames, frames_to_time

from smp_audio.common_essentia import data_load_essentia
from smp_audio.common_essentia import compute_segments_essentia
from smp_audio.common_essentia import compute_tempo_beats_essentia, compute_music_extractor_essentia
from smp_audio.common_librosa import data_load_librosa
from smp_audio.common_librosa import compute_segments_librosa, compute_chroma_librosa, compute_beats_librosa, compute_onsets_librosa
# aubio hardcoded? fix flexible lib loading
from smp_audio.common_aubio import data_load_aubio
from smp_audio.common_aubio import compute_onsets_aubio
from smp_audio.common_aubio import compute_tempo_beats_aubio

# stream input
from smp_audio.common_librosa import data_stream_librosa, data_stream_get_librosa
from smp_audio.common_aubio import data_stream_aubio, data_stream_get_aubio

from smp_audio.assemble_pydub import track_assemble_from_segments, track_assemble_from_segments_sequential, track_assemble_from_segments_sequential_scale
from smp_audio.graphs import graph_walk_collection_flat, graph_walk_collection
from smp_audio.graphs import cb_graph_walk_build_graph
from smp_audio.util import args_to_dict
from smp_audio.segments import compute_event_merge_combined

# from audio_segments_split import main_audio_segments_split
from audio_features_paa import compute_features_paa

# caching joblib
from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

# backend = 'librosa'
backend = 'aubio'

data_streams = {
    'librosa': {
        'open': data_stream_librosa,
        'read': data_stream_get_librosa,
    },
    'aubio': {
        'open': data_stream_aubio,
        'read': data_stream_get_aubio,
    }
}

def get_filefp(filepath):
    duration, fp_encoded = acoustid.fingerprint_file(filepath)
    fingerprint, version = chromaprint.decode_fingerprint(fp_encoded)
    return fingerprint

def get_filelength(filepath):
    # f = sf.SoundFile(filepath)
    # filelength = len(f) / f.samplerate

    with audioread.audio_open(filepath) as f:
        print(f.channels, f.samplerate, f.duration)
        filelength = f.duration
    return filelength

def get_fileinfo(filepath):
    # f = sf.SoundFile(filepath)
    # filelength = len(f) / f.samplerate

    with audioread.audio_open(filepath) as f:
        print(f.channels, f.samplerate, f.duration)
        # filelength = f.duration
        return {'channels': f.channels, 'samplerate': f.samplerate, 'duration': f.duration}
    return {}
    
def get_filehash(filepath):
    # m = hashlib.md5()
    m = hashlib.sha256()
    with open(filepath, 'rb') as f: 
        for byte_block in iter(lambda: f.read(4096),b""):
            m.update(byte_block)
        # for chunk in iter(f.read(1024)):
        #     m.update(chunk)
        return m.hexdigest()
    return None

def main_timing_read_stream(filename, src, sr, framesize, args):
    # global data_streams
    print(f'main_timing_read_stream args = {args}')

    # 1 open file, stream, data-stream/live-src
    # 2 scan the frames and print timing info

    # 3 handle multiple filenames
    # 4 handle multiple samplerates

    X = []
    Y = []
    start = time.time()
    # while src read next
    # for blk_i, blk_y in enumerate(src):
    for blk_i, blk_y_tup in enumerate(data_streams[backend]['read'](src)):
        blk_y = blk_y_tup[0]
        blk_y_len = blk_y_tup[1]
        # src_len_blks = src.duration/blk_y_len
        # print(f'i {blk_i}/{src_len_blks}, y {blk_y.shape}')
        # print(f'i {blk_i}, y {blk_y.shape}')

        if len(blk_y) < framesize: continue
        # D_block = librosa.stft(blk_y, n_fft=framesize, hop_length=framesize, center=False)
        # print(f'D_block {D_block.shape}')
        # D.append(D_block[:,0])
        y = blk_y
        # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        # rmse = librosa.feature.rms(y=y, frame_length=framesize, hop_length=framesize, center=False)
        # spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        # spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        # rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        # zcr = librosa.feature.zero_crossing_rate(y, frame_length=framesize, hop_length=framesize, center=False)
        # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        # print(f'    timing_read_stream y {y.shape}, {chroma_stft.shape}')
    
    end = time.time()
    time_taken = end - start
    print(('\nThe function took {:.2f} s to compute.'.format(time_taken)))
    return {'mode': args.mode, 'filename': filename, 'time': time_taken}
    
def main_autoedit_stream(filename, src, sr, framesize, args):
    """autoedit_stream is an extended aubio_cut

    it generates several different segmentations and finds overlapping
    boundaries.

    issues:
    - timing reference: samples, frames, seconds
    """
    print(f'main_autoedit_stream args = {args}\n    length = {args.srclength/framesize}')
    start = time.time()
    # while src read next
    # for blk_i, blk_y in enumerate(src):
    for blk_i, blk_y_tup in enumerate(data_streams[backend]['read'](src)):
        blk_y = blk_y_tup[0]
        blk_y_len = blk_y_tup[1]
        # src_len_blks = src.duration/blk_y_len
        # print(f'i {blk_i}/{src_len_blks}, y {blk_y.shape}')
        print(f'i {blk_i}, y {blk_y.shape}')

        if len(blk_y) < framesize: continue
        y = blk_y
        
        # insert autoedit_stream computation graph
        # 1 get features for m in method
        chromagram = compute_chroma_librosa(blk_y, sr, hop_length=framesize)['chromagram']
        # 2 get segments for m in method
        # 2.1 m = essentia, broken with essentia frame processing model
        # segments1 = compute_segments_essentia(blk_y, sr, 10)
        # 2.2 m = librosa
        segments2 = compute_segments_librosa(blk_y, sr, 10)
        
        # 3 get onsets for m in method
        # 4 get beat for m in method
        # 5 probabilistic overlap combination
        # 6 assemble
    
    end = time.time()
    time_taken = end - start
    print(('\nThe function took {:.2f} s to compute.'.format(time_taken)))
    return {'mode': args.mode, 'filename': filename, 'time': time_taken}    

def get_src(filename, args):
    # filename = args.filenames[0][filename_i]
    print(f'filename = {filename}')
    # load audio data
    framesize = 1024

    srcfile = wave.open(filename, mode='rb')
    srcparams = srcfile.getparams()
    length = srcparams[3] # pos 3 is nframes

    src, sr = data_streams[backend]['open'](
        filename=filename,
        frame_length=framesize,
        hop_length=framesize,
        num_channels=1,
    )
    # print(f'sr={sr}')
    return src, sr, framesize, filename, length

def main_paa_feature_extractor(args):
    # convert args to dict
    kwargs = args_to_dict(args)
    print('main_paa_feature_extractor enter kwargs {0}'.format(kwargs))

def main_music_extractor(args):
    # convert args to dict
    kwargs = args_to_dict(args)

    # caching
    # compute_tempo_beats_cached = memory.cache(compute_tempo_beats)
    # data_load_essentia_cached = memory.cache(data_load_essentia)
    compute_music_extractor_essentia_cached = memory.cache(compute_music_extractor_essentia)
    
    files = {}
    start = time.time()
    for filename in kwargs['filenames']:
        filename_short = filename.split('/')[-1]
        print(('file: {0}'.format(filename_short)))
        # files[filename_short] = compute_tempo_beats(filename)
        # load data
        # y, sr = data_load_essentia_cached(filename)
        # compute beatiness on data
        files[filename_short] = compute_music_extractor_essentia_cached(filename)
    end = time.time()

    print(('\nThe function took {:.2f} s to compute.'.format(end - start)))
    # print(pformat(files))
    for k, v in list(files.items()):
        # del v['beats_intervals']
        # del v['beats_intervals']
        print(('file {0}\n{1}'.format(k, pformat(v))))

    # save results file to pickle
    joblib.dump(files, '{0}-files-dict-music-extractor-{1}.pkl'.format(sys.argv[0], int(time.time())))

def main_beatiness(args):
    # convert args to dict
    kwargs = args_to_dict(args)

    # caching
    # compute_tempo_beats_cached = memory.cache(compute_tempo_beats)
    # data_load_essentia_cached = memory.cache(data_load_essentia)
    data_load_essentia_cached = memory.cache(data_load_librosa)
    compute_tempo_beats_cached = memory.cache(compute_tempo_beats_essentia)
    
    files = {}
    start = time.time()
    for filename in kwargs['filenames']:
        filename_short = filename.split('/')[-1]
        print(('file: {0}'.format(filename_short)))
        # files[filename_short] = compute_tempo_beats(filename)
        # load data
        y, sr = data_load_essentia_cached(filename)
        # compute beatiness on data
        files[filename_short] = compute_tempo_beats_cached(y)
        del files[filename_short]['beats_intervals']
        del files[filename_short]['beats']
    end = time.time()

    print(('\nThe function took {:.2f} s to compute.'.format(end - start)))
    # print(pformat(files))
    for k, v in list(files.items()):
        # del v['beats_intervals']
        # del v['beats_intervals']
        print(('file {0}\n{1}'.format(k, pformat(v))))

    # save results file to pickle
    joblib.dump(files, 'audio-beatiness-essentia-files-dict-beatiness.pkl')

    # for f in files:
    #     files[f]['danceability'] = files[f]['danceability'][0]
        
    files_ = {}
    for i, f in enumerate(files):              
        files[f]['name'] = f                                  
        files_[i] = files[f]

    # # convert results to dataframe for sorting
    # files_df = pd.DataFrame.from_dict(files_).T

    # # load 
    # import joblib
    # files = joblib.load('../smp_music/scripts/audio-beatiness-essentia-files-dict.pkl')
    # files
    # len(files)
    # filed_df = pd.DataFrame.from_dict(files)
    # filed_df
    # filed_df.columns
    # pd.DataFrame.from_items
    # pd.DataFrame.from_items?
    # files
    # list(files)
    # files_ = {}
    # for i, f in enumerate(files):
    #     files[f]['name'] = f
    #     files_[i] = files[f]
    #     files_
    #     list(files_)
    #     files_df = pd.DataFrame.from_dict(files_)
    #     files_df
    #     files_df.co
    #     files_df.columns
    #     files_df.T
    #     files_df.T.columns
    #     files_df = pd.DataFrame.from_dict(files_).T
    #     files_df.columns
    #     files_df
    #     files
        
    # for f in files:
    #     files[f]['danceability'] = files[f]['danceability'][0]
        
    # files_ = {}
    # for i, f in enumerate(files):
    #     files[f]['name'] = f
    #     files_[i] = files[f]
        
    # files_
    # files_df = pd.DataFrame.from_dict(files_).T
    # files_df
    # files_df.sort_index('name')
    # files_df.sort_values('name')
    # files_df.sort_values('danceability')
    # files_df.sort_values('beats_confidence')
    # files_df.sort_values('danceability')
    # files_df.sort_values('danceability')
    # files_df.to_dict()
    # files_df.T.to_dict()
    # files_df
    # files_df.to_json
    # files_df.to_json()
    # pprint(files_df.to_json())
    # from pprint import pformat
    # print(files_df.to_json())
    # print(pformat(files_df.to_json()))
    # print(pformat(files_df.T.to_json()))
    # print(pformat(files_df.T.to_json()))
    # files_df.to_csv()
    # files_df.to_csv?
    # files_df
    # files_df.sort_values('beats_confidence').to_csv('fm_singles_beats_confidence.csv')
    # files_df.sort_values('danceability').to_csv('fm_singles_danceability.csv')
    # pwd
    # files_df.sort_values('beats_confidence')
    # files_df.sort_values('beats_confidence').name
    # files_df.sort_values('beats_confidence').name.to_csv()
    # files_df.to_string
    # files_df.sort_values('beats_confidence').name.to_string()
    # files_df.sort_values('beats_confidence').name.to_string('fm_singles_beats_confidence.txt')
    # files_df.sort_values('danceability').name.to_string('fm_singles_danceability.txt')
    # files_df.sort_values('danceability').name.to_string('fm_singles_danceability.txt')
    
    # plot_tempogram_and_tempo(data)

    # plt.show()

def main_automix(args):
    """main_automix

    Perform complete automix flow with the following schema:
    
    1. input list of audio files / text file containing list of audio files
    2. loop over files
    2.1. compute bag of measures for each file: beatiness, extractor essentia, features paa

    2.2. sort files by selected feature args.sort_feature
    2.3. assemble output wav from concatenating input files pydub

    2.4. TODO: optional: local measures
    2.4. TODO: optional: complexity / information measures smp/sequence
    """
    # convert args to dict
    kwargs = args_to_dict(args)

    print('main_automix: kwargs {0}'.format(pformat(kwargs)))

    # flow graph g
    g = OrderedDict()

    # cached functions
    g['func'] = {}
    for func in [
            compute_beats_librosa,
            compute_chroma_librosa,
            compute_features_paa,
            compute_music_extractor_essentia,
            compute_onsets_librosa,
            compute_segments_essentia,
            compute_segments_librosa,
            compute_tempo_beats_essentia,
            data_load_essentia,
    ]:
        g['func'][func] = memory.cache(func)

    # uncached functions
    for func in [
            compute_event_merge_combined,
            track_assemble_from_segments,
    ]:
        g['func'][func] = func


    # input type: text file, list of files
    if len(kwargs['filenames']) == 1 and kwargs['filenames'][0].endswith('.txt'):
        filenames = [_.rstrip() for _ in open(kwargs['filenames'][0], 'r').readlines()]
        # print('filenames {0}'.format(pformat(filenames)))
        print('filenames {0}'.format(filenames))
    else:
        filenames = kwargs['filenames']

    # layer 1: file/chunk data
    g['l1_files'] = OrderedDict()
    for i, filename in enumerate(filenames):
        # print('filename {0}: {1}'.format(i, filename))
        
        filename_short = filename.split('/')[-1]
        print(('file: {0}'.format(filename_short)))
        # load data
        # y, sr = g['func'][data_load_essentia](filename)
        g['l1_files'][filename_short] = {}
        tmp_ = g['func'][data_load_essentia](filename)
        g['l1_files'][filename_short]['path'] = filename
        g['l1_files'][filename_short]['data'] = tmp_[0]
        g['l1_files'][filename_short]['numframes'] = samples_to_frames(len(tmp_[0]))
        g['l1_files'][filename_short]['sr'] = tmp_[1]

    # layer 2: beatiness, compute beatiness on data
    # g['l2_beatiness'] = {}
    for file_ in g['l1_files']:
        # file_key = '{0}-{1}'.format(file_, 'beatiness')
        # g['l2_beatiness'][file_] = {}
        tmp_ = g['func'][compute_tempo_beats_essentia](g['l1_files'][file_]['data'])
        # g['l2_beatiness'][file_] = tmp_
        # g['l1_files'][file_]['beatiness'] = tmp_
        g['l1_files'][file_].update(dict([('beatiness' + _, tmp_[_]) for _ in tmp_]))
    
    # layer 3: extractor
    # g['l3_extractor'] = {}
    for file_ in g['l1_files']:
        print('l3_extractor on {0}'.format(file_))
        # file_key = '{0}-{1}'.format(file_, 'extractor')
        # g['l2_extractor'][file_] = {}
        tmp_ = g['func'][compute_music_extractor_essentia](g['l1_files'][file_]['path'])
        # g['l3_extractor'][file_] = tmp_
        # g['l1_files'][file_]['extractor'] = tmp_
        g['l1_files'][file_].update(dict([('extractor_' + _, tmp_[_]) for _ in tmp_]))
    
    # layer 4: paa features
    # g['l4_paa_features'] = {}
    for file_ in g['l1_files']:
        # file_key = '{0}-{1}'.format(file_, 'extractor')
        # g['l4_paa_features'][file_] = {}
        tmp_ = g['func'][compute_features_paa](g['l1_files'][file_]['path'])
        # g['l4_paa_features'][file_]['features_st'] = dict(zip(tmp_[1], tmp_[0]))
        # g['l4_paa_features'][file_]['features_mt'] = dict(zip(tmp_[1], tmp_[2]))
        g['l1_files'][file_].update(dict(zip(['features_st_' + _ for _ in tmp_[1]], [_.mean() for _ in tmp_[0]])))
        g['l1_files'][file_].update(dict(zip(['features_mt_' + _ for _ in tmp_[1]], [_.mean() for _ in tmp_[2]])))
        # g['l1_files'][file_]['features_mt'] = dict(zip(tmp_[1], tmp_[2]))

    # layer 5: 

    pickle.dump(g, open('g.pkl', 'wb'))
    
    # print('files {0}'.format(pformat(files)))
    # plot dictionary g as graph
    autoedit_graph_from_dict(g=g, plot=False)

    l1_files_df = pd.DataFrame.from_dict(g['l1_files']).T

    # sort_key = 'features_mt_energy_entropy_mean'
    # sort_key = 'features_mt_energy_mean'
    # sort_key = 'features_mt_spectral_centroid_mean'
    # sort_key = 'features_mt_spectral_entropy_mean'
    # sort_key = 'features_mt_spectral_flux_mean'
    # sort_key = 'features_mt_spectral_rolloff_mean'
    # sort_key = 'features_mt_spectral_spread_mean'
    # sort_key = 'features_mt_zcr_mean'
    sort_key = kwargs['sorter']
    
    print('Sorting l1_files by {0}'.format(l1_files_df.sort_values(sort_key, ascending=False).path.to_string()))
    l1_files_df.sort_values(sort_key, ascending=False).path.to_csv('sendspaace-assembled-{0}-{1}.{2}'.format(3, sort_key, 'csv'))

    if args.write:
        track_assemble_from_segments_sequential(files=list(l1_files_df.sort_values(sort_key, ascending=False).path),
                                                output_filename='sendspaace-assembled-{0}-{1}.{2}'.format(3, sort_key, 'wav'),
                                                duration=None)
    
def autoedit_graph_from_dict(**kwargs):
    g = kwargs['g']

    # walk flat
    # tmp_ = list(graph_walk_collection_flat(g))
    # print('graph_walk_collection_flat', pformat(tmp_))

    # walk structured with graph construction callback
    G = nx.MultiDiGraph()
    print(('Walking g'))
    graph_walk_collection(g, level=0, cb=cb_graph_walk_build_graph, G=G, P='G')
    print('G.nodes', list(G.nodes))
    print('G.edges', list(G.edges))

    print('Pickling G', type(G))
    nx.readwrite.write_gpickle(G, './G.pkl')

    # not 'plot' in kwargs or 
    if not kwargs['plot']:
        return
    
    print('Plotting G', type(G))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    try:
        G_pos = nx.nx_pydot.pydot_layout(G, prog='dot')
    except Exception as e:
        print('pydot layout failed with {0}'.format(e))
        # G_pos = nx.layout.random_layout(G)
        # G_pos = nx.layout.fruchterman_reingold_layout(G)
    nx.draw_networkx_nodes(G, pos=G_pos, ax=ax)
    nx.draw_networkx_labels(G, pos=G_pos, ax=ax)
    nx.draw_networkx_edges(G, pos=G_pos, ax=ax)
    # plt.gca().set_aspect(1)
    ax.set_title('autoedit_graph_from_dict')
    
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
    kwargs = args_to_dict(args)

    # globals
    sr_comp = kwargs['sr_comp'] # 22050
    numsegs = kwargs['numsegs'] # 10
    seglen_min = time_to_frames(kwargs['seglen_min'])
    seglen_max = time_to_frames(kwargs['seglen_max'])
    duration = kwargs['duration'] # 10
    timebase = "frames"
    
    # caching
    # compute_music_extractor_essentia_cached = memory.cache(compute_music_extractor_essentia)

    g = OrderedDict()

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
    for filename in kwargs['filenames']:
        filename_short = filename.split('/')[-1]
        print(('    filename_short: {0}'.format(filename_short)))
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
        print('    loaded {0} with shape {1}, numsamples {2}, numframes {3}, sr {4}'.format(filename_short, g['l1_files'][filename_short]['data'].shape, g['l1_files'][filename_short]['numsamples'], g['l1_files'][filename_short]['numframes'], g['l1_files'][filename_short]['sr']))

    # layer 2: compute chromagram
    g['l2_chromagram'] = {}
    for file_ in g['l1_files']:
        # file_key = '{0}-{1}'.format(file_, 'chromagram')
        g['l2_chromagram'][file_] = {}
        g['l2_chromagram'][file_]['data'] = g['func'][compute_chroma_librosa](g['l1_files'][file_]['data'], sr_comp)['chromagram']

    # layer 3: compute segments based on chromagram
    g['l3_segments'] = OrderedDict()
    for file_ in g['l2_chromagram']:
        # file_key = '{0}-{1}'.format(file_, 'segments')
        bounds_frames = g['func'][compute_segments_essentia](g['l2_chromagram'][file_]['data'], sr_comp, numsegs)['bounds_frames']
        # print(('    file_: {0}, bounds_frames {1}, {2}'.format(file_, len(bounds_frames), pformat(bounds_frames))))
        g['l3_segments'][file_] = {}
        g['l3_segments'][file_]['seg_sbic'] = np.clip(bounds_frames, 0, [g['l1_files'][filename_short]['numframes'] for filename_short in g['l1_files']][0]-1)

        bounds_frames = g['func'][compute_segments_librosa](g['l2_chromagram'][file_]['data'], sr_comp, numsegs)['bounds_frames']
        # print(('    file_: {0}, bounds_frames {1}, {2}'.format(file_, len(bounds_frames), pformat(bounds_frames))))
        g['l3_segments'][file_]['seg_clust_1'] = bounds_frames

        bounds_frames = g['func'][compute_segments_librosa](g['l2_chromagram'][file_]['data'], sr_comp, numsegs + 5)['bounds_frames']
        # print(('    file_: {0}, bounds_frames {1}, {2}'.format(file_, len(bounds_frames), pformat(bounds_frames))))
        g['l3_segments'][file_]['seg_clust_2'] = bounds_frames
        
    # layer 4: compute onsets
    g['l4_onsets'] = OrderedDict()
    for file_ in g['l1_files']:
        onsets = g['func'][compute_onsets_librosa](g['l1_files'][file_]['data'], sr_comp)
        g['l4_onsets'][file_] = onsets

    # layer 5: compute beats based on onsets
    g['l5_beats'] = OrderedDict()
    for file_ in g['l4_onsets']:
        g['l5_beats'][file_] = {}
        for start_bpm in [60, 90, 120]:
            beats = g['func'][compute_beats_librosa](g['l4_onsets'][file_]['onsets_env'], g['l4_onsets'][file_]['onsets_frames'], start_bpm, sr_comp)
            # print('beats type = {0}'.format(type(beats['beats'])))
            # beats['beats'] = beats['beats'][np.logical_not(np.isnan(beats['beats']))]
            # beats = beats[~np.isnan(beats)]
            # print('    file_: {0}, bounds_frames {1}, {2}'.format(file_, len(bounds_frames), pformat(bounds_frames)))
            g['l5_beats'][file_]['beats_{0}'.format(start_bpm)] = beats['beats']
            g['l5_beats'][file_]['beats_{0}_16'.format(start_bpm)] = beats['beats'][::16]

    joblib.dump(g, './g-pre.pkl')
    # layer 6: compute final segments from merging segments with beats
    # prepare
    g['l6_merge'] = OrderedDict()
    files_ = []
    for file_ in g['l1_files']:
        dirname = os.path.dirname(filename)
        print(f'main_autoedit l6_merge {file_} / {dirname}/{filename}')
        beats_keys = ['beats_60', 'beats_90', 'beats_120'] + ['beats_60_16', 'beats_90_16', 'beats_120_16']
        # beats = [g['l5_beats'][file_][beat_type] for beat_type in beats_keys for file_ in g['l1_files']]
        beats = [g['l5_beats'][file_][beat_type] for beat_type in beats_keys]
        # segs = [g['l3_segments'][file_][seg_type_] for seg_type_ in ['seg_sbic', 'seg_clust_1', 'seg_clust_2'] for file_ in g['l1_files']]
        segs = [g['l3_segments'][file_][seg_type_] for seg_type_ in ['seg_sbic', 'seg_clust_1', 'seg_clust_2']]
        numframes = g['l1_files'][file_]['numframes']
        # compute
        files = g['func'][compute_event_merge_combined](filename_48=dirname + '/' + file_, beats=beats, segs=segs, numframes=numframes, numsegs=numsegs)
        # files_.extend(files)
        g['l6_merge'].update(files)
    # g['l6_merge']['files'] = files_

    # layer 7: compute assembled song from segments and duration
    g['l7_assemble'] = OrderedDict()
    # compute
    g['l6_merge']['duration'] = duration
    filename_short = list(g['l1_files'])[0]
    autoedit_count = autoedit_get_count()
    filename_export = f'{filename_short[:-4]}-autoedit-{autoedit_count}.wav'
    g['l6_merge']['filename_export'] = filename_export

    if kwargs['assemble_mode'] == 'random':
        g['l7_assemble']['outfile'] = g['func'][track_assemble_from_segments](**(g['l6_merge']))
    elif kwargs['assemble_mode'] == 'sequential':
        g['l7_assemble']['outfile'] = g['func'][track_assemble_from_segments_sequential_scale](**(g['l6_merge']))

    # print((pformat(g)))
    # joblib.dump(g, './g.pkl')
    filename_export_graph = f'{filename_export[:-4]}.pkl'
    print(f'exporting graph to {filename_export_graph}')
    joblib.dump(g, filename_export_graph)
    
    # # plot dictionary g as graph
    # autoedit_graph_from_dict(g=g, plot=False)

def autoedit_get_count():
    """autoedit get count

    load autoedit count from file, if it does not exist, init zero and
    save to file.
    """
    autoedit_count_filename = 'data/audio_sort_features/autoedit-count.txt'
    if os.path.exists(autoedit_count_filename):
        autoedit_count = int(open(autoedit_count_filename, 'r').read().strip())
        print(f'autoedit_get_count from file {autoedit_count}, {type(autoedit_count)}')
    else:
        autoedit_count = 0
        print(f'autoedit_get_count file no found, init {autoedit_count}')
    autoedit_count_new = autoedit_count + 1
    f = open(autoedit_count_filename, 'w')
    f.write(f'{autoedit_count_new}\n')
    f.flush()
    return autoedit_count
    
# aubio foo
def autobeat_filter(args):
    """autobeat_filter

    Filter beat array with tempo, quantization or sparsity prior
    """
    return {}

def main_autobeat(args):
    """main_autobeat

    Extract beat from audio, return tempo in bpm, array of all beat
    onset events

    .. TODO::
    - use general lib loading
    - create src at this level for reuse
    - save / return beat array
    """
    # convert args to dict
    kwargs = args_to_dict(args)

    # for filename in kwargs['filenames']:
    filename = kwargs['filenames'][0]
    onsets = compute_onsets_aubio(filename=filename)
    # print ('onsets = {0}'.format(pformat(onsets)))
    print ('onsets[onsets] = {0}'.format(onsets['onsets'].shape))
    onsets_x = frames_to_time(np.arange(0, onsets['onsets'].shape[0]), sr=onsets['src'].samplerate, hop_length=onsets['src'].hop_size)
    
    method = 'specdiff'
    tempo_beats = compute_tempo_beats_aubio(path=filename, method=method)

    # print ('tempo_beats = {0}'.format(pformat(tempo_beats)))
    print ('tempo_beats[bpm] = {0}, tempo_beats[beats] = {1}'.format(tempo_beats['bpm'], tempo_beats['beats'].shape))

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_title('onset detection function')
    ax1.plot(onsets_x, np.array(onsets['onsets']))
    
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.set_title('beats')
    ax2.text(0.1, 0.1, 'Tempo = {0:.2f}'.format(tempo_beats['bpm']))
    ax2.bar(tempo_beats['beats'], 1.0, width=0.1)
    ax2.set_xlabel('time [s]')
    
    plt.show()
    
def main_segtree(args):
    """main_segtree

    OBSOLETE see aubio_cut.py --mode scan

    Build a segmentation tree w/ scanning (tm)
    """
    import aubio
    
    # convert args to dict
    kwargs = args_to_dict(args)

    src = aubio.source('/home/src/QK/data/sound-arglaaa-2018-10-25/24.wav', channels=1)
    src.seek(0)
    onsets = []
    onsets2 = []

    od = aubio.onset(method='kl', samplerate=src.samplerate)
    od.set_threshold(0.7)

    while True:
        samples, read = src()
        # print(samples.shape)
        if read < src.hop_size:
            break
        od_result = od(samples)
        # print(od_result)
        onsets.append(od.get_descriptor())
        onsets2.append(od.get_thresholded_descriptor())
        # onsets.append(od_result)

    onsets_ = np.array(onsets)
    onsets2_ = np.array(onsets2)
    print(onsets)
    plt.plot(onsets_)
    plt.plot(onsets2_)
    plt.plot(onsets_ > od.get_threshold(), linewidth=2, alpha=0.5, linestyle='none', marker='o')
    plt.show()


def main_autocover(args):
    """autocover

    take an audiofile and create a cover for it using data driven
    mapping, be able to select from different mappings
    """
    # autocover_recurrenceplot(args)
    autocover_feature_matrix(args)

def autocover_feature_matrix(args):
    import librosa
    from pyunicorn.timeseries import RecurrencePlot
    from matplotlib.colors import LogNorm
    import colorcet as cc
    kwargs = args_to_dict(args)
    # open file compute frame based features
    
    print(f'autocover kwargs {pformat(kwargs)}')
    # w, samplerate = librosa.load(kwargs['filenames'][0])

    compute_features_paa_cached = memory.cache(compute_features_paa)
    compute_music_extractor_essentia_cached = memory.cache(compute_music_extractor_essentia)
    
    for filename in kwargs['filenames']:
        F, F_names, G = compute_features_paa_cached(filename)
        print(f'F_names {pformat(F_names)}\nF {F.shape}, G {G.shape}')

        feature_matrix = []
        for i, feature_key in enumerate([
                'zcr_mean',
                'energy_mean',
                'energy_entropy_mean',
                'spectral_centroid_mean',
                'spectral_spread_mean',
                'spectral_entropy_mean',
                'spectral_flux_mean',
                'spectral_rolloff_mean',
                'mfcc_1_mean',
                'mfcc_2_mean',
                'mfcc_3_mean',
                'mfcc_4_mean',
                'mfcc_5_mean',
                'mfcc_6_mean',
                'mfcc_7_mean',
                'mfcc_8_mean',
                'mfcc_9_mean',
                'mfcc_10_mean',
                'mfcc_11_mean',
                'mfcc_12_mean',
                'mfcc_13_mean',
                'chroma_1_mean',
                'chroma_2_mean',
                'chroma_3_mean',
                'chroma_4_mean',
                'chroma_5_mean',
                'chroma_6_mean',
                'chroma_7_mean',
                'chroma_8_mean',
                'chroma_9_mean',
                'chroma_10_mean',
                'chroma_11_mean',
                'chroma_12_mean',
        ]):
            feature_matrix.append(G[i])

        me = compute_music_extractor_essentia_cached(filename)
        print(f'music extractor {type(me)}')
            
        feature_matrix = np.array(feature_matrix)
        print(f'feature_matrix {np.min(feature_matrix)} {np.max(feature_matrix)}')

        fig = plt.figure()
    
        # ax2.pcolormesh(xs, ys, plotdata, cmap=plt.get_cmap("Oranges"))

        nmin = np.min(feature_matrix)
        nmax = np.max(feature_matrix)
        ax3 = fig.add_subplot(111)
        # ax3.imshow(np.log(feature_matrix), aspect='auto', origin='lower',
        #            interpolation='none',
        #            # norm=LogNorm(vmin=nmin, vmax=nmax)
        # )

        length0 = feature_matrix.shape[0]
        length1 = feature_matrix.shape[1]
        ys = np.linspace(0, length0, length0)
        xs = np.linspace(0, length1, length1)
        # ax3.pcolormesh(xs, ys, np.log(feature_matrix), cmap=plt.get_cmap("Oranges"))
        ax3.pcolormesh(xs, ys, np.log(feature_matrix), cmap=cc.cm['colorwheel'])
        ax3.set_aspect(length1/length0)
        ax3.axis('off')
        # ax3.set_title(os.path.basename(filename))
        savefilename = os.path.dirname(filename) + '/' + os.path.basename(filename)[:-4] + '.pdf'
        print(f'saving to {savefilename}')
        fig.set_size_inches((10, 10))
        fig.savefig(savefilename, dpi=300, bbox_inches='tight')
        plt.show()
        

def autocover_recurrenceplot(args):
    import librosa
    from pyunicorn.timeseries import RecurrencePlot
    kwargs = args_to_dict(args)
    # open file compute frame based features
    
    print(f'autocover kwargs {pformat(kwargs)}')
    w, samplerate = librosa.load(kwargs['filenames'][0])

    framesize = 4096
    mfcc = librosa.feature.mfcc(y=w, sr=samplerate, n_fft=framesize, hop_length=framesize, center=False)
    rp = RecurrencePlot(mfcc.T, threshold_std=0.5)
    plotdata = rp.recurrence_matrix()
    
    fig = plt.figure()      
    
    ax1 = fig.add_subplot(221)
    ax1.plot(w)
    
    # print ("recmat.shape", rp.recurrence_matrix().shape)

    length = plotdata.shape[0]

    ax2 = fig.add_subplot(222)
    # ax2.matshow(rp.recurrence_matrix())
    xs = np.linspace(0, length, length)
    ys = np.linspace(0, length, length)
    ax2.pcolormesh(xs, ys, plotdata, cmap=plt.get_cmap("Oranges"))
    ax2.set_aspect(1)
    ax2.set_xlabel("$n$")                                                                                    
    ax2.set_ylabel("$n$")

    ax3 = fig.add_subplot(223)
    ax3.imshow(mfcc[1:,:], aspect='auto', origin='lower', interpolation='none')
    
    plt.show()

    
def main_scanfiles(args):
    """main_scanfiles

    scan files and analyze for features: path, length, fingerprint, energy, entropy, segments
    """
    # load list of file to scan, for each file
    # compute features
    # problem: granularity: short files / long files, fixed size chunks / varisized chunks (segmentation)
    # problem: granularity: from onset level to part level
    # problem: localization, fractional calculus
    pass

"""
- batch vs. frame vs. sample based processing: blocksize
- low-level api backend
- track db: 1 file, 2 service, check plb, droptrack
- audiofile_tool
  - audio file info
  - duration
  - duration stats
- grab file, split large files into manageable chunks
- autoedit multifile input
- autoedit power sorting and distribution/envelopes
- graphical sound browser


- merge with droptrack/player/store.py
"""
def recursive_walk(paths, filenames):
    # iterate over list argument paths
    for path in paths:
        # do os.walk
        for root, dirs, files in os.walk(path):
            print(f'        root {root}, dirs {dirs}, files {files}')
            # iterate file on this dir level
            for _file in files:
                # ignore garbage files
                if _file in ['.DS_Store']:
                    continue
                # append the file to expanded list
                filenames.append(root + '/' + _file)

    # return result
    return filenames

def args_filenames_expand(args_filenames):
    _filenames = []
    # check filenames for: audiofile, textfile, directory
    for filename in args_filenames:
        print(f'filename {filename}')
        print(f'    basename {os.path.basename(filename)}')
        print(f'     dirname {os.path.dirname(filename)}')
        # is a directory, walk it
        if os.path.isdir(filename):
            print(f'       isdir {os.path.isdir(filename)}')
            print(f'       walking ...')
            tmp = recursive_walk([filename], [])
            print(f'        tmp filenames from walk {pformat(tmp)}')
            _filenames.extend(tmp)
        # is a text file containing a list of audiofiles
        elif filename.endswith('.txt'):
            # read the entire files into list
            tmp = open(filename, 'r').readlines()
            # strip junk from each string
            tmp = [_.strip() for _ in tmp]
            # tmp = ['sdsd']
            print(f'        tmp filenames from  txt {pformat(tmp)}')
            _filenames.extend(tmp)
        # is a plain audio filename
        else:
            _filenames.append(filename)

    # return result
    return _filenames

class audiofile_store(object):
    def __init__(self):
        # init track store
        self.trackstore_filename = 'data/trackstore.csv'
        # self.trackstore_key = 'miniclub6'
        try:
            self.ts = pd.read_csv(self.trackstore_filename)
            # self.trackstore = pd.HDFStore(self.trackstore_filename, mode='a')
            # self.ts = pd.read_hdf(self.trackstore_filename, self.trackstore_key)
        except Exception as e:
            print('Could not load trackstore from file at {0}'.format(self.trackstore_filename))
            # continue without trackstore
            # self.trackstore = None
            # self.trackstore = pd.DataFrame(columns=['id', 'url', 'filename', 'filepath', 'length', 'fingerprint', 'hash'])
            self.ts = pd.DataFrame({
                'id': 0,
                'url': 'none',
                'filename': 'none',
                'filepath': 'none',
                'duration': 0.0,
                'channels': 0,
                'samplerate': 0,
                'codec': 'none',
                'fingerprint': 'none',
                'hash': ''
            }, index=pd.Index([0]))
            self.ts.to_csv(self.trackstore_filename, index=False)
            
        # self.ts = self.trackstore[self.trackstore_key]

def split_large_audiofile(filename, args):
    """split large audiofile

    If audiofile `filename` is longer than a threshold, split into
    manageable chunks of < threshold.

    Uses aubiocut scanning to find a suitable segmentation combining:

    onset_method, threshold, bufsize, and minioi
    """
    from aubio_cut import _cut_analyze, _cut_slice
    options = argparse.Namespace()
    # options: onset_method, minioi, threshold
    # options: hop_size, buf_size, samplerate, source_uri
    options.onset_method = 'default'
    options.minioi = '300s'
    options.threshold = 1.0
    options.hop_size = 2048
    options.buf_size = 2048
    options.samplerate = 0
    options.source_uri = filename
    options.beat = False
    options.verbose = True
    options.cut_every_nslices = None
    options.cut_until_nslices = None
    options.cut_until_nsamples = None
    options.output_directory = None
    timestamps, total_frames = _cut_analyze(options)

    _cut_slice(options, timestamps)
    info = f'created {len(timestamps)} slices from {total_frames} frames\n'
    # info += base_info
    sys.stderr.write(info)

    return timestamps, total_frames
        
def main_audiofile_tool(args):
    """audiofile tool

    - input files: filenames args, textfile list of files, tree scanning
    - open track db: do we know the track?
    - if known: return precomputed info
    - if unknown: compute info and put it into db
    """
    print(f'main_audiofile_tool args {pformat(args)}')

    # augment filenames with single files, textfile with list, scanned directories
    filenames = args_filenames_expand(args.filenames)
    print(f'filenames expanded {pformat(filenames)}')

    # open existing track db
    # csv, sqlite, hdf5
    # uuid, name, path, duration, channels, fingerprint, color
    afs = audiofile_store()
    print(f'trackstore loaded with {afs.ts.shape[0]} entries')

    # loop over filenames
    for filename in filenames:
        print(f'    processing {filename}')
        # audioread open
        fi = get_fileinfo(filename)
        print(f'      fileinfo {pformat(fi)}')
        # if file is longer than 10 minutes = 600 seconds
        if fi['duration'] > 600.0:
            print(f'      file too long')
            # split file into good chunks using aubiocut with minioi
            _filenames = split_large_audiofile(filename, args)
            print(f'_filenames {pformat(_filenames)}')
    return None
    
def main(args):
    # print(f'main args = {args}')
    
    np.random.seed(args.seed)

    if args.mode == 'beatiness':
        _main = main_beatiness
    elif args.mode == 'music_extractor':
        _main = main_music_extractor
    elif args.mode == 'paa_feature_extractor':
        _main = main_paa_feature_extractor
    elif args.mode == 'autoedit':
        _main = main_autoedit
    elif args.mode == 'automix':
        _main = main_automix
    elif args.mode == 'autobeat':
        _main = main_autobeat
    elif args.mode == 'segtree':
        _main = main_segtree
    elif args.mode == 'audiofile_tool':
        _main = main_audiofile_tool
    elif args.mode == 'timing_read_stream':
        _main = main_timing_read_stream
    elif args.mode == 'autoedit_stream':
        _main = main_autoedit_stream
    elif args.mode == 'autocover':
        _main = main_autocover
    else:
        print('Unknown mode {0}, exiting'.format(args.mode))
        sys.exit(1)

    if args.mode.endswith('_stream'):
        ret = []
        for filename_i, filename in enumerate(args.filenames[0]):
            print(f'main running {args.mode} on {filename}')
            src, sr, framesize, filename, args.srclength = get_src(filename, args)
            ret.append(_main(filename, src, sr, framesize, args))

        print(f'ret = {pformat(ret)}')
    else:
        args.filenames = args.filenames[0]
    
        ret = _main(args)
        plt.show()
    
if __name__ == "__main__":
    print(f'main enter')
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f", "--filenames", action='append', dest='filenames', help="Input file(s) []", nargs = '+', default = [], required=True)
    parser.add_argument("-a", "--assemble-mode", dest='assemble_mode',
                        help="Assemble mode [random] (random, sequential)",
                        default='random')
    parser.add_argument("-m", "--mode", dest='mode',
                        help="Feature mode [beatiness] (beatiness, music_extractor, paa_feature_extractor, autoedit, automix, autobeat, segtree, timing_read_stream, autoedit_stream)",
                        default='beatiness')
    parser.add_argument("-d", "--duration", dest='duration', default=180, type=float, help="Desired duration in seconds [180]")
    parser.add_argument("-ns", "--numsegs", dest='numsegs', default=10, type=int, help="Number of segments for segmentation")
    parser.add_argument("-src", "--sr-comp", dest='sr_comp', default=22050, help="Sample rate for computations [22050]")
    parser.add_argument("-s", "--sorter", dest='sorter', default='features_mt_spectral_spread_mean', help="Sorting feature [features_mt_spectral_spread_mean]")
    parser.add_argument("--seed", dest='seed', type=int, default=123, help="Random seed [123]")
    parser.add_argument("-smin", "--seglen-min", dest='seglen_min', default=2, help="Segment length minimum in seconds [2]")
    parser.add_argument("-smax", "--seglen-max", dest='seglen_max', default=60, help="Segment length maximum in seconds [60]")
    parser.add_argument("-w", "--write", dest='write', type=bool, default=False, help="Write output [False]")
    # params: numsegments, duration, minlength, maxlength, kernel params
    args = parser.parse_args()

    main(args)
