"""sort audio files by some scalar feature: beatiness, density

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
import argparse, time, pickle, sys
from collections import OrderedDict
from pprint import pformat
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from librosa import samples_to_frames

from smp_audio.common_essentia import data_load_essentia
from smp_audio.common_essentia import compute_segments_essentia
from smp_audio.common_essentia import compute_tempo_beats_essentia, compute_music_extractor_essentia
from smp_audio.common_librosa import compute_segments_librosa, compute_chroma_librosa, compute_beats_librosa, compute_onsets_librosa
from smp_audio.assemble_pydub import track_assemble_from_segments, track_assemble_from_segments_sequential
from smp_audio.graphs import graph_walk_collection_flat, graph_walk_collection
from smp_audio.graphs import cb_graph_walk_build_graph
from smp_audio.util import args_to_dict

from audio_segments_split import main_audio_segments_split
from audio_features_paa import compute_features_paa

# caching joblib
from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

def paa_feature_extractor_main(args):
    # convert args to dict
    kwargs = args_to_dict(args)
    print('paa_feature_extractor_main enter kwargs {0}'.format(kwargs))
    
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
    data_load_essentia_cached = memory.cache(data_load_essentia)
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

def automix_main(args):
    """automix_main

    Complete automix flow with the schema
    
    1. input list of audio files / text file containing list of audio files
    2. loop over files
    2.1. compute bag of measures for each file: beatiness, extractor essentia, features paa

    2.2. TODO: sort files by selected feature args.sort_feature
    2.3. TODO: assemble output wav from concatenating input files pydub

    2.4. TODO: optional: local measures
    2.4. TODO: optional: complexity / information measures smp/sequence
    """
    # convert args to dict
    kwargs = args_to_dict(args)

    print('kwargs {0}'.format(pformat(kwargs)))

    # flow graph g
    g = OrderedDict()

    # functions cached / non-cached
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

    for func in [
            main_audio_segments_split,
            track_assemble_from_segments,
    ]:
        g['func'][func] = func


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

    track_assemble_from_segments_sequential(files=list(l1_files_df.sort_values(sort_key, ascending=False).path),
                                            output_filename='sendspaace-assembled-{0}-{1}.{2}'.format(3, sort_key, 'wav'),
                                            duration=None)
    
def autoedit_graph_from_dict(**kwargs):
    g = kwargs['g']

    print((pformat(g)))
    joblib.dump(g, './g.pkl')
    
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
    
def autoedit_main(args):
    """autoedit_main

    Complete autoedit flow
    """
    # convert args to dict
    kwargs = args_to_dict(args)

    # globals
    sr_comp = kwargs['sr_comp'] # 22050
    numsegs = kwargs['numsegs'] # 10
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
            main_audio_segments_split,
            track_assemble_from_segments,
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
        g['l1_files'][filename_short]['numframes'] = samples_to_frames(len(tmp_[0]))
        g['l1_files'][filename_short]['sr'] = tmp_[1]

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
        print(('    file_: {0}, bounds_frames {1}, {2}'.format(file_, len(bounds_frames), pformat(bounds_frames))))
        g['l3_segments'][file_] = {}
        g['l3_segments'][file_]['seg_sbic'] = np.clip(bounds_frames, 0, [g['l1_files'][filename_short]['numframes'] for filename_short in g['l1_files']][0]-1)

        bounds_frames = g['func'][compute_segments_librosa](g['l2_chromagram'][file_]['data'], sr_comp, numsegs)['bounds_frames']
        print(('    file_: {0}, bounds_frames {1}, {2}'.format(file_, len(bounds_frames), pformat(bounds_frames))))
        g['l3_segments'][file_]['seg_clust_1'] = bounds_frames

        bounds_frames = g['func'][compute_segments_librosa](g['l2_chromagram'][file_]['data'], sr_comp, numsegs + 5)['bounds_frames']
        print(('    file_: {0}, bounds_frames {1}, {2}'.format(file_, len(bounds_frames), pformat(bounds_frames))))
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
            # print('    file_: {0}, bounds_frames {1}, {2}'.format(file_, len(bounds_frames), pformat(bounds_frames)))
            g['l5_beats'][file_]['beats_{0}'.format(start_bpm)] = beats['beats']

    # layer 6: compute final segments from merging segments with beats
    # prepare
    g['l6_merge'] = OrderedDict()
    beats = [g['l5_beats'][file_][beat_type] for beat_type in ['beats_60', 'beats_90', 'beats_120'] for file_ in g['l5_beats']]
    segs = [g['l3_segments'][file_][seg_type_] for seg_type_ in ['seg_sbic', 'seg_clust_1', 'seg_clust_2'] for file_ in g['l3_segments']]
    numframes = [g['l1_files'][filename_short]['numframes'] for filename_short in g['l1_files']]
    # compute
    files = g['func'][main_audio_segments_split](filename_48=filename, beats=beats, segs=segs, numframes=numframes[0], numsegs=numsegs)
    g['l6_merge'].update(files)

    # layer 7: compute assembled song from segments and duration
    g['l7_assemble'] = OrderedDict()
    # compute
    g['l6_merge']['duration'] = duration
    g['l7_assemble']['outfile'] = g['func'][track_assemble_from_segments](**(g['l6_merge']))

    # plot dictionary g as graph
    autoedit_graph_from_dict(g=g, plot=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filenames", action='append', dest='filenames', help="Input file(s) []", nargs = '+', default = [])
    parser.add_argument("-m", "--mode", dest='mode',
                        help="Feature mode [beatiness] (beatiness, music_extractor, paa_feature_extractor, autoedit, automix)",
                        default='beatiness')
    parser.add_argument("-d", "--duration", dest='duration', default=180, type=float, help="Desired duration in seconds [180]")
    parser.add_argument("-ns", "--numsegs", dest='numsegs', default=10, type=int, help="Number of segments for segmentation")
    parser.add_argument("-src", "--sr-comp", dest='sr_comp', default=22050, help="Sample rate for computations [22050]")
    parser.add_argument("-s", "--sorter", dest='sorter', default='features_mt_spectral_spread_mean', help="Sorting feature [features_mt_spectral_spread_mean]")
    # params: numsegments, duration, minlength, maxlength, kernel params
    args = parser.parse_args()

    args.filenames = args.filenames[0]

    np.random.seed(123)

    if args.mode == 'beatiness':
        main_beatiness(args)
    elif args.mode == 'music_extractor':
        main_music_extractor(args)
    elif args.mode == 'paa_feature_extractor':
        main_paa_feature_extractor(args)
    elif args.mode == 'autoedit':
        autoedit_main(args)
    elif args.mode == 'automix':
        automix_main(args)
    else:
        print('Unknown mode {0}, exiting'.format(args.mode))
        sys.exit(1)

    plt.show()
