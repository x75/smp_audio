import os
from pprint import pformat

import json, codecs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import colorcet as cc

from smp_audio.audio_features_paa import compute_features_paa
from smp_audio.caching import memory
from smp_audio.common import NumpyEncoder
from smp_audio.common import args_to_dict, ns2kw, kw2ns
from smp_audio.common_essentia import compute_music_extractor_essentia


# TODO: create autocover_conf_default
autocover_conf = {
    'default': {
        'filenames': [],
        'mode': 'autocover',
        'autocover_mode': 'feature_matrix',
        'rootdir': './',
        'seed': 1234,
        'sorter': 'features_mt_spectral_spread_mean',
        'sr_comp': 22050,
        'verbose': False,
        'outputs': ['json', 'jpg'],
    },
}
autocover_conf_default = autocover_conf['default']

def main_autocover(args, **kwargs):
    """autocover

    take an audiofile and create a cover for it using data driven
    mapping, be able to select from different mappings
    """
    if args.autocover_mode == 'recurrence_matrix':
        # TODO: activate recurrence matrix
        return autocover_recurrenceplot(args, **kwargs)
    elif args.autocover_mode == 'feature_matrix':
        return autocover_feature_matrix(args, **kwargs)


def export_graphics(feature_matrix, args):
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
    # ax3.pcolormesh(xs, ys, np.log(feature_matrix), cmap=cc.cm['colorwheel'])
    # mycmap = cc.cm['colorwheel']
    mycmap = cc.cm[np.random.choice(list(cc.cm.keys()))]
    if args.verbose:
        print(f'autocover mycmap = {mycmap.name}')
    ax3.pcolormesh(xs, ys, np.log(feature_matrix), cmap=mycmap)
    ax3.set_aspect(length1/length0)
    ax3.axis('off')
    # ax3.set_title(os.path.basename(filename))
    # if len(os.path.dirname(filename)) > 0:
    #     sep = '/'
    # else:
    #     sep = ''
    
    fig.set_size_inches((10, 10))
    
    for savetype in ['pdf', 'jpg']:
        if not savetype in args.outputs:
            continue
        
        # os.path.basename(filename)[:-4] + savetype
        savefilename = os.path.join(
            args.rootdir,
            os.path.basename(args.filename_export) + "." + savetype)

        if args.verbose:
            print(f'autocover saving to {savefilename}')
        fig.savefig(savefilename, dpi=300, bbox_inches='tight')
        
    
def autocover_feature_matrix(args, **kwargs):
    # import librosa

    kwargs_ns = args_to_dict(args)

    if args.verbose:
        print(f'autocover kwargs_ns {pformat(kwargs_ns)}')
        
    # open file compute frame based features
    # w, samplerate = librosa.load(kwargs_ns['filenames'][0])
    compute_features_paa_cached = memory.cache(compute_features_paa)
    # compute_music_extractor_essentia_cached = memory.cache(compute_music_extractor_essentia)
    
    for filename in kwargs_ns['filenames']:
        F, F_names, G, F_time, G_time = compute_features_paa_cached(filename, with_timebase=True, verbose=args.verbose)
        if args.verbose:
            print(f'autocover F_names {pformat(F_names)}')
            print(f'autocover F.shape {F.shape}, G.shape {G.shape}')

        feature_matrix = []
        feature_matrix_dict = {}
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
            feature_matrix_dict[feature_key] = G[i]

        feature_matrix_dict['t_seconds'] = G_time

        # # not used?
        # me = compute_music_extractor_essentia_cached(filename)
        # print(f'autocover music extractor {type(me)}')
            
        feature_matrix = np.array(feature_matrix)
        if args.verbose:
            print(f'autocover feature_matrix {np.min(feature_matrix)} {np.max(feature_matrix)}')

        if len(os.path.dirname(filename)) > 0:
            sep = '/'
        else:
            sep = ''
            
        # write = False

        # savefilename = os.path.dirname(filename) + sep + os.path.basename(filename)[:-4] + '.json'
        # savefilename = args.filename_export[:-4] + '.json'
        savefilename = os.path.join(
            args.rootdir,
            os.path.basename(args.filename_export) + "-feature-matrix")
        
        # this saves the array in .json format
        json.dump(
            feature_matrix_dict,
            codecs.open(savefilename + ".json", 'w', encoding='utf-8'),
            separators=(',', ':'),
            sort_keys=True,
            indent=4,
            cls=NumpyEncoder,
        )

        # res_ = json.dumps(
        #     feature_matrix_dict,
        #     cls=NumpyEncoder,
        # )
        # res = json.loads(res_)

        # if args.verbose:
        #     print(f"autocover_feature_matrix res {type(res)}")
        #     print(f"autocover_feature_matrix res {res.keys()}")
        
        # json.dump(feature_matrix_dict, open(savefilename, 'w'))
            
        # save as png / jpg straightaway?
        # use inkscape to post process
        # inkscape --export-png=11.84.0.-1.0-1.1-1_5072.884286-autoedit-11_master_16bit.png --export-dpi=400 11.84.0.-1.0-1.1-1_5072.884286-autoedit-11_master_16bit.pdf
        # plt.show()
        
    res = {
        'data': {
            'output_files': [
                # {'format': 'json', 'filename': os.path.basename(savefilename)}
            ],
        }
    }

    # export graphics
    if 'pdf' in args.outputs or 'jpg' in args.outputs:
        export_graphics(feature_matrix, args)

    # record all output files
    for output_type in args.outputs:
        res['data']['output_files'].append(
            {'format': output_type, 'filename': os.path.basename(args.filename_export) + "." + output_type}
        )
        
    filename_result = os.path.join(
        args.rootdir,
        os.path.basename(args.filename_export) + ".json")

    # this saves the array in .json format
    json.dump(
        res,
        codecs.open(filename_result, 'w', encoding='utf-8'),
        # separators=(',', ':'),
        # sort_keys=True,
        # indent=4,
        # cls=NumpyEncoder,
    )
           
    if 'task' in kwargs:
        kwargs['task'].set_done(result_location=os.path.basename(args.filename_export) + ".json")
    
    return res

def autocover_recurrenceplot(args, **kwargs):
    import librosa
    from pyunicorn.timeseries import RecurrencePlot
    from matplotlib.colors import LogNorm
    
    kwargs_ns = args_to_dict(args)
    # open file compute frame based features
    
    print(f'autocover kwargs_ns {pformat(kwargs_ns)}')
    # TODO: multiple filenames
    filename = kwargs_ns['filenames'][0]
    w, samplerate = librosa.load(filename)

    RecurrencePlot_cached = memory.cache(RecurrencePlot)
    # RecurrencePlot_cached = RecurrencePlot
    
    framesize = 4096
    mfcc = librosa.feature.mfcc(y=w, sr=samplerate, n_fft=framesize, hop_length=framesize, center=False)
    rp = RecurrencePlot_cached(mfcc.T, threshold_std=0.5)
    plotdata = rp.recurrence_matrix()
    
    fig = plt.figure()      
    
    # ax1 = fig.add_subplot(221)
    # ax1.plot(w)
    
    # print ("recmat.shape", rp.recurrence_matrix().shape)

    length = plotdata.shape[0]
    
    # ax2 = fig.add_subplot(222)
    ax2 = fig.add_subplot(111)
    # ax2.matshow(rp.recurrence_matrix())
    xs = np.linspace(0, length, length)
    ys = np.linspace(0, length, length)
    #mycmap = plt.get_cmap("Oranges")
    mycmap = cc.cm[np.random.choice(list(cc.cm.keys()))]
    print(f'autocover mycmap = {mycmap.name}, min {plotdata.min()}, max {plotdata.max()}')
    plotdata = plotdata + 1
    ax2.pcolormesh(xs, ys, plotdata,
                   norm=colors.LogNorm(vmin=plotdata.min(), vmax=plotdata.max()),
                   cmap=mycmap)
    ax2.set_aspect(1)
    ax2.axis('off')
    # ax2.set_xlabel("$n$")                                                                                    
    # ax2.set_ylabel("$n$")

    # ax3 = fig.add_subplot(223)
    # ax3.imshow(mfcc[1:,:], aspect='auto', origin='lower', interpolation='none')
    
    if len(os.path.dirname(filename)) > 0:
        sep = '/'
    else:
        sep = ''
            
    fig.set_size_inches((10, 10))

    # for savetype in ['.pdf', '.jpg']:
    for savetype in ['.jpg']:
        savefilename = os.path.dirname(filename) + sep + os.path.basename(filename)[:-4] + savetype
        print(f'autocover saving to {savefilename}')
        fig.savefig(savefilename, dpi=300, bbox_inches='tight')
            
    # plt.show()

    
