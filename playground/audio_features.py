"""music_features.py

Compute features for pieces of music audio

Oswald Berthold, 2017

Taking the essentia [1] library and the extractor module to compute a
comprehensive feature-gram over a given chunk of audio.
"""

# stdlib
import argparse, pickle, re, os, sys, copy
from collections import OrderedDict

# numpy/scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

import mdp

# essentia
import essentia as e
import essentia.standard as estd

import logging
# from smp_base.common import get_module_logger
from smp_audio.common import get_module_logger

logger = get_module_logger(modulename = 'music_features', loglevel = logging.DEBUG)

# from essentia.standard import *
# import essentia.streaming

# print dir(essentia.standard)

class SFA(object):
    def __init__(self, numcomps = 1, numexps = 3, rbfc = np.zeros((10, 1)), rbfs = 1.0):
        self.flow = (
#            mdp.nodes.EtaComputerNode() +
#            mdp.nodes.TimeFramesNode(3) +
            mdp.nodes.PolynomialExpansionNode(numexps) +
#            mdp.nodes.RBFExpansionNode(rbfc, rbfs) +
            mdp.nodes.SFANode(output_dim = numcomps) 
#            mdp.nodes.EtaComputerNode()
        )

    def fit_transform(self, X):
        self.fit(X)
        return self.predict(X)
        
    def fit(self, X):
        self.flow.train(X)

    def predict(self, X):
        slow = self.flow(X)
        return slow


def makefig(rows = 1, cols = 1, title = '', add_subplots = True):
    """util.makefig

    Create a plot figure and subplot grid

    Return figure handle 'fig'
    """
    fig = plt.figure()
    gs = GridSpec(rows, cols)
    if add_subplots:
        for i in gs:
            fig.add_subplot(i)
    fig.suptitle(title)
    return (fig, gs)

def loadaudio(args):
    """util.loadaudio

    Load data from an audio file of any format supported by Monoloader
    """
    loader = estd.MonoLoader(filename=args.file, sampleRate=args.samplerate)
    return loader()

def main_simple(args):
    """main_simple

    Compute short time spectral feature map
    """

    plt.ion()
    
    audio = loadaudio(args)
    
    w = estd.Windowing(type = 'hamming')
    spectrum = estd.Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    mfcc = estd.MFCC()
    
    specgram = []
    mfccs = []
    melbands = []

    for frame in estd.FrameGenerator(audio, frameSize = args.frame_size_low_level, hopSize = args.frame_size_low_level, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)
        melbands.append(mfcc_bands)
        specgram.append(spectrum(w(frame)))
        

    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    mfccs = np.array(mfccs).T
    melbands = np.array(melbands).T
    specgram = np.array(specgram).T

    fig, gs = makefig(rows = 3, cols = 1, add_subplots = False)
    fig.show()

    print(("specgram.shape", specgram.shape))
    print(("melbands.shape", melbands.shape))
    
    ax1 = fig.add_subplot(gs[0,0])
    ax1.imshow(np.log(specgram[1:,:]), aspect = 'auto', origin='lower', interpolation='none')

    ax2 = fig.add_subplot(gs[1,0])
    ax2.imshow(mfccs[1:,:], aspect='auto', origin='lower', interpolation='none')

    ax3 = fig.add_subplot(gs[2,0])
    ax3.imshow(np.log(melbands[1:,:]), aspect = 'auto', origin='lower', interpolation='none')
    
    plt.draw()
    plt.pause(1e-9)

    # process
    numcomps = 3
    melbands_ = scale(melbands.T).T
    # wt = PCA(n_components = melbands.shape[0], whiten = True)
    # melbands_ = wt.fit_transform(melbands.T).T # scale(melbands.T).T
    # melbands_ = np.log(melbands + 1)  * 10

    print(("melbands", melbands.shape, "melbands_", melbands_.shape))
    print(("means", np.mean(melbands_, axis = 1)))
    
    sfa_in = melbands_[1:,:]
    sfa_cov = np.cov(sfa_in)
    print(("sfa_cov", sfa_cov.shape))
    # rbfcs = np.random.uniform(-5, 5, (numcomps, sfa_in.shape[0]))
    # sfa = SFA(numcomps = numcomps, numexps = 2) # , rbfc = rbfcs)
    sfa = KernelPCA(kernel="rbf", degree=5, fit_inverse_transform=True, gamma=10, n_components = numcomps)
    

    fig3, gs3 = makefig(rows = 1, cols = 2)
    
    fig3.axes[0].plot(sfa_in.T)
    
    fig3.axes[1].imshow(sfa_cov, aspect = 'auto', origin='upper', interpolation='none')
    fig3.axes[1].set_aspect(1)
    plt.draw()
    plt.pause(1e-9)

    try:
        # sfa_in += np.random.uniform(-1e-3, 1e-3, sfa_in.shape)
        melbands_sfa = sfa.fit_transform(sfa_in.T)
        # melbands_sfa = sfa.fit_transform(specgram[1:,:].T)
        print(("melbands_sfa.shape", melbands_sfa.shape))

        fig2, gs2 = makefig(rows = 1, cols = 2, add_subplots = False)
        fig2.show()

        ax = fig2.add_subplot(gs2[0,0])
        # ax.plot(melbands_sfa)
        # ax.imshow(np.log(melbands_sfa.T), aspect = 'auto', origin='lower', interpolation='none')
        # ax.imshow(np.log(np.abs(melbands_sfa.T)), aspect = 'auto', origin='lower', interpolation='none')
        ax.imshow(np.abs(melbands_sfa.T), aspect = 'auto', origin='lower', interpolation='none')
        
        ax = fig2.add_subplot(gs2[0,1])
        maxs = []
        for fr_ in melbands_sfa:
            print(("fr_", fr_.shape))
            maxs.append(np.argmax(np.abs(fr_)))
        ax.plot(np.array(maxs), "bo")

        plt.draw()
        plt.pause(1e-9)
        
    except Exception as e:
        print(("SFA failed", e))

    
    plt.ioff()
    plt.show()
    
    
def main_mfcc(args):
    """main_mfcc

    Compute short time windowed MFCC features for input waveform and
    plot them over time (mfcc-spectrogram)
    """
    plt.ion()

    audio = loadaudio(args)
    
    print(("audio", type(audio), audio.shape))

    # pylab contains the plot() function, as well as figure, etc... (same names as Matlab)
    plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default

    fig, gs = makefig(rows = 2, cols = 2)

    w = estd.Windowing(type = 'hann')
    spectrum = estd.Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    mfcc = estd.MFCC()

    # print "w", repr(w)
    # print "spectrum", repr(spectrum)
    # print "mfcc", repr(mfcc)

    frame = audio[int(0.2*args.samplerate) : int(0.2*args.samplerate) + 1024]
    print(("frame.shape", frame.shape))
    spec = spectrum(w(frame))
    mfcc_bands, mfcc_coeffs = mfcc(spec)

    print(("type(spec)", type(spec)))
    print(("spec.shape", spec.shape))

    fig.axes[0].plot(audio[int(0.2*args.samplerate):int(0.4*args.samplerate)])
    fig.axes[0].set_title("This is how the 2nd second of this audio looks like:")
    # plt.show() # unnecessary if you started "ipython --pylab"

    fig.axes[1].plot(spec)
    fig.axes[1].set_title("The spectrum of a frame:")

    fig.axes[2].plot(mfcc_bands)
    fig.axes[2].set_title("Mel band spectral energies of a frame:")

    fig.axes[3].plot(mfcc_coeffs)
    fig.axes[3].set_title("First 13 MFCCs of a frame:")

    fig.show()

    # plt.show() # unnecessary if you started "ipython --pylab"
    ################################################################################
    fig2, gs2 = makefig(rows = 2, cols = 2, add_subplots = False)

    mfccs = []
    melbands = []

    for frame in estd.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)
        melbands.append(mfcc_bands)

    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    mfccs = np.array(mfccs).T
    melbands = np.array(melbands).T

    pool = e.Pool()

    for frame in estd.FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.mfcc_bands', mfcc_bands)


    ax1 = fig2.add_subplot(gs2[0,0])
    ax1.imshow(pool['lowlevel.mfcc_bands'].T, aspect = 'auto', origin='lower', interpolation='none')
    ax1.set_title("Mel band spectral energies in frames")

    ax2 = fig2.add_subplot(gs2[0,1])
    ax2.imshow(pool['lowlevel.mfcc'].T[1:,:], aspect='auto', origin='lower', interpolation='none')
    ax2.set_title("MFCCs in frames")

    # and plot
    ax3 = fig2.add_subplot(gs2[1,0])
    ax3.imshow(melbands[:,:], aspect = 'auto', origin='lower', interpolation='none')
    ax3.set_title("Mel band spectral energies in frames")
    # show() # unnecessary if you started "ipython --pylab"

    ax4 = fig2.add_subplot(gs2[1,1])
    ax4.imshow(mfccs[1:,:], aspect='auto', origin='lower', interpolation='none')
    ax4.set_title("MFCCs in frames")

    fig2.show()


    plt.ioff()
    plt.show() # unnecessary if you started "ipython --pylab"

def main_danceability(args):
    """main_danceability

    Compute the danceability feature over input waveform and plot it
    """
    audio = loadaudio(args)
    
    # create the pool and the necessary algorithms
    pool = e.Pool()
    w = estd.Windowing()
    spec = estd.Spectrum()
    centroid = estd.SpectralCentroidTime()

    # compute the centroid for all frames in our audio and add it to the pool
    for frame in estd.FrameGenerator(audio, frameSize = 1024, hopSize = 512):
        c = centroid(spec(w(frame)))
        pool.add('lowlevel.centroid', c)

    # aggregate the results
    aggrpool = estd.PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)


    # create the pool and the necessary algorithms
    pool = e.Pool()
    w = estd.Windowing()
    # spec = estd.Spectrum()
    # centroid = estd.SpectralCentroidTime()
    danceability = estd.Danceability(maxTau = 10000, minTau = 300, sampleRate = args.samplerate)
    
    # compute the centroid for all frames in our audio and add it to the pool
    for frame in estd.FrameGenerator(audio, frameSize = 10 * args.samplerate, hopSize = 5 * args.samplerate):
        dreal, ddfa = danceability(w(frame))
        print(("d", dreal)) # , "frame", frame
        pool.add('rhythm.danceability', dreal)

    print((type(pool['rhythm.danceability'])))
        
    # aggregate the results
    # aggrpool = estd.PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)
    
    # write result to file
    # estd.YamlOutput(filename = args.file + '.features.yaml')(aggrpool)

    fig, gs = makefig(rows = 2, cols = 2)
    ax = fig.axes

    ax[0].plot(pool['rhythm.danceability'])

    plt.show()

def main_segment(args):
    """perform sgementation of a piece of audio

    The idea is to identify different parts in a recording and
    annotate them for editing further down the pipeline.

    TODO
     - framesize hierarchy/ pyramid
     - pimp criteria with existing alternative criteria
     - pimp criteria with our own predictability criteria
     - clustering on spec frame pyramid
     - recurrence plot on feature frames
    """

    # FFT framesize and hopsize parameters
    frameSize = args.frame_size_low_level
    hopSize = frameSize / 2
    if args.hop_size_low_level is not None:
        hopSize = args.hop_size_low_level

    # load the audio
    audio = loadaudio(args)
    logger.debug('audio loaded, type = %s' % (audio.dtype))
    audio_labels = None

    # check if labels exist
    labelfile = args.file[:-4] + '_labels.txt'
    if os.path.exists(labelfile):
        audio_labels_t = np.genfromtxt(labelfile, delimiter='\t')
        audio_labels = (audio_labels_t[:,0] * args.samplerate) / hopSize
        logger.debug('labels = %s', audio_labels)
    # frame = audio

    # init window func
    w = estd.Windowing(type = 'hamming')
    # init spectrum
    spectrum = estd.Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum

    # feature operators
    features = {
        'BarkBands': {
            'op': estd.BarkBands,
            'opargs': {
                'numberBands': 12, # 28,
            },
            'opout': [0],
        },
        'ERBBands': {
            'op': estd.ERBBands,
            'opargs': {
                'numberBands': 12, # 40,
            },
            'opout': [0],
        },
        'MFCC':      {
            'op': estd.MFCC,
            'opargs': {
                'numberBands': 20, 'numberCoefficients': 10, 'highFrequencyBound': 10000, 'logType': 'dbamp', 'normalize': 'unit_sum', # 40 numbands
            },
            'opout': [1],
        },
        'GFCC':      {
            'op': estd.GFCC,
            'opargs': {
                'numberBands': 20, 'numberCoefficients': 10, 'highFrequencyBound': 10000, 'logType': 'dbamp', # 40 numberBands
            },
            'opout': [1],
        },
        'LPC':      {
            'op': estd.LPC,
            'opargs': {
                # 'order': 20, 'type': 'regular',
                'order': 8, 'type': 'regular',
            },
            'opout': [1],
        },
        'MelBands':  {
            'op': estd.MelBands,
            'opargs': {'numberBands': 10},
            'opout': [0],
        },
    }
    for fk, fv in list(features.items()):
        features[fk]['inst'] = fv['op'](**fv['opargs'])
        features[fk]['gram'] = []
        
        # # init mfcc features
        # mfcc = estd.MFCC()
    
    # segmentation operator
    sbic = estd.SBic(
        cpw = args.sbic_complexity_penalty_weight,
        inc1=args.sbic_inc1, inc2=args.sbic_inc2,
        minLength=args.sbic_minlength,
        size1=args.sbic_size1, size2=args.sbic_size2
    )
    
    # sbic = estd.SBic(cpw = 1.5, inc1 = 60, inc2 = 20, sbic_minlength = 10, size1 = 300, size2 = 200)
    # sbic = estd.SBic(cpw = 1.5, inc1 = 60, inc2 = 20, sbic_minlength = 80, size1 = 300, size2 = 200)
    # sbic = estd.SBic(cpw = 0.05, inc1 = 60, inc2 = 20, sbic_minlength = 120, size1 = 300, size2 = 200)
    # sbic = estd.SBic(cpw = 0.3, inc1 = 20, inc2 = 10, sbic_minlength = 10, size1 = 100, size2 = 70)

    # print "w", repr(w)
    # print "spectrum", repr(spectrum)
    # print "mfcc", repr(mfcc)

    # frame = audio[int(0.2*args.samplerate) : int(0.2*args.samplerate) + 1024]
    # print "frame.shape", frame.shape
    # spec = spectrum(w(frame))
    # mfcc_bands, mfcc_coeffs = mfcc(spec)
    
    pool = e.Pool()

    numframes = 0
    specgram = []
    # mfcc_bandsgram = []
    # mfcc_coefsgram = []
    logger.debug('main_segment: computing spec and features for audio of size %s', audio.shape)
    for frame in estd.FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize, startFromZero=True):
        # mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        # pool.add('lowlevel.mfcc', mfcc_coeffs)
        # pool.add('lowlevel.mfcc_bands', mfcc_bands)
        print(("frame", frame.shape))
        # frame = np.atleast_2d(frame)
        spec = spectrum(w(frame))
        specgram.append(spec)

        for fk, fv in list(features.items()):
            # logger.debug('computing feature %s', fk)
            fspec_ = fv['inst'](spec)
            # logger.debug('   type(fspec_) = %s', type(fspec_))
            fspec = fspec_
            
            if type(fspec_) is tuple:
                fspec = fspec_[fv['opout'][0]]

            fv['gram'].append(fspec)
            # mfcc_bands, mfcc_coefs = mfcc(spec)
            # mfcc_bandsgram.append(mfcc_bands)
            # mfcc_coefsgram.append(mfcc_coefs)

        numframes += 1
        if numframes % 10000 == 0:
            logger.debug('main_segment: crunched %d frames of shape %s', numframes, frame.shape)
    logger.debug('main_segment: crunched %d frames of shape %s', numframes, frame.shape)
    # sys.exit(0)

    specgram = np.array(specgram).T
    logger.debug("main_segment: %s-gram = %s", 'spec', specgram.shape)

    for fk, fv in list(features.items()):
        fv['gram'] = np.array(fv['gram']).T
        logger.debug("main_segment computing sbic for %s-gram = %s", fk, fv['gram'].shape)
        # mfcc_bandsgram = np.array(mfcc_bandsgram).T
        # mfcc_coefsgram = np.array(mfcc_coefsgram).T
        # print "segmenting mfcc_bandsgram", mfcc_bandsgram.shape
        # print "segmenting mfcc_coefsgram", mfcc_coefsgram.shape
        
        # segidx = sbic(specgram)
        # segidx = sbic(mfcc_bandsgram)
        # segidx = sbic(mfcc_coefsgram)
        fv['segidx'] = sbic(fv['gram'])
        # pool.add('segment.sbic', segidx)
        # logger.debug("    pool['segment.sbic'] = %s", pool['segment.sbic'])

        
        # logger.debug("%s seg indices[frame] = %s" % (fk, fv['segidx'], ))
        # logger.debug("       indices[time]  = %s" % ((fv['segidx'] * hopSize) / args.samplerate, ))
        logger.debug("%s seg |indices[frame]| = %s" % (fk, len(fv['segidx']), ))
        # logger.debug("       indices[time]  = %s" % ((fv['segidx'] * hopSize) / args.samplerate, ))
        # logger.debug("       framesize = %d, hopsize = %d" % (frameSize, hopSize))

    # copy spectrum into features_ for plotting
    features_ = features
    # features_['Spectrum'] = {'gram': specgram[1:40,...]}

    logger.debug('main_segment: starting plot of %d grams', len(features_))
    fig = plt.figure()
    fig.suptitle("part segmentation for %s with fs=%d, hs=%d" % (args.file.split('/')[-1], frameSize, hopSize))
    fig.show()
    gs = GridSpec(len(features_), 1)

    axi = 0
    for fk, fv in list(features_.items()):
        logger.debug('main_segment: plotting feature %s with shape %s', fk, fv['gram'].shape)
        ax = fig.add_subplot(gs[axi])
        ax.title.set_text(fk)
        ax.title.set_position((0.1, 0.9))
        ax.pcolormesh(fv['gram'])
        if 'segidx' in fv:
            ax.plot(fv['segidx'], fv['gram'].shape[0]/2 * np.ones_like(fv['segidx']), 'ro')
        if audio_labels is not None:
            ax.plot(audio_labels, (fv['gram'].shape[0]/2 + 1) * np.ones_like(audio_labels), 'go', alpha=0.7)
        if axi < (len(features) - 1): # all but last axis
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('feature-gram, framesize = %d' % (args.frame_size_low_level, ))
        axi += 1
    # ax1 = fig.add_subplot(3, 1,1)
    # ax1.pcolormesh(specgram)
    # # for segidx in fv['segidx']:
    # ax1.plot(fv['segidx'], specgram.shape[0]/2 * np.ones_like(fv['segidx']), 'ro')
    # ax2 = fig.add_subplot(3, 1,2)
    # ax2.pcolormesh(mfcc_bandsgram)
    # ax2.plot(fv['segidx'], mfcc_bandsgram.shape[0]/2 * np.ones_like(fv['segidx']), 'ro')
    # ax3 = fig.add_subplot(3, 1,3)
    
    # ax3 = fig.add_subplot(1, 1, 1)
    # ax3.pcolormesh(mfcc_coefsgram)
    # ax3.plot(fv['segidx'], mfcc_coefsgram.shape[0]/2 * np.ones_like(fv['segidx']), 'ro')

    logger.debug('main_segment: done plotting of %d grams', len(features_))
    plt.draw()
    plt.pause(1e-9)

    logger.debug('saving figure for %s' % (args.file, ))
    fig.set_size_inches((12, 3 * len(features)))
    fig.savefig('data/music_features/segment_%s.png' % (args.file.split('/')[-1]), dpi = 100, bbox_inches = 'tight')
    logger.debug('done saving, next')
    
def main_extractor(args):
    """main_extractor

    Compute the comprehensive extractor feature set over the input
    waveform and dump the pickled result into a file
    """
    audio = loadaudio(args)

    frame = audio

    extr = estd.Extractor(
        lowLevelFrameSize = args.frame_size_low_level,
        lowLevelHopSize   = int(args.frame_size_low_level/2),
        )

    # # compute the centroid for all frames in our audio and add it to the pool
    # for frame in estd.FrameGenerator(audio, frameSize = 10 * args.samplerate, hopSize = 5 * args.samplerate):
    #     dreal, ddfa = danceability(w(frame))
    #     print "d", dreal # , "frame", frame
    #     pool.add('rhythm.danceability', dreal)

    # compute feature of frame and put them into dict p
    p = extr(frame)

    pdict = {}
    # print "type(p)", type(p)
    for desc in p.descriptorNames():
        print((desc, type(p[desc])))
        #     print "{0: >20}: {1}".format(desc, p[desc])
        pdict[desc] = p[desc]

    pickle.dump(pdict, open("data/music_features_%s.pkl" % (util_filename_clean(args.file), ), "wb"))

def util_filename_clean(filename):
    # args.file.replace("/", "_")
    filename_1 = re.sub(r'[/\. ]', r'_', filename)
    filename_2 = filename.replace("/", "_")
    ret = filename_1
    print(("util_filename_clean ret = %s" % (ret, )))
    return ret
    
def main_extractor_pickle_plot(args):
    """main_extractor_pickle_plot

    Load a precomputed pickled feature dictionary and plot it
    """
    from matplotlib import rcParams
    rcParams['axes.titlesize'] = 7
    rcParams['axes.labelsize'] = 6
    rcParams['xtick.labelsize'] = 6
    rcParams['ytick.labelsize'] = 6
    # load bag of features computed above from a pickle
    pdict = pickle.load(open(args.file, "rb"))

    # get sorted keys
    feature_keys = sorted(pdict.keys())

    feature_groups = np.unique([k.split(".")[0] for k in feature_keys])

    # print np.unique(feature_keys_groups)

    feature_keys_groups = {}
    for group in feature_groups:
        feature_keys_groups[group] = [k for k in feature_keys if k.split(".")[0] == group]

    print((len(feature_keys_groups)))

    for group in feature_groups:
        fk_ = feature_keys_groups[group]
        plot_features(pdict, feature_keys = fk_, group = group)
    
    plt.show()


def plot_features(pdict, feature_keys, group):
    """util.plot_features

    Plot all features 'feature_keys' from a feature group 'group' in
    feature dict 'pdict'.
    """
    numrows = int(np.sqrt(len(feature_keys)))
    numcols = int(np.ceil(len(feature_keys)/float(numrows)))

    # print numrows, numcols
    
    fig = plt.figure()
    fig.suptitle(
        "essentia extractor %s features for %s" % (group, args.file, ),
        fontsize = 8)
    gs = GridSpec(numrows, numcols, hspace = 0.2)

    for i, k in enumerate(feature_keys):
        ax = fig.add_subplot(gs[i])
        if type(pdict[k]) is np.ndarray:
            ax.plot(pdict[k], alpha = 0.5, linewidth = 0.8, linestyle = "-")
            title = "%s/%s" % (k.split(".")[1], pdict[k].shape)
            if i/numcols < (numrows - 1):
                ax.set_xticklabels([])
        elif type(pdict[k]) is float:
            title = "%s/%s" % (k.split(".")[1], type(pdict[k]))
            ax.text(0, 0, "%f" % (pdict[k],), fontsize = 8)
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
        elif type(pdict[k]) is str:
            title = "%s/%s" % (k.split(".")[1], type(pdict[k]))
            ax.text(0, 0, "%s" % (pdict[k],), fontsize = 8)
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            
        ax.set_title(title)

    fig.show()

def main_extractor_pickle_plot_timealigned(args):
    """main_extractor_pickle_plot_timealigned

    Load a precomputed pickled feature dictionary, temporally align
    all of them in a matrix and a) plot them, b) compute dim-reducing
    feature transforms (PCA, KPCA, tsne) on the multivariate feature
    data and plot those.
    """
    from matplotlib import rcParams
    rcParams['axes.titlesize'] = 7
    rcParams['axes.labelsize'] = 6
    rcParams['xtick.labelsize'] = 6
    rcParams['ytick.labelsize'] = 6
    # load bag of features computed above from a pickle
    pdict = pickle.load(open(args.file, "rb"))

    # get sorted keys
    feature_keys = sorted(pdict.keys())

    # hack to get number of frames
    numframes = pdict['lowLevel.spectral_centroid'].shape[0]

    st = ()
    for ftkey in list(pdict.keys()):
        if type(pdict[ftkey]) is np.ndarray:
            # print ftkey, pdict[ftkey].shape
            if pdict[ftkey].shape[0] == numframes:
                tmp = pdict[ftkey]
                if len(tmp.shape) == 1:
                    tmp = tmp.reshape((numframes, 1))
                print((ftkey, "sum(tmp)", np.sum(np.abs(tmp)), tmp.shape))
                if np.sum(np.abs(tmp)) < 1e9:
                    st += (tmp, )
    # print "st", st
    full_time_aligned_raw = np.hstack(st)
    print(("full_time_aligned_raw", full_time_aligned_raw.shape))
    
    # clean up
    sum_raw = np.sum(np.abs(full_time_aligned_raw), axis = 0)
    print(("sum_raw", np.argmax(sum_raw), np.max(sum_raw)))
    var_raw = np.var(full_time_aligned_raw, axis = 0)
    # print "var_raw", var_raw != 0
    
    # scale / whiten
    full_time_aligned = scale(full_time_aligned_raw[...,var_raw != 0])
    print(("full_time_aligned", full_time_aligned.shape))
    
    # print "mean(full_time_aligned) = %s" % (np.mean(full_time_aligned, axis = 0), )
    # print "var(full_time_aligned) = %s" % (np.var(full_time_aligned, axis = 0), )

    plt.ion()
    
    fig2, gs2 = makefig(rows = 1, cols = 3)
    fig2.show()
    
    mu = np.mean(full_time_aligned, axis = 0)
    var = np.var(full_time_aligned, axis = 0)
    ax = fig2.add_subplot(gs2[0,0])
    ax.plot(mu, "bo", alpha = 0.5)
    ax.plot(var, "go", alpha = 0.5)
    # ax.plot(mu + var, "go", alpha = 0.5)
    # ax.plot(mu - var, "go", alpha = 0.5)

    ax2 = fig2.add_subplot(gs2[0,1])
    ax2.plot(full_time_aligned_raw)
    ax3 = fig2.add_subplot(gs2[0,2])
    ax3.plot(full_time_aligned)
    
    plt.draw()
    plt.pause(1e-9)
    
    # reduction
    numcomps = 5

    def decomp(X, algo = 'pca', numcomps = 3, datasig = 'data/ep1_wav'):
        
        filename_decomp = datasig + "_" + algo + ".npy"
        print(("filename_decomp", filename_decomp))

        TF = {
            'pca': PCA(n_components = numcomps, whiten = True),
            'kpca': KernelPCA(kernel="rbf", degree=5, fit_inverse_transform=True, gamma=10, n_components = numcomps),
            # 'tsne': TSNE(n_components=numcomps, random_state=0, verbose = 1),
            'sfa': SFA(numcomps = numcomps, numexps = 3),
            }
            
        if os.path.exists(filename_decomp):
            # load precomputed result
            X_ = np.load(filename_decomp)
        else:
            # compute
            # pca
            # pca   = PCA(n_components = numcomps)
            print(("Fitting with algo = %s" % (algo, )))
            X_ = TF[algo].fit_transform(X)
            # X_pca.transform(full_time_aligned)
            # print X_pca.transform(full_time_aligned).shape
            np.save('%s' % (filename_decomp, ), X_)
        return X_

    datasig_ = util_filename_clean(args.file)
    X_pca = decomp(X = full_time_aligned, algo = 'pca', numcomps = numcomps, datasig = datasig_)
    # X_kpca = decomp(X = X_pca, algo = 'kpca', numcomps = numcomps, datasig = datasig_)
    # X_tsne = decomp(X = X_pca, algo = 'tsne', numcomps = numcomps, datasig = datasig_)
    # X_sfa = decomp(X = X_pca, algo = 'sfa', numcomps = numcomps, datasig = datasig_)
    X_kpca = decomp(X = full_time_aligned, algo = 'kpca', numcomps = numcomps, datasig = datasig_)
    # X_tsne = decomp(X = full_time_aligned, algo = 'tsne', numcomps = numcomps, datasig = datasig_)
    X_sfa = decomp(X = full_time_aligned, algo = 'sfa', numcomps = numcomps, datasig = datasig_)

    # print "X_pca", X_pca.shape
    
    # pca
    # from sklearn.decomposition import PCA, KernelPCA
    # pca   = PCA(n_components = numcomps)
    # print "fitting pca"
    # X_pca = pca.fit_transform(full_time_aligned)
    # # X_pca.transform(full_time_aligned)
    # # print X_pca.transform(full_time_aligned).shape
    # np.save('%s', X_pca)

    # # kernel PCA
    # kpca = KernelPCA(
    #     kernel="rbf", degree=5, fit_inverse_transform=True,
    #     gamma=10,
    #     n_components = numcomps)
    # print "fitting kpca"
    # X_kpca = kpca.fit_transform(full_time_aligned)

    # # tsne
    # from sklearn.manifold import TSNE
    # tsne = TSNE(n_components=numcomps, random_state=0)
    # np.set_printoptions(suppress=True)
    # print "fitting tsne"
    # X_tsne = tsne.fit_transform(full_time_aligned)

    # plot
    fig, gs = makefig(numcomps, 3, title = '%s - pca, kpca, tsne' % (args.file, ), add_subplots = False)
    # fig.suptitle('pca, kpca, tsne')
    # gs = GridSpec(numcomps, 3)
    
    for i in range(numcomps):
        ax = fig.add_subplot(gs[i,0])
        ax.plot(X_pca[:,i])
        if i == 0: ax.set_title('PCA')
        ax.set_ylabel('c_%d' % (i, ))
        if i == (numcomps-1): ax.set_xlabel('t')


        ax = fig.add_subplot(gs[i,1])
        ax.plot(X_kpca[:,i])
        if i == 0: ax.set_title('KPCA')
        ax.set_ylabel('c_%d' % (i, ))
        if i == (numcomps-1): ax.set_xlabel('t')
        
        # ax = fig.add_subplot(gs[i,2])
        # ax.plot(X_tsne[:,i])
        # if i == 0: ax.set_title('TSNE')
        # ax.set_ylabel('c_%d' % (i, ))
        # if i == (numcomps-1): ax.set_xlabel('t')
            
        ax = fig.add_subplot(gs[i,2])
        ax.plot(X_sfa[:,i])
        if i == 0: ax.set_title('SFA')
        ax.set_ylabel('c_%d' % (i, ))
        if i == (numcomps-1): ax.set_xlabel('t')
        # ax.set_s

    plt.ioff()
    plt.show()
    
    # tsne

    # feature_groups = np.unique([k.split(".")[0] for k in feature_keys])

    # print "feature_groups", feature_groups
    # # print np.unique(feature_keys_groups)

    # feature_keys_groups = {}
    # for group in feature_groups:
    #     feature_keys_groups[group] = [k for k in feature_keys if k.split(".")[0] == group]

    # print len(feature_keys_groups)

    # for group in feature_groups:
    #     ftkeys = feature_keys_groups[group]

    #     for ftkey in ftkeys:
    #         print "%s-%s" % (group, ftkey)
        
    # plot_features(pdict, feature_keys = fk_, group = group)
    
    # plt.show()


def main_print_file_info(args):
    """Walk the data directory and print path, type, length (samples),
    samplerate for mp3 files.

    Computes the input file for end-to-start feature matching mixing
    algorithm used in farmers manual's guest mix @sorbierd 2017
    """
    import os
    import taglib

    print(("print_file_info: walking %s directory" % (args.datadir, )))
    for datadir_item in os.walk(args.datadir):
        # print "    type(f) = %s\n    f = %s\n" % (type(f), f[0], ),
        for datafile in datadir_item[2]:
            datafile_path = '%s/%s' % (datadir_item[0], datafile)
            tlf = taglib.File(datafile_path)
            tlf_type = tlf.path.split('.')[-1]
            if tlf_type != 'mp3': continue
            if tlf.length < 60.0: continue
            
            # print "        datafile = %s" % (datafile, )
            # print "            tags = %s" % tlf.tags
            #       path  type  numframes
            print(("    ('%s', '%s', %d, %d)," % (tlf.path, tlf_type, tlf.length * tlf.sampleRate, tlf.sampleRate)))
            
def main_mix(args):
    """Render the final mix into a single bitstream file from ordered
    track list using pydub.

    Computes the output file for end-to-start feature matching mixing
    algorithm used in farmers manual's guest mix @sorbierd 2017
    """
    from pydub import AudioSegment
    # f = open('trk_seq_559.txt', 'r')
    # print "args.file", args.file
    assert type(args.file) is str, "main_mix assumes args.file to be singular string, not %s" % (type(args.file), )
    f = open(args.file, 'r')
    trk_seq_raw = "".join(f.readlines())


    gv = {}
    lv = {}
    code = compile(trk_seq_raw, "<string>", "exec")
    exec(code, gv, lv)

    trk_seq = lv['trk_seq']

    # mix = AudioSegment.empty()

    silence_thresh = -40
    
    print("trk_seq")
    for i, tup in enumerate(trk_seq):
        print(("trk", i, tup[0], tup[1]))
        
        if i < 1:
            print("just appending")
            mix = AudioSegment.from_mp3(tup[1]) # .strip_silence()
            print((" mix dur", mix.duration_seconds))
            mix = mix.strip_silence(silence_len = 1500, silence_thresh = silence_thresh, padding = 60)
            print((" mix dur", mix.duration_seconds))
        
        else:
            xfadetime = np.random.randint(1500, 3000)
            print(("appending with xfade = %s ms" % (xfadetime, )))
            trk_ = AudioSegment.from_mp3(tup[1])
            print(("trk_ dur", trk_.duration_seconds))
            trk_ = trk_.strip_silence(silence_len = 1500, silence_thresh = silence_thresh, padding = 60)
            print(("trk_ dur", trk_.duration_seconds))
            mix = mix.append(trk_, crossfade = xfadetime)
            
        print((" mix dur", mix.duration_seconds))

    mix_fin = mix.fade_in(1000).fade_out(1000)
    
    mix_fin.export(
        "mix_fin.mp3",
        format = "mp3",
        bitrate = '320k',
        tags={'artist': 'farmersmanual (DJ)', 'title': 'the mix', 'album': 'fm playlist selection for sorbie rd. @subcity radio', 'comments': 'This album is awesome!'})

def main_single(args):
    if args.mode == "mfcc":
        main_mfcc(args)
    elif args.mode == "simple":
        main_simple(args)
    elif args.mode == "danceability":
        main_danceability(args)
    elif args.mode == "segment":
        main_segment(args)
    elif args.mode == "extractor":
        main_extractor(args)
    elif args.mode == "extractor_plot":
        main_extractor_pickle_plot(args)
    elif args.mode == "extractor_plot_timealigned":
        main_extractor_pickle_plot_timealigned(args)
    elif args.mode == "print_file_info":
        main_print_file_info(args)
    elif args.mode == "mix":
        main_mix(args)
    else:
        logger.error('Unknown mode %s', args.mode)
    
def main_files(args):

    plt.ion()
    
    files = args.file[0]
    for file in files:
        logger.debug("main_files: running mode %s on file %s" % (args.mode, file))
        args_ = copy.copy(args)
        setattr(args_, 'file', file)
        # main_segment(args_)
        main_single(args_)
        
    plt.ioff()
    plt.show()
    
if __name__ == "__main__":
    modes = ['mfcc', 'danceability', 'extractor', 'extractor_plot', 'simple', 'segment', 'extractor_plot_timealigned', 'print_file_info']
    modes.sort()
    default_frame_size_low_level = 2048
    parser = argparse.ArgumentParser()
    parser.add_argument("-scpw", "--sbic-complexity-penalty-weight", help="SBic param cpw [1.5]", type=float, default=1.5)
    parser.add_argument("-sml", "--sbic-minlength", help="SBic param sbic_minlength [10]", type=int, default=10)
    parser.add_argument("-sinc1", "--sbic-inc1", type=int, default=60, help="SBic param sbic_inc1 [60]")
    parser.add_argument("-sinc2", "--sbic-inc2", type=int, default=20, help="SBic param sbic_inc2 [20]")
    parser.add_argument("-ssize1", "--sbic-size1", type=int, default=300, help="SBic param sbic_size1 [300]")
    parser.add_argument("-ssize2", "--sbic-size2", type=int, default=200, help="SBic param sbic_size2 [200]")
    parser.add_argument("-d", "--datadir", help="Data directory [.]", type = str, default = ".")
    # parser.add_argument("-f", "--file", help="Input file [data/ep1.wav]", type = str, default = "data/ep1.wav")
    parser.add_argument("-f", "--file", action = 'append', dest = 'file', help="Input file(s) []", nargs = '+', default = [])
    parser.add_argument("-fsl", "--frame-size-low-level", help="Framesize for low-level features [%d]" % (default_frame_size_low_level,), type = int, default = default_frame_size_low_level)
    parser.add_argument("-hsl", "--hop-size-low-level", help="Hopsize for low-level features, [frame-size-low-level/2]", type = int, default = None)
    parser.add_argument("-m", "--mode", help="Program mode [mfcc]: %s" % ", ".join(modes), type = str, default = "mfcc")
    parser.add_argument("-sr", "--samplerate", help="Sample rate to use [44100]", type = int, default = 44100)

    args = parser.parse_args()

    main_files(args)
