"""smp_audio.segments

time, event, segment computations
"""

from pprint import pformat
from scipy import signal
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import librosa

def compute_event_mean_intervals(**kwargs):
    # number of frames, FIXME winsize, hopsize
    numframes = kwargs['numframes']
    # list of beat positions arrays
    beats = kwargs['beats']
    # list of segment boundary position arrays
    segs = kwargs['segs']
    
    print('beats', pformat(beats))
    print('segs', pformat(segs))
    print('numframes', pformat(numframes))
    
    # list of average expected interval per single beat estimate
    beats_interval = list([np.mean(np.diff(beats[i])) for i in range(len(beats))])
    # list of average segment length per single segment estimate
    segs_interval = [np.mean(np.diff(segs[i])) for i in range(len(segs))]
    # combined list of intervals
    intervals = segs_interval + beats_interval

    numinputs = len(intervals)

    print('intervals = {0}, numinputs = {1}'.format(intervals, numinputs))

    final = np.zeros((numframes, numinputs))
    for i in range(len(segs)):
        print('i', i)
        final[:,i][segs[i]] = 1
    for i in range(len(beats)):
        print('i', i)
        final[:,i+len(segs)][beats[i]] = 1
    
    return {'intervals': intervals, 'numinputs': numinputs, 'final': final}

def compute_event_merge_mexhat(**kwargs):
    numframes = kwargs['numframes']
    numinputs = kwargs['numinputs']
    intervals = kwargs['intervals']
    final = kwargs['final']
    numsegs = kwargs['numsegs']
    
    # final empty
    final_ = np.zeros((numframes, 10))

    # scale_factor = 25
    # norm_factor = 3
    scale_factor = 10
    norm_factor = 3

    kernels_gaussian = []
    for i in range(numinputs):
        tmp_ = norm.pdf(np.arange(-intervals[i]/ norm_factor, intervals[i]/ norm_factor), loc=0, scale=intervals[i]/ scale_factor)
        kernels_gaussian.append(tmp_)
    
    kernels_mexhat = [signal.ricker(points=250, a=intervals[i]/10.0) for i in range(numinputs)]

    kernels = kernels_mexhat
    # print('kernels = {0}'.format(kernels))

    kernel_weights_raw = intervals
    kernel_weights_max = np.max(kernel_weights_raw)
    # kernel_weights = [1 for _ in range(numinputs)]
    # kernel_weights = [1 - _/10 for _ in range(numinputs)]
    kernel_weights = [1 + _ for _ in range(numinputs)]
    # kernel_weights = [1/(0.25*_+1) for _ in range(numinputs)]
    # kernel_weights = [kernel_weights_raw[i]/kernel_weights_max for i in range(numinputs)]

    print('kernel_weights_raw {0}\nkernel_weights {1}'.format(kernel_weights_raw, kernel_weights))

    for i in range(numinputs):
        final_[:,i] = np.convolve(final[:,i], kernels[i], mode='same')

    final_sum = np.sum(final_, axis=1)
    ind = np.argpartition(final_sum, -numsegs)[-numsegs:] - 2
    # ind2 = np.argpartition(final_sum, -(numsegs+5))[-(numsegs+5):] - 2

    return {'kernels': kernels, 'kernel_weights': kernel_weights, 'final_sum': final_sum, 'ind': ind} # , 'ind2': ind2}

def plot_event_merge_results(**kwargs):
    numframes = kwargs['numframes']
    kernels = kwargs['kernels']
    kernel_weights = kwargs['kernel_weights']
    final_sum = kwargs['final_sum']
    ind = kwargs['ind']
    ind_full = kwargs['ind_full']
    # ind2 = kwargs['ind2']
    
    plt.subplot(2,1,2)
    for i, kernel in enumerate(kernels):
        kernel /= np.max(np.abs(kernel))
        kernel *= kernel_weights[i]
        plt.plot(kernel)

    plt.subplot(2,1,1)
    plt.plot(final_sum, linewidth=1.0, alpha=0.5)
    ax = plt.gca()
    linescale = np.max(final_sum)
    ax.vlines(ind, 0.5 * linescale, 1.0 * linescale, 'r', linewidth=1)
    plt.draw()
    # ax.vlines(ind2, 0 * linescale, 0.5 * linescale, 'g', linewidth=2)
    ax.vlines(ind_full, 0 * linescale, 0.5 * linescale, 'g', linewidth=2)
    ax.set_xlim((0, numframes))

    plt.draw()
    # plt.show()

    return {}

def compute_event_merge_heuristics(**kwargs):
    numframes = kwargs['numframes']
    ind = kwargs['ind']
    seglen_min = 43 # 10
    # ind2 = kwargs['ind2']

    # sort indices, ind2 has lower treshold / more events than ind(1)
    ind.sort()
    # ind2.sort()

    # debug printing
    print('ind {0}\nind2 {1}'.format(ind, None)) # ind2

    # min length heuristic brute force
    ind_full = np.array([0] + ind.tolist() + [numframes-1])
    print('ind_full {0}'.format(ind_full))
    # idx = np.array([True] + (np.diff(ind_full)>10).tolist()).astype(bool)
    idx = np.diff(ind_full) > seglen_min
    print('idx {0}'.format(idx))
    print('idx {0}, ind_full {1}'.format(np.sum(idx), ind_full.shape))
    ind_ = [0] + ind_full[1:][idx].tolist()

    # ind2_full = np.array([0] + ind2.tolist() + [numframes-1])
    # print('ind2_full {0}'.format(ind2_full))
    # # idx = np.array([True] + (np.diff(ind_full)>10).tolist()).astype(bool)
    # idx2 = np.diff(ind2_full)>10
    # print('idx2 {0}'.format(idx2))
    # ind2_ = [0] + ind2_full[1:][idx2].tolist()
    
    print('ind_ {0}\nind2 {1}'.format(ind_, None)) # ind2_

    return {'ind_': ind_, 'ind_full': ind_full.tolist()} # , 'ind2_': ind2_}

def compute_event_merge_index_to_file(**kwargs):
    ind_ = kwargs['ind_']
    # ind2_ = kwargs['ind2_']
    filename_48 = kwargs['filename_48']
    
    # ind_cut = librosa.frames_to_samples(ind)
    # ind_cut = librosa.frames_to_samples(ind.tolist() + [numframes-1])

    # convert frame indices to sample indices
    ind_cut = librosa.frames_to_samples(ind_) # FIXME: hardcoded choice ind2 (smaller segments / more events)

    # load high-quality data
    y_48, sr_48 = librosa.load(filename_48, sr=None, mono=False)
    print('y_48 {0}, sr_48 {1}'.format(y_48, sr_48))
    assert len(y_48.shape) > 1 and y_48.shape[0] == 2
    
    # convert sample indices for samplerate
    ind_cut_48 = (ind_cut * (48000/22050)).astype(int)
    ind_cut_48.sort()

    # loop over segments, grab the data and write int wav
    ret = []
    for i in range(1, len(ind_cut_48)):
        # if i < 1:
        #     i_start = 0
        # else:
        i_start = ind_cut_48[i-1]
        
        tmp_ = y_48[:,i_start:ind_cut_48[i]]
        outfilename = filename_48[:-4] + "-seg-%d.wav" % (i)
        print('writing seg %d to outfile %s' % (i, outfilename))
        librosa.output.write_wav(outfilename, tmp_, sr_48)
        ret.append(outfilename)

    return {'files': ret}

def compute_event_merge_combined(**kwargs):
    # number of frames, FIXME winsize, hopsize
    numframes = kwargs['numframes']

    # get mean intervals and number of event sequences to merge
    tmp_ = compute_event_mean_intervals(**kwargs)
    kwargs.update(tmp_)
    intervals = tmp_['intervals']
    numinputs = tmp_['numinputs']
    final = tmp_['final']

    tmp_ = compute_event_merge_mexhat(**kwargs)
    kwargs.update(tmp_)
    kernels = tmp_['kernels']
    final_sum = tmp_['final_sum']
    ind = tmp_['ind']
    # ind2 = tmp_['ind2']

    tmp_ = compute_event_merge_heuristics(**kwargs)
    kwargs.update(tmp_)

    # kwargs['filename_48'] = '/home/src/QK/data/sound-arglaaa-2018-10-25/22.wav'
    tmp_ = compute_event_merge_index_to_file(**kwargs)

    plot_event_merge_results(**kwargs)

    print(('write files {0}'.format(tmp_)))

    return tmp_
