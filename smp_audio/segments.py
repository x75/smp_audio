"""smp_audio.segments

time, event, segment related computations
"""

from pprint import pformat
import os
from scipy import signal
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile

def compute_event_mean_intervals(**kwargs):
    """compute_event_mean_intervals

    Compute the mean segement lengths/intervals for a set of event
    list like beats and parts.
    """
    # collect the input into local variables
    
    # number of frames, FIXME winsize, hopsize
    numframes = kwargs['numframes']
    # list of beat positions arrays
    beats = kwargs['beats']
    # list of segment boundary position arrays
    segs = kwargs['segs']
    # verbosity
    verbose = kwargs['verbose']

    if verbose:
        print('    mean_intervals beats', pformat(len(beats)))
        print('    mean_intervals segs', pformat(len(segs)))
        print('    mean_intervals numframes', pformat(numframes))
    
    # # list of average expected interval per single beat estimate
    # for b_ in beats:
    #     # print('beat len {1} = {0}'.format('b_', len(b_)))
    #     pass
    
    # valid interval lists have more than one element
    beats_interval = list([np.mean(np.diff(beats[i])) for i in range(len(beats)) if len(beats[i]) > 1])
    # valid interval lists indices
    beats_i = list([i for i in range(len(beats)) if len(beats[i]) > 1])
    # list of average segment length per single segment estimate
    segs_interval = [np.mean(np.diff(segs[i])) for i in range(len(segs))]
    # print('segs_interval = {0}\nbeats_interval = {1}'.format(segs_interval, beats_interval))

    # combined list of intervals
    intervals = segs_interval + beats_interval

    numinputs = len(intervals)

    # print('intervals = {0}\nnuminputs = {1}'.format(intervals, numinputs))

    final = np.zeros((numframes, numinputs))
    for i in range(len(segs)):
        # print(f'    segs[{i}] = {segs[i]}, {len(segs)}')
        final[:,i][segs[i][:-1]] = 1
    # use only valid indices
    for i in range(len(beats_i)):
        # print(f'    beat i={i}, beats[{i}]')
        final[:,i+len(segs)][beats[i]] = 1
    
    return {
        'intervals': intervals,
        'numinputs': numinputs,
        'final': final
    }

def compute_event_merge_mexhat(**kwargs):
    """compute_event_merge_mexhat

    Compute merged events with superimposed mexican-hat function
    activation per event

    Less frequent events like segment boundaries are given more
    weight.
    """
    # collect the input into local variables
    
    numframes = kwargs['numframes']
    numinputs = kwargs['numinputs']
    intervals = kwargs['intervals']
    final = kwargs['final']
    numsegs = kwargs['numsegs']
    verbose = kwargs['verbose']
    
    # final initialize empty
    # final_ = np.zeros((numframes, 10))
    final_ = np.zeros((numframes, numinputs))

    # kernel setup

    # scale_factor = 25
    # norm_factor = 3
    scale_factor = 10
    norm_factor = 3

    # # compute gaussian kernels using scale and norm factors
    # kernels_gaussian = []
    # for i in range(numinputs):
    #     tmp_ = norm.pdf(np.arange(-intervals[i]/ norm_factor, intervals[i]/ norm_factor), loc=0, scale=intervals[i]/ scale_factor)
    #     kernels_gaussian.append(tmp_)

    # number kernel points
    numpoints = min(250, numframes)
    if verbose:
        print(f'    mexhat numpoints = {numpoints}')
    kernels_mexhat = [signal.ricker(points=numpoints, a=intervals[i]/10.0) for i in range(numinputs)]

    # set kernels to mexhat
    kernels = kernels_mexhat
    # print('kernels = {0}'.format(kernels))

    # compute the kernel weights
    kernel_weights_raw = intervals
    kernel_weights_max = np.max(kernel_weights_raw)
    # kernel_weights = [1 for _ in range(numinputs)]
    # kernel_weights = [1 - _/10 for _ in range(numinputs)]
    kernel_weights = [1 + _ for _ in range(numinputs)]
    # kernel_weights = [1/(0.25*_+1) for _ in range(numinputs)]
    # kernel_weights = [kernel_weights_raw[i]/kernel_weights_max for i in range(numinputs)]
    # print('kernel_weights_raw {0}\nkernel_weights {1}'.format(kernel_weights_raw, kernel_weights))

    # compute combined activations for all inputs
    for i in range(numinputs):
        # print(f'    event_merge_mexhat numinput {i}/{numinputs}, {final_.shape}')
        final_[:,i] = np.convolve(final[:,i], kernels[i], mode='same')

    # compute final output, the beat aligned segment boundaries
    final_sum = np.sum(final_, axis=1)
    ind = np.argpartition(final_sum, -numsegs)[-numsegs:] - 2
    # ind2 = np.argpartition(final_sum, -(numsegs+5))[-(numsegs+5):] - 2
    return {
        'kernels': kernels,
        'kernel_weights': kernel_weights,
        'final_sum': final_sum,
        'ind': ind,
        # 'ind2': ind2,
    }

def plot_event_merge_results(**kwargs):
    """plot_event_merge_results

    Plot the event merge results, used for debugging
    """
    # collect input into local variables
    numframes = kwargs['numframes']
    kernels = kwargs['kernels']
    kernel_weights = kwargs['kernel_weights']
    final_sum = kwargs['final_sum']
    ind = kwargs['ind']
    ind_full = kwargs['ind_full']
    # ind2 = kwargs['ind2']
    verbose = kwargs['verbose']

    # create the plot
    # plot the kernels
    plt.subplot(2,1,2)
    for i, kernel in enumerate(kernels):
        kernel /= np.max(np.abs(kernel))
        kernel *= kernel_weights[i]
        plt.plot(kernel)

    # plot the weighted kernel sum
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

    # return figure handle?
    return {'figure': plt.gcf()}

def compute_event_merge_heuristics(**kwargs):
    """compute_event_merge_heuristics

    Compute event merge using some heuristics
    """
    # collect input into local variables
    numframes = kwargs['numframes']
    ind = kwargs['ind']
    seglen_min = 43 # 10
    # ind2 = kwargs['ind2']
    verbose = kwargs['verbose']

    # sort indices, ind2 has lower treshold / more events than ind(1)
    ind.sort()
    # ind2.sort()

    # debug printing
    if verbose:
        print('    compute_event_merge_heuristics: ind {0}\n        ind2 {1}'.format(ind, None)) # ind2

    # minimum length heuristic (brute force)
    ind_full = np.array([0] + ind.tolist() + [numframes-1])
    # print('ind_full {0}'.format(ind_full))
    # idx = np.array([True] + (np.diff(ind_full)>10).tolist()).astype(bool)
    idx = np.diff(ind_full) > seglen_min
    # print('idx {0}'.format(idx))
    # print('idx {0}, ind_full {1}'.format(np.sum(idx), ind_full.shape))
    ind_ = [0] + ind_full[1:][idx].tolist()

    # ind2_full = np.array([0] + ind2.tolist() + [numframes-1])
    # print('ind2_full {0}'.format(ind2_full))
    # # idx = np.array([True] + (np.diff(ind_full)>10).tolist()).astype(bool)
    # idx2 = np.diff(ind2_full)>10
    # print('idx2 {0}'.format(idx2))
    # ind2_ = [0] + ind2_full[1:][idx2].tolist()
    
    # print('ind_ {0}\nind2 {1}'.format(ind_, None)) # ind2_

    return {
        'ind_': ind_,
        'ind_full': ind_full.tolist(),
        # 'ind2_': ind2_,
    }

def compute_event_merge_index_to_file(**kwargs):
    """compute_event_merge_index_to_file

    Compute event merge mapping from the indices to the file
    """
    # collect input into local variables    
    ind_ = kwargs['ind_']
    # ind2_ = kwargs['ind2_']
    filename_48 = kwargs['filename_48']
    verbose = kwargs['verbose']
    
    if verbose:
        print(f'    compute_event_merge_index_to_file filename_48 = {filename_48}')
    
    # ind_cut = librosa.frames_to_samples(ind)
    # ind_cut = librosa.frames_to_samples(ind.tolist() + [numframes-1])

    # convert frame indices to sample indices
    ind_cut = librosa.frames_to_samples(ind_) # FIXME: hardcoded choice ind2 (smaller segments / more events)

    # load high-quality data
    y_48, sr_48 = librosa.load(filename_48, sr=None, mono=False)
    # print('y_48 {0}, sr_48 {1}'.format(y_48, sr_48))
    # assert more than one sample in the file and stereo channels (FIXME)
    assert len(y_48.shape) > 1 and y_48.shape[0] == 2
    
    # convert sample indices for samplerate
    # FIXME: global parameters analysis-rate 22050, input-file-rate 48000
    ind_cut_48 = (ind_cut * (48000/22050)).astype(int)
    ind_cut_48.sort()

    # compute basedir for writing segment files to disk
    filename_48_dir = os.path.dirname(filename_48)
    filename_48_dir_data = os.path.join(filename_48_dir, 'data')
    filename_48_base = os.path.basename(filename_48)
    filename_48_base_list = filename_48_base.split('.')
    filename_48_base_name = ".".join(filename_48_base_list[:-1])
    filename_48_base_type = filename_48_base_list[-1]
    suflen = len(filename_48_base_type)+1
    filename_48_dir_data_segs = os.path.join(filename_48_dir, 'data/segs')
    if not os.path.exists(filename_48_dir_data_segs):
        # os.makedirs(filename_48_dir_data)
        os.makedirs(filename_48_dir_data_segs)
            
    if verbose:
        print('segments.compute_event_merge_index_to_file')
        print(f'    filename_48_dir {filename_48_dir}')
        print(f'    filename_48_dir_data {filename_48_dir_data}')
        print(f'    filename_48_dir_data {filename_48_dir_data_segs}')
        print(f'    filename_48_base {filename_48_base}')
        print(f'    filename_48_base_list {filename_48_base_list}')
        print(f'    filename_48_base_name {filename_48_base_name}')
        print(f'    filename_48_base_type {filename_48_base_type}')
        print(f'    suflen {suflen}')

    # loop over segments, grab the data and write into a wav file
    ret = []
    for i in range(1, len(ind_cut_48)):
        # if i < 1:
        #     i_start = 0
        # else:
        i_start = ind_cut_48[i-1]

        # temporary segment buf
        tmp_ = y_48[:,i_start:ind_cut_48[i]]

        # compute filename for segment file
        # outfilename = 'data/' + filename_48[:-4] + "-seg-%d.wav" % (i)
        # outfilename =  f"{filename_48_dir}/data/{filename_48_base_name}-seg-{i}.wav"
        outfilename =  f"{filename_48_base_name}-seg-{i}.wav"
        outfilename =  os.path.join(
            filename_48_dir_data_segs,
            outfilename
        )
        
        if verbose:
            print('segments.compute_event_merge_index_to_file\n    writing seg %d to outfile %s' % (i, outfilename))
            
        # librosa.output.write_wav(outfilename, tmp_, sr_48)
        soundfile.write(outfilename, tmp_.T, sr_48, 'PCM_16', format='WAV')
        ret.append(outfilename)

    # return list of segment filenames created
    return {'files': ret}

def compute_event_merge_combined(**kwargs):
    """compute_event_merge_combined

    Compute the event merge combining beats and segment events
    """
    # collect input into local variables    
    
    # number of frames, FIXME winsize, hopsize
    numframes = kwargs['numframes']
    verbose = kwargs['verbose']

    # get mean intervals and number of event sequences to merge
    tmp_ = compute_event_mean_intervals(**kwargs)
    kwargs.update(tmp_)
    intervals = tmp_['intervals']
    numinputs = tmp_['numinputs']
    final = tmp_['final']

    # compute combined weighted kernel activations for all event streams
    tmp_ = compute_event_merge_mexhat(**kwargs)
    kwargs.update(tmp_)
    kernels = tmp_['kernels']
    final_sum = tmp_['final_sum']
    ind = tmp_['ind']
    # ind2 = tmp_['ind2']

    # compute additional heuristics to apply on the combined events
    tmp_ = compute_event_merge_heuristics(**kwargs)
    kwargs.update(tmp_)

    # write the final segments out to files
    # kwargs['filename_48'] = '/home/src/QK/data/sound-arglaaa-2018-10-25/22.wav'
    tmp_ = compute_event_merge_index_to_file(**kwargs)

    # # debug plotting
    # plot_event_merge_results(**kwargs)

    # if verbose:
    #     print(('compute_event_merge_combined write files {0}'.format(tmp_)))

    # return the dict
    return tmp_
