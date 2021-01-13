"""audio_features_paa

audio feature extraction from pyAudioAnalysis (paa) package

- 2021-01-13 updated to paa 0.3.6
"""
import argparse
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures as mF
import matplotlib.pyplot as plt
import numpy as np

def compute_features_paa(filename, with_timebase=False, verbose=False):
    """compute_features_paa

    Compute a bag of standard audio features to be used for some
    downstream task.
    """
    if verbose:
        print('compute_features_paa loading from {0}'.format(filename))
    [Fs, x_] = audioBasicIO.read_audio_file(filename)
    if verbose:
        print('compute_features_paa: loaded {1} samples from {0}'.format(filename, x_.shape))
    if len(x_.shape) > 1 and x_.shape[1] > 1:
        x = audioBasicIO.stereo_to_mono(x_)
    else:
        x = x_
    x_duration = x.shape[0]/Fs
    if verbose:
        print(f'compute_features_paa: {x_duration} seconds of audio at {Fs}Hz')

    mt_win = 1.0*Fs
    mt_step = 0.5*Fs
    st_win = 0.050*Fs
    st_step = 0.025*Fs
    # F, F_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, st_win, st_step)
    # G, F, F_names = audioFeatureExtraction.mtFeatureExtraction(x, Fs, mt_win, mt_step, st_win, st_step)
    G, F, F_names = mF.mid_feature_extraction(x, Fs, mt_win, mt_step, st_win, st_step)

    if with_timebase:
        G_time = np.linspace(0, G.shape[1] * 0.5, G.shape[1] + 1)
        F_time = np.linspace(0, F.shape[1] * 0.025, F.shape[1] + 1)
    else:
        G_time = None
        F_time = None

    if verbose:
        print(f'compute_features_paa: F = {F.shape} {F}')
        print(f'compute_features_paa:     {F_time}')
        print(f'compute_features_paa: G = {G.shape} {G}')
        print(f'compute_features_paa:     {G_time}')

    if with_timebase:
        return F, F_names, G, F_time, G_time
    else:
        return F, F_names, G
    
def plot_features_st_paa(F, F_names, filename):
    num_st = 8
    for p in range(num_st):
        plt.subplot(num_st,1,p+1)
        plt.plot(F[p,:])
        plt.xlabel('Frame no')
        plt.ylabel(F_names[p]); 
    plt.title('{0}'.format(filename.split('/')[-1]))

def plot_features_mt_paa(G, filename):
    fig2 = plt.figure()
    num_mt_ = int(G[0].shape[0]/2)
    num_mt = 8
    for p in range(num_mt):
        ax = fig2.add_subplot(num_mt, 1, p+1)
        ax.plot(G[0][p,:])
        ax.plot(G[0][p,:] + G[0][(p + num_mt_),:], c='r', alpha=0.5)
        ax.plot(G[0][p,:] - G[0][(p + num_mt_),:], c='r', alpha=0.5)
        ax.set_xlabel('10-Frames no')
        ax.set_ylabel(G[2][p])
    fig2.suptitle('{0}'.format(filename.split('/')[-1]))

    # plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]);

def main(args):
    
    F, F_names, G = compute_features_paa(args.filename)

    plot_features_st_paa(F, F_names, args.filename)

    plot_features_mt_paa(G, args.filename)
    
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default="/home/src/QK/data/sound-arglaaa-2018-10-25/22-mono.wav")

    args = parser.parse_args()
    
    main(args)
