import argparse
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt

def compute_features_paa(filename):
    print('Loading from {0}'.format(filename))
    [Fs, x_] = audioBasicIO.readAudioFile(filename)
    print('Loaded from {0} data x_ = {1}'.format(filename, x_.shape))
    if len(x_.shape) > 1 and x_.shape[1] > 1:
        x = audioBasicIO.stereo2mono(x_)
    else:
        x = x_

    # F, F_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
    G, F, F_names = audioFeatureExtraction.mtFeatureExtraction(x, Fs, 1.0*Fs, 0.5*Fs, 0.050*Fs, 0.025*Fs)
    # F = G[1]

    print(('F = {0}'.format(F)))
    print(('G = {0}, {1}'.format(len(G), G)))

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
