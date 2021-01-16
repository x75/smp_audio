# coding: utf-8
import base64, argparse, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import acoustid as ai

# filename = '/home/lib/audio/work/tsx_recur_3/trk001-7.mp3'
# filename = '/home/lib/audio/work/tsx_recur_1/release/rc3/1going_export8a_n0.wav'

# af = ai.audioread.audio_open('/home/lib/audio/work/tsx_recur_3/trk001-7.mp3')

def fingerprint_file(filename):
    # fp is tuple with (duration, fingerprint)
    fp = ai.fingerprint_file(filename)

    # get fingerprint for modification
    fp_bytes = fp[1]

    # pad fingerprint
    padlen = (len(fp_bytes) % 4)
    if padlen < 1: padlen = 0
    padstr = b'=' * padlen
    print(('padstr {0} {1} {2}'.format(len(fp_bytes), padlen, padstr)))
    fp_bytes += padstr

    # decode to list of char
    fp_int = base64.b64decode(fp_bytes)

    # decode to list of int32s
    fb_bin = [list('{:032b}'.format(abs(x))) for x  in fp_int]

    # allocate array
    arr = np.zeros([len(fb_bin), len(fb_bin[0])])

    # copy to array
    for i in range(arr.shape[0]):
        arr[i,0] = int(fp_int[i] > 0) # The sign is added to the first bit
        for j in range(1, arr.shape[1]):
            arr[i,j] = float(fb_bin[i][j])

    return arr

def fingerprint_plot(fps):
    numfps = len(fps)
    
    fig = plt.figure()
    gs = GridSpec(2, int(np.ceil(numfps/2)))

    for i, k in enumerate(fps):
        ax = fig.add_subplot(gs[i])
        ax.imshow(fps[k]['fingerprint'].T, aspect='auto', origin='lower')
        ax.set_title('Binary repr chromaprint {0}'.format(k))
        
    return fig

def main(args):
    filenames = args.filenames
    if type(filenames[0]) is list:
        filenames = filenames[0]
        
    numfiles = len(filenames)
    if numfiles < 1:
        print('No filenames supplied, exiting')
        sys.exit(1)

    fps = {}
        
    print(('filenames {0}'.format(filenames)))
    for filename in filenames:
        print(('filename {0}'.format(filename)))
        filename_key = filename.split('/')[-1][:-4]
        fps[filename_key] = {}
        fps[filename_key]['path'] = filename
        fps[filename_key]['fingerprint'] = fingerprint_file(filename)

    fig_fp = fingerprint_plot(fps)

    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filenames", action = 'append', dest = 'filenames', help="Input file(s) []", nargs = '+', default = [])

    args = parser.parse_args()

    main(args)
