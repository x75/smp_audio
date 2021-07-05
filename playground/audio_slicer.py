import argparse, random, sys, os
# import librosa
import numpy as np
import soundfile

def main(args):
    y, sr = soundfile.read(args.filename, always_2d=True)
    print(f"File loaded with y {y.shape}, sr = {sr}")

    # multichannel
    if y.shape[1] > 1:
        y = y.mean(axis=1, keepdims=True)
        print(f"File downmix y {y.shape}, sr = {sr}")

    for i in range(args.numslices):
        offset = np.random.randint(0, y.shape[0] - args.duration - 1)
        print(f"Sampling offset {offset}")
        y_new = y[offset:offset+args.duration]
        print(f"Sampling slice at offset {y_new.shape}")

        # write slice
        filename_new = args.filename[:-4] + f"-slice-{offset}.wav"
        print(f"Writing file to {filename_new}")
        soundfile.write(filename_new, y_new, sr, 'PCM_16', format='WAV')

if __name__ == '__main__':
    print(sys.argv)
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--conf', help='Config key to load from autovoice configuration module [sco_2]', default='sco_2', type=str)
    parser.add_argument('-d', '--duration', help='Output duration (samples) to select from input file [4096]',
                        default=4096, type=int)
    parser.add_argument('-f', '--filename', help='Sound file to process', default=None, type=str)
    parser.add_argument('-n', '--numslices', help='Number of slices [1]', default=1, type=int)
    parser.add_argument('-s', '--seed', help='Random seed [0]', default=0, type=int)

    args = parser.parse_args()
    print(args)
    main(args)
    
