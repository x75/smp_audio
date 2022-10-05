"""autofold

Oswald Berthold, 2022

autofold, satisfy entropy criterion by folding an audio recording
(timeseries) on itself until threshold reached
"""
import argparse, random, sys, os
import librosa
from pydub import AudioSegment
# from librosa import samples_to_time
# import pydub
# import pippi
import numpy as np
import soundfile

def main(args):
    # load soundfile
    # get length of soundfile
    # get consecutive slices of equal length
    # mix them: static, with modulated amplitude
    # write output file

    if args.filename is None:
        print("No filename given, exiting")
        sys.exit()

    # y, sr = librosa.load(args.filename, sr=48000, mono=False)
    # # y = np.atleast_2d(y)
    # # y, sr = soundfile.read(args.filename, always_2d=True)
    # print(f"File loaded with y {y.shape}, sr = {sr}")

    # y_len_s = librosa.samples_to_time([y.shape[0]], sr)
    # print(f"File length(s) = {y_len_s}, {y_len_s/60}")

    # audio_segment = AudioSegment(
    #     y.tobytes(),
    #     frame_rate=sr,
    #     sample_width=y.dtype.itemsize,
    #     channels=2
    # )

    audio_segment = AudioSegment.from_mp3(args.filename)

    print(f"audio_segment {audio_segment.duration_seconds} {audio_segment.channels} {audio_segment}")

    slicelen = audio_segment.duration_seconds / args.numfolds
    print(f"creating {args.numfolds} slices with length(ms) {slicelen}")
    
    # split sound1 in 5-second slices
    slices = audio_segment[::int(slicelen*1000)]

    slices = list(slices)
    
    print(f"slices {type(slices)}")

    for i, slice_ in enumerate(slices):
        print(f"slice {i} {slice_}")

    folded = []

    gain_factor = -0
    
    for i, slice_ in enumerate(slices):
        # fade_in_time = np.random.randint(0, int(slicelen*1000))
        # fade_out_time = int(slicelen*1000) - fade_in_time - 10
        if i == 0:
            folded.append(slice_.apply_gain(gain_factor))
        else:
            fade_in_time = np.random.randint(int(slicelen*1000)/4, int(slicelen*1000)/2)
            fade_out_time = np.random.randint(int(slicelen*1000)/100, int(slicelen*1000)/10)
            print(f"slice {i} {slice_}, fade_in_time {fade_in_time} fade_out_time {fade_out_time}")
            folded.append(folded[-1].overlay(
                slice_.fade(from_gain=-120.0, start=0, duration=fade_in_time).fade(to_gain=-120.0, end=0, duration=fade_out_time),
                gain_during_overlay=gain_factor))
        print(f"folded {folded}")

    output_filename = f"{os.path.dirname(args.filename)}/autofold.mp3"
    print(f"exporting to {output_filename}")
    file_handle = folded[-1].export(output_filename, format="mp3")
        
    # louder_via_method = sound1.apply_gain(-6)
    
    # # multichannel
    # if y.shape[1] > 1:
    #     y = y.mean(axis=1, keepdims=True)
    #     print(f"File downmix y {y.shape}, sr = {sr}")

    # for i in range(args.numslices):
    #     offset = np.random.randint(0, y.shape[0] - args.duration - 1)
    #     print(f"Sampling offset {offset}")
    #     y_new = y[offset:offset+args.duration]
    #     print(f"Sampling slice at offset {y_new.shape}")

    #     # write slice
    #     filename_new = args.filename[:-4] + f"-slice-{offset}.wav"
    #     print(f"Writing file to {filename_new}")
    #     soundfile.write(filename_new, y_new, sr, 'PCM_16', format='WAV')

if __name__ == '__main__':
    print(sys.argv)
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--conf', help='Config key to load from autovoice configuration module [sco_2]', default='sco_2', type=str)
    # parser.add_argument('-d', '--duration', help='Output duration (samples) to select from input file [4096]',
    #                     default=4096, type=int)
    parser.add_argument('-f', '--filename', help='Sound file to process', default=None, type=str)
    parser.add_argument('-n', '--numfolds', help='Number of folds to perform [1]', default=1, type=int)
    parser.add_argument('-s', '--seed', help='Random seed [0]', default=0, type=int)

    args = parser.parse_args()
    print(args)
    main(args)
    
