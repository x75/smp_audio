"""smp_audio.main

2020-08-25 main finding: segmentation cannot be run on frame based
processing without additional work

main script for smp_audio functions: segment_scan, beat_detect,
measures_to_sort, autoedit, autovoice, automix

scripts:
 - aubio_cut.py with scanning
 - audio_beat_detect_2.py with autovoice
 - audio_classify.py with frame based processing and keras model
 - audio_sort_features.py with automix, autoedit

tasks:
 - common main
 - tasks in funclib
 - switchable frame based / batch processing
 - control backend libs: librosa, aubio, essentia, madmom, paa
 - consider to be called from react via websocket / flask
 - create test for backend functions

 - consolidate framesize, blocksize, hopsize, hoplength
 - 
"""
import argparse, pprint, time

import wave
import librosa

from smp_audio.common_librosa import data_stream_librosa, data_stream_get_librosa
from smp_audio.common_aubio import data_stream_aubio, data_stream_get_aubio

from smp_audio.common_librosa import compute_chroma_librosa, compute_segments_librosa
# from smp_audio.common_essentia import compute_segments_essentia

# backend = 'librosa'
backend = 'aubio'

data_streams = {
    'librosa': {
        'open': data_stream_librosa,
        'read': data_stream_get_librosa,
    },
    'aubio': {
        'open': data_stream_aubio,
        'read': data_stream_get_aubio,
    }
}

def main_timing_read_stream(filename, src, sr, framesize, args):
    # global data_streams
    print(f'main_timing_read_stream args = {args}')

    # 1 open file, stream, data-stream/live-src
    # 2 scan the frames and print timing info

    # 3 handle multiple filenames
    # 4 handle multiple samplerates

    X = []
    Y = []
    start = time.time()
    # while src read next
    # for blk_i, blk_y in enumerate(src):
    for blk_i, blk_y_tup in enumerate(data_streams[backend]['read'](src)):
        blk_y = blk_y_tup[0]
        blk_y_len = blk_y_tup[1]
        # src_len_blks = src.duration/blk_y_len
        # print(f'i {blk_i}/{src_len_blks}, y {blk_y.shape}')
        # print(f'i {blk_i}, y {blk_y.shape}')

        if len(blk_y) < framesize: continue
        # D_block = librosa.stft(blk_y, n_fft=framesize, hop_length=framesize, center=False)
        # print(f'D_block {D_block.shape}')
        # D.append(D_block[:,0])
        y = blk_y
        # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        # rmse = librosa.feature.rms(y=y, frame_length=framesize, hop_length=framesize, center=False)
        # spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        # spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        # rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        # zcr = librosa.feature.zero_crossing_rate(y, frame_length=framesize, hop_length=framesize, center=False)
        # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        # print(f'    timing_read_stream y {y.shape}, {chroma_stft.shape}')
    
    end = time.time()
    time_taken = end - start
    print(('\nThe function took {:.2f} s to compute.'.format(time_taken)))
    return {'mode': args.mode, 'filename': filename, 'time': time_taken}
    
def main_autoedit_stream(filename, src, sr, framesize, args):
    """autoedit_stream is an extended aubio_cut

    it generates several different segmentations and finds overlapping
    boundaries.

    issues:
    - timing reference: samples, frames, seconds
    """
    print(f'main_autoedit_stream args = {args}\n    length = {args.srclength/framesize}')
    start = time.time()
    # while src read next
    # for blk_i, blk_y in enumerate(src):
    for blk_i, blk_y_tup in enumerate(data_streams[backend]['read'](src)):
        blk_y = blk_y_tup[0]
        blk_y_len = blk_y_tup[1]
        # src_len_blks = src.duration/blk_y_len
        # print(f'i {blk_i}/{src_len_blks}, y {blk_y.shape}')
        print(f'i {blk_i}, y {blk_y.shape}')

        if len(blk_y) < framesize: continue
        y = blk_y
        
        # insert autoedit_stream computation graph
        # 1 get features for m in method
        chromagram = compute_chroma_librosa(blk_y, sr, hop_length=framesize)['chromagram']
        # 2 get segments for m in method
        # 2.1 m = essentia, broken with essentia frame processing model
        # segments1 = compute_segments_essentia(blk_y, sr, 10)
        # 2.2 m = librosa
        segments2 = compute_segments_librosa(blk_y, sr, 10)
        
        # 3 get onsets for m in method
        # 4 get beat for m in method
        # 5 probabilistic overlap combination
        # 6 assemble
    
    end = time.time()
    time_taken = end - start
    print(('\nThe function took {:.2f} s to compute.'.format(time_taken)))
    return {'mode': args.mode, 'filename': filename, 'time': time_taken}    

def get_src(filename, args):
    # filename = args.filenames[0][filename_i]
    print(f'filename = {filename}')
    # load audio data
    framesize = 1024

    srcfile = wave.open(filename, mode='rb')
    srcparams = srcfile.getparams()
    length = srcparams[3] # pos 3 is nframes

    src, sr = data_streams[backend]['open'](
        filename=filename,
        frame_length=framesize,
        hop_length=framesize,
        num_channels=1,
    )
    # print(f'sr={sr}')
    return src, sr, framesize, filename, length

def main(args):
    # print(f'main args = {args}')
    if args.mode == 'timing_read_stream':
        _main = main_timing_read_stream
    elif args.mode == 'autoedit_stream':
        _main = main_autoedit_stream

    ret = []
    for filename_i, filename in enumerate(args.filenames[0]):
        print(f'main running {args.mode} on {filename}')
        src, sr, framesize, filename, args.srclength = get_src(filename, args)
        ret.append(_main(filename, src, sr, framesize, args))

    print(f'ret = {pprint.pformat(ret)}')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # filenames array
    # parser.add_argument('-f', '--filename', type=str, help="Audio file to analyze [None]", default=None)
    parser.add_argument("-f", "--filenames", action='append', dest='filenames', help="Input file(s) []", nargs = '+', default = [])
    # mode: modes
    parser.add_argument('-m', '--mode', type=str, help="Processing mode [timing_read_stream]", default='timing_read_stream')
    # remove
    # parser.add_argument('-m', '--modelfile', type=str, help="Model file to load [model.json]", default='model.json')
    args = parser.parse_args()

    main(args)
