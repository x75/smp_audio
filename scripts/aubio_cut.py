#! /usr/bin/env python

""" this file was written by Paul Brossier
  it is released under the GNU/GPL license.

 - 2020-09-01 Extended original aubio_cut script with threshold
   scanning method
"""

import sys
from aubio.cmd import AubioArgumentParser

from multiprocessing import Pool
    
from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

def aubio_cut_parser():
    parser = AubioArgumentParser()
    parser.add_input()
    # operation mode
    parser.add_argument("-m","--mode",
            action="store", dest="mode", default='default',
            metavar = "<mode>",
            help="program operation mode [default=default] \
                    default|scan")
    parser.add_argument("-O","--onset-method",
            action="store", dest="onset_method", default='default',
            metavar = "<onset_method>",
            help="onset detection method [default=default] \
                    complexdomain|hfc|phase|specdiff|energy|kl|mkl")
    # cutting methods
    parser.add_argument("-b","--beat",
            action="store_true", dest="beat", default=False,
            help="slice at beat locations")
    """
    parser.add_argument("-S","--silencecut",
            action="store_true", dest="silencecut", default=False,
            help="use silence locations")
    parser.add_argument("-s","--silence",
            metavar = "<value>",
            action="store", dest="silence", default=-70,
            help="silence threshold [default=-70]")
            """
    # algorithm parameters
    parser.add_buf_hop_size()
    parser.add_argument("-t","--threshold", "--onset-threshold",
            metavar = "<threshold>", type=float,
            action="store", dest="threshold", default=0.3,
            help="onset peak picking threshold [default=0.3]")
    parser.add_argument("-c","--cut",
            action="store_true", dest="cut", default=False,
            help="cut input sound file at detected labels")
    parser.add_minioi()

    """
    parser.add_argument("-D","--delay",
            action = "store", dest = "delay", type = float,
            metavar = "<seconds>", default=0,
            help="number of seconds to take back [default=system]\
                    default system delay is 3*hopsize/samplerate")
    parser.add_argument("-C","--dcthreshold",
            metavar = "<value>",
            action="store", dest="dcthreshold", default=1.,
            help="onset peak picking DC component [default=1.]")
    parser.add_argument("-L","--localmin",
            action="store_true", dest="localmin", default=False,
            help="use local minima after peak detection")
    parser.add_argument("-d","--derivate",
            action="store_true", dest="derivate", default=False,
            help="derivate onset detection function")
    parser.add_argument("-z","--zerocross",
            metavar = "<value>",
            action="store", dest="zerothres", default=0.008,
            help="zero-crossing threshold for slicing [default=0.00008]")
    # plotting functions
    parser.add_argument("-p","--plot",
            action="store_true", dest="plot", default=False,
            help="draw plot")
    parser.add_argument("-x","--xsize",
            metavar = "<size>",
            action="store", dest="xsize", default=1.,
            type=float, help="define xsize for plot")
    parser.add_argument("-y","--ysize",
            metavar = "<size>",
            action="store", dest="ysize", default=1.,
            type=float, help="define ysize for plot")
    parser.add_argument("-f","--function",
            action="store_true", dest="func", default=False,
            help="print detection function")
    parser.add_argument("-n","--no-onsets",
            action="store_true", dest="nplot", default=False,
            help="do not plot detected onsets")
    parser.add_argument("-O","--outplot",
            metavar = "<output_image>",
            action="store", dest="outplot", default=None,
            help="save plot to output.{ps,png}")
    parser.add_argument("-F","--spectrogram",
            action="store_true", dest="spectro", default=False,
            help="add spectrogram to the plot")
    """
    parser.add_slicer_options()
    parser.add_verbose_help()
    return parser


def _cut_analyze(options):
    hopsize = options.hop_size
    bufsize = options.buf_size
    samplerate = options.samplerate
    source_uri = options.source_uri

    # analyze pass
    from aubio import onset, tempo, source

    s = source(source_uri, samplerate, hopsize)
    if samplerate == 0:
        samplerate = s.get_samplerate()
        options.samplerate = samplerate

    if options.beat:
        o = tempo(options.onset_method, bufsize, hopsize, samplerate=samplerate)
    else:
        o = onset(options.onset_method, bufsize, hopsize, samplerate=samplerate)
        if options.minioi:
            if options.minioi.endswith('ms'):
                o.set_minioi_ms(int(options.minioi[:-2]))
            elif options.minioi.endswith('s'):
                o.set_minioi_s(int(options.minioi[:-1]))
            else:
                o.set_minioi(int(options.minioi))
    o.set_threshold(options.threshold)

    timestamps = []
    total_frames = 0
    while True:
        samples, read = s()
        if o(samples):
            timestamps.append (o.get_last())
            if options.verbose: print ("%.4f" % o.get_last_s())
        total_frames += read
        if read < hopsize: break
    del s
    return timestamps, total_frames

def _cut_slice(options, timestamps):
    # cutting pass
    nstamps = len(timestamps)
    if nstamps > 0:
        # generate output files
        from aubio.slicing import slice_source_at_stamps
        timestamps_end = None
        if options.cut_every_nslices:
            timestamps = timestamps[::options.cut_every_nslices]
            nstamps = len(timestamps)
        if options.cut_until_nslices and options.cut_until_nsamples:
            print ("warning: using cut_until_nslices, but cut_until_nsamples is set")
        if options.cut_until_nsamples:
            timestamps_end = [t + options.cut_until_nsamples for t in timestamps[1:]]
            timestamps_end += [ 1e120 ]
        if options.cut_until_nslices:
            timestamps_end = [t for t in timestamps[1 + options.cut_until_nslices:]]
            timestamps_end += [ 1e120 ] * (options.cut_until_nslices + 1)
        slice_source_at_stamps(options.source_uri,
                timestamps, timestamps_end = timestamps_end,
                output_dir = options.output_directory,
                samplerate = options.samplerate)

def main_default(options):

    # analysis
    timestamps, total_frames = _cut_analyze(options)
        
    # print some info
    duration = float (total_frames) / float(options.samplerate)
    base_info = '%(source_uri)s' % {'source_uri': options.source_uri}
    base_info += ' (total %(duration).2fs at %(samplerate)dHz)\n' % \
            {'duration': duration, 'samplerate': options.samplerate}

    info = "found %d timestamps in " % len(timestamps)
    info += base_info
    sys.stderr.write(info)

    return {
        'timestamps': timestamps,
        'total_frames': total_frames,
        'duration': duration,
        'base_info': base_info,
        'info': info,
    }

# def _cut_analyze_with_opts(options):
#     timestamps, total_frames = cut_analyze_cached(options)
#     return timestamps, total_frames

def scan_with_onset_method(options):
    
    cut_analyze_cached = memory.cache(_cut_analyze)

    # thr_init = 0.05
    thr_init = 0.2
    thr_incr = 0.05
    # thr_init = 3.00
    # thr_incr = 0.01
    thr = thr_init
    segcnt = 1e6

    thrs = []
    segcnts = []

    while segcnt > 1:
        options.threshold = thr
        
        timestamps, total_frames = cut_analyze_cached(options)
        segcnt = len(timestamps)
        print(options.onset_method, thr, segcnt)
        thrs.append(thr)
        segcnts.append(segcnt)
        thr += thr_incr

    return thrs, segcnts

def main_scan(options):
    import numpy as np
    import matplotlib.pyplot as plt
    
    print('options', options)

    analyze_runs = {}


    # def f(x):
    #     return x*x
    
    # options_list = []
    # for onset_method_ in ['default', 'complexdomain', 'hfc', 'phase', 'specdiff', 'energy', 'kl', 'mkl']:
    #     options_ = copy(options)
    #     options_.onset_method = onset_method
    #     options_list.append(options_)

    #     # if __name__ == '__main__':
    # with Pool(len(options_list)) as p:
    #     # print(p.map(f, [1, 2, 3]))
    #     print(p.map(f, [1, 2, 3]))
            
    for onset_method_ in ['default', 'complexdomain', 'hfc', 'phase', 'specdiff', 'energy', 'kl', 'mkl']:
        options.onset_method = onset_method_
        thrs, segcnts = scan_with_onset_method(options)
        analyze_runs[onset_method_] = {'thr': thrs, 'seg': segcnts, 'onset_method': onset_method_}

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i,k in enumerate(analyze_runs):
        # ax = fig.add_subplot(len(analyze_runs),1,i+1)
        ax.plot(analyze_runs[k]['thr'], analyze_runs[k]['seg'], '-o', label=analyze_runs[k]['onset_method'], alpha=0.5)
        # ax.set_title(analyze_runs[k]['onset_method'])

    ax.set_title('aubio_cut scan num-segs / threshold / method')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('threshold')
    ax.set_ylabel('num. segments')
    plt.legend()
    plt.show()
        
    # # print some info
    # duration = float (total_frames) / float(options.samplerate)
    # base_info = '%(source_uri)s' % {'source_uri': options.source_uri}
    # base_info += ' (total %(duration).2fs at %(samplerate)dHz)\n' % \
    #         {'duration': duration, 'samplerate': options.samplerate}

    # info = "found %d timestamps in " % len(timestamps)
    # info += base_info
    # sys.stderr.write(info)

    # return {
    #     'timestamps': timestamps,
    #     'total_frames': total_frames,
    #     'duration': duration,
    #     'base_info': base_info,
    #     'info': info,
    # }
    
def main():
    parser = aubio_cut_parser()
    options = parser.parse_args()
    if not options.source_uri and not options.source_uri2:
        sys.stderr.write("Error: no file name given\n")
        parser.print_help()
        sys.exit(1)
    elif options.source_uri2 is not None:
        options.source_uri = options.source_uri2

    # print(options)

    if options.mode == 'default':
       ret = main_default(options)
    elif options.mode == 'scan':
       ret = main_scan(options)
        
    if options.cut:
        _cut_slice(options, timestamps)
        info = "created %d slices from " % len(timestamps)
        info += base_info
        sys.stderr.write(info)

if __name__ == '__main__':
    main()
