"""detect onsets, beats, segments in audio

Starting from librosa "onset times from a signal" example, extended a
bit with additional variations and algorithms.

Using librosa, madmom, essentia

## todo
- persistent engine
- iterative / interactive run
- incremental graph expansion
- bayesian ensembles for event constraints and integration
"""
import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import madmom

from scipy.stats import mode
from smp_base.plot import make_fig

DEBUG=True

from slurp.common import myprint
from slurp.common import data_load_librosa
from slurp.common import compute_chroma_librosa, compute_tempogram_librosa
from slurp.common import compute_onsets_librosa, compute_beats_librosa, compute_beats_madmon
from slurp.common import compute_segments_librosa

def plotit(**kwargs):
    ############################################################
    # plotting
    y = kwargs['y']
    o_env = kwargs['o_env']
    times = kwargs['times']
    onset_frames = kwargs['onset_frames']
    tempo = kwargs['tempo']
    beats = kwargs['beats']
    dtempo = kwargs['dtempo']
    tempo2 = kwargs['tempo2']
    beats2 = kwargs['beats2']
    dtempo2 = kwargs['dtempo2']
    chroma = kwargs['chroma']
    bounds = kwargs['bounds']
    bound_times = kwargs['bound_times']
    mm_beat_times = kwargs['mm_beat_times']
    file = kwargs['file']
    
    # fig = plt.figure()
    # fig.suptitle('%s' % (file))
    # fig_numrow = 5

    ax1 = plt.subplot(fig_numrow, 1, 1)
    myplot_specshow_librosa(ax1, y)
    
    ax2 = plt.subplot(fig_numrow, 1, 2, sharex=ax1)
    myplot_onsets(ax2, times, o_env, onset_frames)
    
    ax3 = plt.subplot(fig_numrow, 1, 3, sharex=ax1)
    # myplot_beats(beattimes, ylow, yhigh, alpha, color, linestyle, label)
    # plt.plot(times, librosa.util.normalize(o_env),
    #          label='Onset strength', alpha=0.33)
    for beat_ in [('beats', 0.5, 1.0, 0.5, 'b', '--', 'Beats'),
                  ('beats2', 0.0, 0.5, 0.5, 'k', '-.', 'Beats2'),
                  ('mm_beat_times', 0.25, 0.75, 'r', '-', 'mm_beats')]:
        # beattimes = times[kwargs[beat_[0]]]
        myplot_beats(ax, times[kwargs[beat_[0]]], ylow=beat_[1], yhigh=beat_[2], alpha=beat_[3],
                     color=beat_[4], linestyle=beat_[5], label=beat_[6])
        
        # plt.vlines(times[beats], 0.5, 1, alpha=0.5, color='b',
        #        linestyle='--', label='Beats')
        # plt.vlines(times[beats2], 0, 0.5, alpha=0.5, color='k',
        #        linestyle='-.', label='Beats2')
        # plt.vlines(mm_beat_times, 0.25, 0.75, alpha=0.75, color='r',
        #        linestyle='-', linewidth=2, label='mm beats')
    
    ax4 = plt.subplot(fig_numrow, 1, 4, sharex=ax1)
    myplot_tempo(ax4, times, dtempo)
    myplot_tempo(ax4, times, dtempo2)
    # plt.plot(times, dtempo, alpha=0.5, color='b',
    #            linestyle='none', marker='o', label='Tempo')
    # plt.plot(times, dtempo2, alpha=0.5, color='k',
    #            linestyle='none', marker='o', label='Tempo2')

    
    ax5 = plt.subplot(fig_numrow, 1, 5, sharex=ax1)
    myplot_segments(ax5, chroma, bound_times)    
    
    # plt.axis('tight')
    plt.legend(frameon=True, framealpha=0.75)

def main_segmentation_iter_clust(args):
    pass

def main(args):
    """main beat detection"""
    # load data from file
    y, sr, filename = data_load_librosa(args)
    
    # myprint('Computing onsets')
    # onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    # librosa.frames_to_time(onset_frames, sr=sr)
    
    # # Or use a pre-computed onset envelope
    # # array([ 0.07 ,  0.395,  0.511,  0.627,  0.766,  0.975,
    # # 1.207,  1.324,  1.44 ,  1.788,  1.881])

    chroma = compute_chroma_librosa(y, sr)
    # tempogram = compute_tempogram_librosa(y, sr, onset_env)
    
    # compute onsets
    onset_env, onset_times_ref, onset_frames = compute_onsets_librosa(y, sr)

    # FIXME: is there a beat? danceability?
    # compute_beat(onset_env, onset_frames)

    # compute beat tracking (librosa)
    beats = {}
    # for start_bpm in [30, 60, 90]:
    for start_bpm in [120]:
        t_, dt_, b_ = compute_beats_librosa(onset_env, onset_frames, start_bpm, sr)
        # b_f_ = librosa.util.fix_frames(b_, x_max=chroma.shape[1])
        beatsk = 'lr_bpm{0}'.format(start_bpm)
        beats[beatsk] = {}
        beats[beatsk]['tempo'] = t_
        beats[beatsk]['dtempo'] = dt_
        beats[beatsk]['beats'] = b_
        # beats[beatsk]['beats_f'] = b_f_

    # Csync = librosa.util.sync(chroma, b_f_, aggregate=np.median)
    
    # # compute beat tracking (madmom)
    # t_, dt_, b_ = compute_beats_madmon(None, None, None, sr, filename)
    # beats['mm'] = {}
    # beats['mm']['tempo'] = t_
    # beats['mm']['dtempo'] = dt_
    # beats['mm']['beats'] = b_

    # part segmentation
    # 1. how many parts?
    # 1.1 iterate clustering
    # 1.2 information criterion
    # 1.3 recurrence_plot edge detect

    numparts = 5
    segments = {}
    # for numparts in [5, 10]: # range(2, 20):
    for numparts in [6,7]:
        bd_, bdt_, bds_ = compute_segments_librosa(chroma, sr, numparts)
        segments[numparts] = {}
        myprint('bounds', bd_)
        segments[numparts]['bounds'] = bd_
        segments[numparts]['bound_times'] = bdt_ # wrong: is frames
        # segments[numparts]['bound_times'] = librosa.frames_to_time(bdt_) # wrong: is frames
        segments[numparts]['bound_samples'] = bds_
        segments[numparts]['bounds_hist'] = np.histogram(np.diff(segments[numparts]['bound_times']), bins=20)
        segments[numparts]['bounds_mode'] = mode(np.diff(segments[numparts]['bound_times']))
        # myprint('segments %s' % (numparts), librosa.frames_to_time(segs, sr=sr))

    from essentia.standard import SBic
    sbic = SBic(cpw=0.75, inc1=60, inc2=20, minLength=10, size1=200, size2=300)
    # myprint('chroma.T.tolist()', chroma.T.tolist())
    segs = sbic(chroma.tolist())
    segments['sbic'] = {}
    # myprint('bounds', bd_)
    segments['sbic']['bounds'] = segs
    segments['sbic']['bound_times'] = librosa.frames_to_time(segs, sr=sr)
    segments['sbic']['bound_samples'] = librosa.frames_to_samples(segs)
    segments['sbic']['bounds_hist'] = np.histogram(np.diff(segments['sbic']['bound_times']), bins=20)
    segments['sbic']['bounds_mode'] = mode(np.diff(segments[numparts]['bound_times']))
    myprint('segments sbic', librosa.frames_to_time(segs, sr=sr))

    # FIXME: compute distance of sound frames / segments to each other, use dist matrix for selecting numsegments

    # bound_samples = segments[numparts]['bound_samples']
    bound_samples = segments['sbic']['bound_samples']
    for i in range(len(bound_samples)):
        i_start = bound_samples[i-1]
        if i < 1:
            i_start = 0
            
        tmp_ = y[i_start:bound_samples[i]]
        outfilename = filename[:-4] + "-%d.wav" % (i)
        myprint('writing seg %d to outfile %s' % (i, outfilename))
        librosa.output.write_wav(outfilename, tmp_, sr)
    
    # bound_times
    # array([  0.   ,   1.672,   2.322,   2.624,   3.251,   3.506,
    #      4.18 ,   5.387,   6.014,   6.293,   6.943,   7.198,
    #      7.848,   9.033,   9.706,   9.961,  10.635,  10.89 ,
    #     11.54 ,  12.539])


    myprint('Plotting')
    colors = ['k', 'b', 'g', 'r', 'm', 'c']
    fig = make_fig(rows=6, cols=1, title='%s' % (filename))
    myplot_specshow_librosa(fig.axes[0], y)
    myplot_onsets(fig.axes[1], onset_times_ref, onset_env, onset_frames)
    for i, beat_ in enumerate(beats):
        beats_ = beats[beat_]['beats']
        # [('beats', 0.5, 1.0, 0.5, 'b', '--', 'Beats'),
        #  ('beats2', 0.0, 0.5, 0.5, 'k', '-.', 'Beats2'),
        #  ('mm_beat_times', 0.25, 0.75, 'r', '-', 'mm_beats')]:
        if beat_.startswith('lr'):
            beattimes = onset_times_ref[beats_]
        else:
            beattimes = beats_
        myplot_beats(fig.axes[2], beattimes, ylow=0, yhigh=1, alpha=0.5,
                     color=colors[i], linestyle='--', label=beat_)
        if beats[beat_]['dtempo'] is not None:
            myplot_tempo(fig.axes[3], onset_times_ref, beats[beat_]['dtempo'])

    myplot_chroma(fig.axes[4], chroma)
    myplot_segments(fig.axes[4], chroma, segments['sbic']['bound_times'], color='r')
    # myplot_chroma_sync(fig.axes[4], Csync, librosa.frames_to_time(beats['lr_bpm120']['beats_f'], sr=sr))
    # myplot_segments(fig.axes[4], Csync, segments['sbic']['bound_times'], color='r')
    for i, seg_ in enumerate(segments):
        if type(seg_) is str: continue
        bd_hist_ = segments[seg_]['bounds_hist']
        bd_hist_ = segments[seg_]['bounds_mode']
        # boundidx = librosa.frames_to_time(segments[seg_]['bound_times'])
        # boundidx = segments[seg_]['bound_times']
        # myprint('boundidx', boundidx)
        # myplot_segments(fig.axes[4], Csync, boundidx, color=colors[i%len(colors)])
        myplot_segments(fig.axes[4], chroma, segments[seg_]['bound_times'], color=colors[i%len(colors)])
        myplot_segments_hist(fig.axes[5], bd_hist_, idx=i, color=colors[i%len(colors)])
        
    fig.axes[5].set_yscale('log')

    # plotit(y=y, o_env=o_env, times=times, onset_frames=onset_frames, file=args.file,
    #        tempo=tempo, beats=beats, dtempo=dtempo, 
    #        tempo2=tempo2, beats2=beats2, dtempo2=dtempo2,
    #        chroma=chroma, bounds=bounds, bound_times=bound_times,
    #        mm_beat_times=mm_beat_times,
    # )

    # # more segmentation with recurrence_matrix
    # winsize = int(2**14)
    # hopsize = int(winsize/4)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=winsize, hop_length=hopsize)
    # R = librosa.segment.recurrence_matrix(mfcc)
    # R_aff = librosa.segment.recurrence_matrix(mfcc, mode='affinity')
    # myprint('R = %s, R_aff = %s' % (R.shape, R_aff.shape))

    # fig2 = plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # librosa.display.specshow(R, x_axis='time', y_axis='time')
    # plt.title('Binary recurrence (symmetric)')
    # plt.subplot(1, 2, 2)
    # librosa.display.specshow(R_aff, x_axis='time', y_axis='time', cmap='magma_r')
    # plt.title('Affinity recurrence')
    # plt.tight_layout()
    
    plt.show()
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--duration', help='Input duration (secs) to select from input file [10.0]',
                        default=10.0, type=float)
    parser.add_argument('-f', '--file', help='Sound file to process', default=None, type=str)

    args = parser.parse_args()

    main(args)
    
