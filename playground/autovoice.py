"""autovoice

take a beat and a voice reocrding each. slice the voices into
snippets, process the snippets, and create a new track by placing a
main unprocessed voice track and a background processed voice track
over the original track in sync with the beat.

 - Started audio_beat_detect_2.py: detect onsets, beats, segments in audio
 - Starting from librosa "onset times from a signal" example, extended a bit with additional variations and algorithms.
 - Using librosa, madmom, essentia

## TODO
- persistent engine (general, smp_audio)
- iterative / interactive run (general, smp_audio)
- incremental graph expansion (general, smp_audio)
- bayesian ensembles for event constraints and integration (general, smp_audio)
"""
import argparse, random, sys
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
# import madmom

from scipy.stats import mode
import soundfile
# from smp_base.plot import make_fig
print("post import")

DEBUG=True
HEADLESS=False

from smp_audio.common import myprint
from smp_audio.common_librosa import data_load_librosa
from smp_audio.common_librosa import compute_chroma_librosa, compute_tempogram_librosa
from smp_audio.common_librosa import compute_onsets_librosa, compute_beats_librosa
from smp_audio.common_librosa import compute_segments_librosa
# from smp_audio.common_madmom import compute_beats_madmon
from config import args_to_dict

from smp_audio.common_librosa import myplot_specshow_librosa
from smp_audio.common_librosa import myplot_onsets, myplot_beats, myplot_tempo, myplot_chroma, myplot_segments, myplot_segments_hist

# # caching joblib
# from joblib import Memory
# location = './cachedir'
# memory = Memory(location, verbose=0)

def plotit(**kwargs):
    """plotit

    plot it. 'it' is the analysis result.
    """
    # copy parameters fwiw
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

def autovoice_align(beats, **kwargs):
    """autovoice
    """
    # TODO this needs some redesign
    # from autovoice_conf import voice_snips_trk006 as voice_snips
    # from autovoice_conf import voice_snips_1_trk006 as voice_snips_1
    # from autovoice_conf import voice_snips_trk008 as voice_snips
    # from autovoice_conf import voice_snips_1_trk008 as voice_snips_1
    # from autovoice_conf import voice_snips_trk026 as voice_snips
    # from autovoice_conf import voice_snips_1_trk026 as voice_snips_1
    # from autovoice_conf import voice_snips_trk_AUD20200213 as voice_snips
    # from autovoice_conf import voice_snips_1_trk_AUD20200213 as voice_snips_1
    # from autovoice_conf import voice_snips_trk_shmock2 as voice_snips
    # from autovoice_conf import voice_snips_1_trk_shmock2 as voice_snips_1
    # from autovoice_conf import voice_snips_trk_shluff2 as voice_snips
    # from autovoice_conf import voice_snips_1_trk_shluff2 as voice_snips_1
    # from autovoice_conf import voice_snips_trk021 as voice_snips
    # from autovoice_conf import voice_snips_1_trk021 as voice_snips_1
    # from autovoice_conf import voice_snips_multitrack002 as voice_snips
    # from autovoice_conf import voice_snips_1_multitrack002 as voice_snips_1
    # from autovoice_conf import voice_snips_multipat0001 as voice_snips
    # from autovoice_conf import voice_snips_1_multipat0001 as voice_snips_1
    # from autovoice_conf import voice_snips_swud as voice_snips
    # from autovoice_conf import voice_snips_1_swud as voice_snips_1
    # from autovoice_conf import voice_snips_sco2 as voice_snips
    # from autovoice_conf import voice_snips_1_sco2 as voice_snips_1

    # new
    from autovoice_conf import voice_snip_data

    print(f'autovoice main kw keys {list(kwargs)}')
    print(f'autovoice main kw conf {kwargs["conf"]}')

    voice_snips = voice_snip_data[kwargs['conf']]
    # voice_snips_1 = voice_snip_data[kwargs['conf']]['raw']
    
    d = beats['beats']
    beatarray = np.concatenate((d, d + d[-1] + 1))/2.0
    beats['beatsa'] = beatarray

    # print(voice_snips['proc'])
    # print('beats.shape = {0}'.format(librosa.frames_to_time(beats['beats']).shape))
    # print('beats = {0}'.format(librosa.frames_to_time(beats['beats'])))
    
    # imports on the fly
    from pydub import AudioSegment
    
    # create silence from base track
    a_ = AudioSegment.from_wav(kwargs['filename'])
    a_0 = AudioSegment.silent(duration=a_.duration_seconds*1000, frame_rate=a_.frame_rate)
    a_1 = AudioSegment.silent(duration=a_.duration_seconds*1000, frame_rate=a_.frame_rate)
    # a = AudioSegment.empty()
    a = AudioSegment.from_wav(kwargs['filename'])

    # number of main and side voices
    voice_main_numlayer = 5
    voice_side_numlayer = 10
    if 'numlayer' in voice_snips['raw']:
        voice_main_numlayer = voice_snips['raw']['numlayer']
    if 'numlayer' in voice_snips['proc']:
        voice_side_numlayer = voice_snips['proc']['numlayer']
    
    # voice density
    voice_main_density = 0.33
    voice_side_density = 0.66
    if 'density' in voice_snips['raw']:
        voice_main_density = voice_snips['raw']['density']
    if 'density' in voice_snips['proc']:
        voice_side_density = voice_snips['proc']['density']

    # voice reversity
    voice_main_reversity = 0.8
    voice_side_reversity = 0.6
    if 'reversity' in voice_snips['raw']:
        voice_main_reversity = voice_snips['raw']['reversity']
    if 'reversity' in voice_snips['proc']:
        voice_side_reversity = voice_snips['proc']['reversity']
        
    # voice swapity
    voice_main_swapity = 0.9
    voice_side_swapity = 0.7
    if 'swapity' in voice_snips['raw']:
        voice_main_swapity = voice_snips['raw']['swapity']
    if 'swapity' in voice_snips['proc']:
        voice_side_swapity = voice_snips['proc']['swapity']
        
    # # swud
    # voice_main_density = 0.9
    # voice_side_density = 0.6

    j_voice = voice_main_numlayer + voice_side_numlayer
    for j in range(j_voice):
        gain = 0
        # gain = random.randint(-3, 2)
        # print('gain = {0}'.format(gain))
        print('    voice layer = {0}, gain = {1}'.format(j, gain))
        # for beat_i, beat in enumerate(beatarray):
        
        for beat_i, beat in enumerate(librosa.frames_to_time(beats['beatsa'])):
            # if random.uniform(0, 1) > 0.5:
            # frame_mod = 2**np.random.choice([-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.8])
            frame_mod = 2**np.random.choice([-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.8, 1.3, 2.0])
            
            # gain_max = -9
            # gain_min = -12
            
            if j < voice_main_numlayer:
                # main / raw gain limits
                gain_max_raw = voice_snips['raw']['gain_max'] # param
                gain_min_raw = voice_snips['raw']['gain_min'] # param
                # density param
                if np.random.uniform(0, 1) < voice_main_density:
                    beat_i_scaled = int(beat_i * (len(voice_snips['raw']['files'])/beats['beatsa'].shape[0]))
                    # b = AudioSegment.from_wav(random.choice(voice_snips['raw']))
                    # voice_snips_1_idx = np.clip(beat_i + np.random.randint(-5, 5), 0, len(voice_snips['raw'])-1)
                    # voice_snips_1_idx = np.clip(beat_i_scaled + int(np.random.normal(0, 3)), 0, len(voice_snips['raw'])-1)
                    voice_snips_1_idx = np.clip(beat_i_scaled + int(np.random.normal(0, 1)), 0, len(voice_snips['raw']['files'])-1)
                    b_0 = AudioSegment.from_wav(voice_snips['raw']['files'][voice_snips_1_idx])
                    fr_ = b_0.frame_rate * frame_mod
                    b_0.set_frame_rate(int(fr_))
                    
                    # # gain distribution limits. last best default.
                    # gain_max = 0
                    # gain_min = -9
                else:
                    b_0 = AudioSegment.empty()

                # ah that's a weird choice, FIXME
                l_gain = random.randint(gain_min_raw, gain_max_raw)
                r_gain = random.randint(gain_min_raw, l_gain)
                b_0 = b_0.apply_gain_stereo(l_gain, r_gain)
                # b = b.apply_gain_stereo(b)

                # reverse density param
                if np.random.uniform(0, 1) > voice_main_reversity:
                    b_0 = b_0.reverse()

                # hae?!?! param
                if np.random.uniform(0, 1) > voice_main_swapity:
                    a_0 = a_0.overlay(b_0, position=beat*1000, gain_during_overlay=gain)
                    a = a.overlay(b_0, position=beat*1000, gain_during_overlay=gain)

            # side voice
            else:
                # parameters
                gain_max_proc = voice_snips['proc']['gain_max']
                gain_min_proc = voice_snips['proc']['gain_min']

                # event density
                if np.random.uniform(0, 1) < voice_side_density: # param
                    beat_i_scaled = int(beat_i * (len(voice_snips['proc']['files'])/beats['beatsa'].shape[0]))
                    # voice_snips_idx = np.clip(beat_i + np.random.randint(-10, 10), 0, len(voice_snips)-1)
                    voice_snips_idx = np.clip(beat_i_scaled + int(np.random.normal(0, 5)), 0, len(voice_snips['proc']['files'])-1)
                    if np.random.uniform(0, 1) > 0.66:
                        b_1 = AudioSegment.from_wav(voice_snips['proc']['files'][voice_snips_idx])
                    else:
                        b_1 = AudioSegment.from_wav(random.choice(voice_snips['proc']['files']))
                else:
                    b_1 = AudioSegment.empty()
                    
                # # last best default
                # gain_max = -9
                # gain_min = -33

                # weird again, FIXME
                l_gain = random.randint(gain_min_proc, gain_max_proc)
                r_gain = random.randint(gain_min_proc, l_gain)
                b_1 = b_1.apply_gain_stereo(l_gain, r_gain)
                # b = b.apply_gain_stereo(b)

                # random reverse param
                if np.random.uniform(0, 1) > voice_side_reversity:
                    b_1 = b_1.reverse()

                # random hae? stutter? param
                if np.random.uniform(0, 1) > voice_side_swapity:
                    a_1 = a_1.overlay(b_1, position=beat*1000, gain_during_overlay=gain)
                    a = a.overlay(b_1, position=beat*1000, gain_during_overlay=gain)

    a_0.export("out_a_0.wav", format="wav")
    a_1.export("out_a_1.wav", format="wav")
    a.export("out_a.wav", format="wav")

    # make new buffer / wav file
    # skip to time
    # insert random voice sample
    # write


def main(args):
    """main.autovoice

    apply the autovoice process.

    WIP. it is incomplete because the second part of the input, the
    voice recording, is hidden in a prepared code file and imported
    above, marked TODO or FIXME or both. .WIP

    Arguments:
    - filename(str): base track, the beat
    - duration(float): duration in seconds of output to create
    - seed(int): the random seed

    Returns:
    - side-effects !!!
    - results(dict): ideally the computation graph as a dictionary. WIP.
    """
    # set random set
    np.random.seed(args.seed)
    
    # transform argument dictionary
    kwargs = args_to_dict(args)

    # load the track data
    y, sr = data_load_librosa(**kwargs)
    filename = kwargs['filename']
    
    # myprint('Computing onsets')
    # onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    # librosa.frames_to_time(onset_frames, sr=sr)
    
    # # Or use a pre-computed onset envelope
    # # array([ 0.07 ,  0.395,  0.511,  0.627,  0.766,  0.975,
    # # 1.207,  1.324,  1.44 ,  1.788,  1.881])

    # compute track features
    # chromagram
    chroma = compute_chroma_librosa(y, sr)['chromagram']
    # # tempogram
    # tempogram = compute_tempogram_librosa(y, sr, onset_env)
    # onsets
    # onset_env, onset_times_ref, onset_frames = compute_onsets_librosa(y, sr)
    onsets = compute_onsets_librosa(y, sr)

    # FIXME: is there a beat? danceability?
    # compute_beat(onsets['onsets_env'], onsets['onsets_frames'])

    ############################################################
    # compute the beat tracking (using librosa)

    # beats is a dictionary of events, to be used later in the autovoice_align function
    beats = {}
    
    # loop over tempo priors
    # for start_bpm in [30, 60, 90]:
    for start_bpm in [120]:
        # whatever legacy
        # t_, dt_, b_ = compute_beats_librosa(onsets['onsets_env'], onsets['onsets_frames'], start_bpm, sr)
        
        # compute the beats
        beats_dict = compute_beats_librosa(onsets['onsets_env'], onsets['onsets_frames'], start_bpm, sr)

        # namespace moves
        t_ = beats_dict['tempo']
        dt_ = beats_dict['dtempo']
        b_ = beats_dict['beats']

        # unknown
        # b_f_ = librosa.util.fix_frames(b_, x_max=chroma.shape[1])

        # copy operations
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

    ############################################################
    # track segmentation. dont think this is used on the autovoice_align bit.
    
    # numparts prior
    numparts = 5
    segments = {}
    # for numparts in [5, 10]: # range(2, 20):
    for numparts in [6,7]:
        # bd_, bdt_, bds_ = compute_segments_librosa(chroma, sr, numparts)
        segments_dict = compute_segments_librosa(chroma, sr, numparts)
        bd_ = segments_dict['bounds_frames']
        bdt_ = segments_dict['bounds_times']
        bds_ = segments_dict['bounds_samples']
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
        # librosa.output.write_wav(outfilename, tmp_, sr)
        soundfile.write(outfilename, tmp_.T, sr, 'PCM_16', format='WAV')
    
    # bound_times
    # array([  0.   ,   1.672,   2.322,   2.624,   3.251,   3.506,
    #      4.18 ,   5.387,   6.014,   6.293,   6.943,   7.198,
    #      7.848,   9.033,   9.706,   9.961,  10.635,  10.89 ,
    #     11.54 ,  12.539])

    # end track segmentation
    
    ############################################################
    # autovoice, the juice
    
    # call the autovoice align function
    autovoice_align(beats['lr_bpm120'], **kwargs)
    # done, show the results

    ############################################################
    # plotting and output

    # plotting
    myprint('Plotting')
    colors = ['k', 'b', 'g', 'r', 'm', 'c']
    # fig = make_fig(rows=6, cols=1, title='%s' % (filename))
    fig = plt.figure()
    ax_ = fig.add_subplot(6, 1, 1)
    for i in range(1, 6):
        _ = fig.add_subplot(6, 1, i+1, sharex=ax_)
    
    myplot_specshow_librosa(fig.axes[0], y)
    myplot_onsets(fig.axes[1], onsets['onsets_times_ref'], onsets['onsets_env'], onsets['onsets_frames'])
    for i, beat_ in enumerate(beats):
        beats_ = beats[beat_]['beats']
        # [('beats', 0.5, 1.0, 0.5, 'b', '--', 'Beats'),
        #  ('beats2', 0.0, 0.5, 0.5, 'k', '-.', 'Beats2'),
        #  ('mm_beat_times', 0.25, 0.75, 'r', '-', 'mm_beats')]:
        print(beat_, beats_)
        if beat_.startswith('lr'):
            beattimes = onsets['onsets_times_ref'][beats_]
        else:
            beattimes = beats_
        myplot_beats(fig.axes[2], beattimes, ylow=0, yhigh=1, alpha=0.5,
                     color=colors[i], linestyle='--', label=beat_)
        if beats[beat_]['dtempo'] is not None:
            myplot_tempo(fig.axes[3], onsets['onsets_times_ref'], beats[beat_]['dtempo'])

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

    # plotit(y=y, o_env=o_env, times=times, onset_frames=onsets['onsets_frames'], file=args.file,
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
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', help='Config key to load from autovoice configuration module [sco_2]', default='sco_2', type=str)
    parser.add_argument('-d', '--duration', help='Input duration (secs) to select from input file [10.0]',
                        default=10.0, type=float)
    parser.add_argument('-f', '--filename', help='Sound file to process', default=None, type=str)
    parser.add_argument('-s', '--seed', help='Random seed [0]', default=0, type=int)

    args = parser.parse_args()
    print(args)
    main(args)
    
