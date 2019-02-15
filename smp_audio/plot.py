import librosa
import librosa.display
import numpy as np

colors = ['k', 'b', 'g', 'r', 'm', 'c']

def myplot_specshow_librosa(ax, y):
    D = np.abs(librosa.stft(y))
    librosa.display.specshow(
        librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log',
        ax=ax
    )
    ax.set_title('Power spectrogram')
    print('myplot_specshow_librosa xlimits', ax.get_xlim())

def myplot_onsets(ax, times, o_env, onset_frames):
    ax.plot(times, o_env, label='Onset strength')
    ax.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
              linestyle='--', label='Onsets')
    ax.set_xlim((times[0], times[-1]))
    print('myplot_onsets xlimits', ax.get_xlim())
    print('myplot_onsets times', times)

def myplot_beats(ax, times, beattimes, ylow, yhigh, alpha, color, linestyle, label):
    ax.vlines(times[beattimes], ylow, yhigh, alpha=alpha, color=color, linestyle=linestyle, label=label)
    ax.set_xlim((times[0], times[-1]))

def myplot_tempo(ax, times, tempo):
    # myprint('myplot_tempo times = %s, tempo = %s' % (times, tempo))
    ax.plot(times, tempo, alpha=0.5, color='b',
             linestyle='none', marker='o', label='Tempo')

def myplot_chroma(ax, chroma):
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    
def myplot_chroma_sync(ax, chroma, beat_t):
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax, x_coords=beat_t)
    
def myplot_segments(ax, chroma, bound_times, times, color='g', alpha=0.66):
    ax.vlines(times[bound_times], 0, chroma.shape[0], color=color, linestyle='-',
            linewidth=3, alpha=alpha, label='Segment boundaries')
    ax.set_xlim((times[0], times[-1]))

def myplot_segments_hist(ax, bound_hist, idx=None, color='k'):
    myprint('segment bound hist', bound_hist)
    if idx is not None:
        ax.bar([idx], bound_hist.mode, alpha=0.4, color=color)
    else:
        ax.bar(bound_hist[1][:-1], bound_hist[0], alpha=0.4, color=color)
