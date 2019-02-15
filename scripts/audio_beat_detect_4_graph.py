# import argparse
import os
# os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'

from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

import numpy as np
import matplotlib.pyplot as plt
from smp_base.plot import make_fig

from slurp.common import data_load_librosa
from slurp.common import compute_onsets_librosa, compute_chroma_librosa, compute_tempogram_librosa
from slurp.common import compute_beats_librosa, compute_beats_madmon
from slurp.common import compute_segments_librosa, compute_segments_essentia

from slurp.plot import colors
from slurp.plot import myplot_specshow_librosa
from slurp.plot import myplot_onsets, myplot_beats, myplot_tempo, myplot_chroma, myplot_segments


# costly_compute_cached = memory.cache(costly_compute_cached)
# start = time.time()
# data_trans = costly_compute_cached(data)
# end = time.time()

np.random.seed(1)

def music_beat_detect_4_graph(*args, **kwargs):
    # 1 - every function defines a node
    # 2 - node input and output conf implicitly defines edges

    # node-00: input arguments
    assert 'filename' in kwargs, 'No filename given in kwargs with keys {0}'.format(list(kwargs))
    assert 'duration' in kwargs, 'No duration given in kwargs with keys {0}'.format(list(kwargs))

    # node-01: load data
    y, sr = data_load_librosa(filename=kwargs['filename'], duration=kwargs['duration'])

    # node-02: compute onsets
    onset_env, onset_frames, onset_times_ref = compute_onsets_librosa(y, sr)

    # node-03: compute chromagram
    chroma = compute_chroma_librosa(y, sr)

    # node-04: compute tempogram
    tempogram = compute_tempogram_librosa(y, sr, onset_env)
    
    # compute beat tracking (librosa) based on tempogram prediction
    beats = []
    for start_bpm in [60, 90, 120]:
        # tempo, dtempo, beats
        t_, dt_, b_ = compute_beats_librosa(onset_env, onset_frames, start_bpm, sr)
        beats.append((t_, dt_, b_))
    
    # compute beat tracking (madmom)
    compute_beats_madmon_cached = memory.cache(compute_beats_madmon)
    # t_, dt_, b_ = compute_beats_madmon(None, None, None, sr, filename)
    t_, dt_, b_ = compute_beats_madmon_cached(None, None, None, sr, filename)
    beats.append((t_, dt_, b_))

    # compute segmentation (librosa)
    parts = []
    for numparts in [6,7,8]:
        bd_, bdt_, bds_ = compute_segments_librosa(chroma, sr, numparts)
        parts.append((bd_, bdt_, bds_))

    # compute segmentation (essentia)
    bd_, bdt_, bds_ = compute_segments_essentia(chroma, sr, numparts)
    parts.append((bd_, bdt_, bds_))

    # plot results by plotting graph w/ plot configuration included in graph_conf
    # - one panel per node
    fig = make_fig(rows=6, cols=1, title='%s' % (filename))

    # plot spectra
    myplot_specshow_librosa(fig.axes[0], y)

    # plot onsets
    myplot_onsets(fig.axes[1], onset_times_ref, onset_env, onset_frames)

    # plot beats
    for i, beat_ in enumerate(beats):
        myplot_beats(fig.axes[2], beat_[2], ylow=0, yhigh=1, alpha=0.5,
                     color=colors[i], linestyle='--', label=beat_)
        if beat_[1] is not None:
            myplot_tempo(fig.axes[3], onset_times_ref, beat_[1])

    # plot segmentation on top of feature-gram
    myplot_chroma(fig.axes[4], chroma)
    for i, part_ in enumerate(parts):
        myplot_segments(fig.axes[4], chroma, part_[1], color=colors[i])
    plt.legend()
    # myplot_segments(fig.axes[4], chroma, parts[1][1], color='g')
    # myplot_segments(fig.axes[4], chroma, parts[2][1], color='r')

# args = argparse.Namespace()
filename = '/home/src/QK/data/sound-arglaaa-2018-10-25/22-mono.wav'
music_beat_detect_4_graph(filename=filename, duration=75)
plt.show()
