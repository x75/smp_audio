"""audio-segments-split.py

.. moduleauthor:: Oswald Berthold, 2019

Split a single audio wav file into segments and save the segments to
files.

1. Take ensemble of event predictions (segments, beat positions, ...)
2. combine them into an improved prediction
3. slice the data into segments based on the improved prediction
4. write the segment slices to disk as wavs for use down the pipeline
"""
from slurp.segments import compute_event_mean_intervals
from slurp.segments import compute_event_merge_mexhat
from slurp.segments import plot_event_merge_results
from slurp.segments import compute_event_merge_heuristics
from slurp.segments import compute_event_merge_index_to_file

def main_audio_segments_split(**kwargs):
    # number of frames, FIXME winsize, hopsize
    numframes = kwargs['numframes']

    # get mean intervals and number of event sequences to merge
    tmp_ = compute_event_mean_intervals(**kwargs)
    kwargs.update(tmp_)
    intervals = tmp_['intervals']
    numinputs = tmp_['numinputs']
    final = tmp_['final']

    tmp_ = compute_event_merge_mexhat(**kwargs)
    kwargs.update(tmp_)
    kernels = tmp_['kernels']
    final_sum = tmp_['final_sum']
    ind = tmp_['ind']
    ind2 = tmp_['ind2']

    plot_event_merge_results(**kwargs)

    tmp_ = compute_event_merge_heuristics(**kwargs)
    kwargs.update(tmp_)

    # kwargs['filename_48'] = '/home/src/QK/data/sound-arglaaa-2018-10-25/22.wav'
    tmp_ = compute_event_merge_index_to_file(**kwargs)

    print(('write files {0}'.format(tmp_)))

    return tmp_
        
if __name__ == '__main__':
    numframes = 3191

    #   DEBUG:           nodes_plot: step beat_
    beats = [
    [ 19, 51, 83, 115, 147, 179, 211, 242, 274, 306, 338, 370, 402,
      434, 466, 498, 530, 562, 593, 625, 657, 689, 721, 753, 785, 817,
      848, 880, 912, 937, 966, 995, 1024, 1056, 1088, 1120, 1152, 1184,
      1216, 1248, 1279, 1311, 1344, 1375, 1407, 1439, 1471, 1503, 1535,
      1566, 1598, 1630, 1662, 1694, 1726, 1758, 1790, 1822, 1854, 1885,
      1918, 1950, 1981, 2013, 2045, 2077, 2109, 2140, 2172, 2205, 2237,
      2268, 2300, 2332, 2364, 2396, 2428, 2460, 2492, 2524, 2555, 2587,
      2619, 2651, 2683, 2715, 2747, 2779, 2811, 2842, 2874, 2906, 2938,
      2970, 3002, ],
    #   DEBUG:           nodes_plot: step beat_
    [  19, 51, 83, 115, 147, 179, 211, 242, 274, 306, 338, 370, 402, 434, 
       466, 498, 530, 562, 593, 625, 657, 689, 721, 753, 785, 817, 848, 880, 
       912, 937, 966, 995, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1279, 1311, 
       1344, 1375, 1407, 1439, 1471, 1503, 1535, 1566, 1598, 1630, 1662, 1694, 1726, 1758, 
       1790, 1822, 1854, 1885, 1918, 1950, 1981, 2013, 2045, 2077, 2109, 2140, 2172, 2205, 
       2237, 2268, 2300, 2332, 2364, 2396, 2428, 2460, 2492, 2524, 2555, 2587, 2619, 2651, 
       2683, 2715, 2747, 2779, 2811, 2842, 2874, 2906, 2938, 2970, 3002, ],
    #   DEBUG:           nodes_plot: step beat_
    [ 19, 51, 83, 115, 147, 179, 211, 242, 274, 306, 338, 370, 402,
      434, 466, 498, 530, 562, 593, 625, 657, 689, 721, 753, 785, 817,
      848, 880, 912, 937, 966, 995, 1024, 1056, 1088, 1120, 1152, 1184,
      1216, 1248, 1279, 1311, 1344, 1375, 1407, 1439, 1471, 1503, 1535,
      1566, 1598, 1630, 1662, 1694, 1726, 1758, 1790, 1822, 1854, 1885,
      1918, 1950, 1981, 2013, 2045, 2077, 2109, 2140, 2172, 2205, 2237,
      2268, 2300, 2332, 2364, 2396, 2428, 2460, 2492, 2524, 2555, 2587,
      2619, 2651, 2683, 2715, 2747, 2779, 2811, 2842, 2874, 2906, 2938,
      2970, 3002, ],
    #    [   3, 19, 35, 51, 67, 83, 99, 115, 131, 147, 163, 179, 195, 211, 
    #  226, 242, 258, 274, 290, 306, 322, 338, 354, 370, 386, 402, 418, 434, 
    #  450, 466, 482, 498, 514, 530, 546, 562, 577, 593, 609, 625, 641, 657, 
    #  673, 689, 705, 721, 737, 753, 769, 785, 801, 817, 832, 848, 864, 880, 
    #  896, 912, 929, 945, 961, 977, 992, 1008, 1024, 1040, 1056, 1072, 1088, 1104, 
    # 1120, 1136, 1152, 1168, 1184, 1200, 1216, 1232, 1248, 1264, 1279, 1295, 1311, 1328, 
    # 1344, 1359, 1375, 1391, 1407, 1423, 1439, 1455, 1471, 1487, 1503, 1519, 1535, 1551, 
    # 1566, 1582, 1598, 1614, 1630, 1646, 1662, 1678, 1694, 1710, 1726, 1742, 1758, 1774, 
    # 1790, 1806, 1822, 1838, 1854, 1869, 1885, 1902, 1918, 1934, 1950, 1966, 1981, 1997, 
    # 2013, 2029, 2045, 2061, 2077, 2093, 2109, 2125, 2140, 2156, 2172, 2189, 2205, 2221, 
    # 2237, 2253, 2268, 2284, 2300, 2316, 2332, 2348, 2364, 2380, 2396, 2412, 2428, 2444, 
    # 2460, 2476, 2492, 2508, 2524, 2540, 2555, 2571, 2587, 2603, 2619, 2635, 2651, 2667, 
    # 2683, 2699, 2715, 2731, 2747, 2763, 2779, 2795, 2811, 2827, 2842, 2858, 2874, 2890, 
    # 2906, 2922, 2938, 2954, 2970, 2986, 3002, 3018, 3034, ]
    ]

    #   DEBUG:           nodes_plot: step segment_
    segs = [
        [   0, 773, 834, 902, 1119, 2844, 2911, 2953, 3033, 3065, ],
        [ 0, 537, 627, 665, 737, 773, 834, 902, 1119, 1175, 1267, 1303, 1389, 2065, 
          2131, 2844, 2911, 2953, 3033, 3065, ],
        [ 0, 27, 99, 154, 245, 282, 372, 537, 627, 665, 737, 773, 834,
          902, 1119, 1175, 1267, 1303, 1389, 2065, 2131, 2192, 2250, 2324,
          2394, 2844, 2911, 2953, 3033, 3065, ]
    ]
    
    main_audio_segments_split(beats=beats, segs=segs, numframes=numframes)
