"""aubio_onsets

Oswald Berthold, 2019

Compute and display all aubio onset detection functions for a given
input file.

aubio load and calling pattern deduced from ipy session
""" 
import sys
import aubio
import numpy as np
import matplotlib.pyplot as plt

from smp_audio.common_aubio import data_stream_aubio, data_stream_get_aubio
from smp_audio.common_librosa import data_stream_librosa, data_stream_get_librosa

if len(sys.argv) < 2:
    input_file = '/home/lib/audio/work/arglaaa-mini/24.wav'
else:
    input_file = sys.argv[1]
print('input_file = {0}'.format(input_file))
# onset_method = 'specdiff'
onset_detectors = {
    "energy": {}, "hfc": {}, "complex": {}, "phase": {}, "wphase": {},
    "specdiff": {}, "kl": {}, "mkl": {}, "specflux": {},
}

frame_size = 1024
# src = aubio.source(input_file, channels=1)
# src.seek(0)
src, src_samplerate = data_stream_aubio(filename=input_file, frame_size=frame_size)
# src, src_samplerate = data_stream_librosa(filename=input_file)

print(src, src_samplerate)


# onset(onset_method, bufsize, hopsize, samplerate)

for onset_detector in onset_detectors:
    onset_detectors[onset_detector]['onsets'] = []
    onset_detectors[onset_detector]['onsets_thresholded'] = []
    onset_detectors[onset_detector]['instance'] = aubio.onset(onset_detector, frame_size, frame_size, samplerate=src_samplerate)
    onset_detectors[onset_detector]['instance'].set_threshold(1.0)
    # onsets = []
    # onsets_thresholded = []
    # onset_detectors = []
    
# buf_size=512, hop_size=512,

# for func in onset_detection_functions:
#     od = aubio.onset(method=onset_method, samplerate=src.samplerate)
#     onset_detectors.append(od)
#     od.set_threshold(1.0)

# while True:
for item in data_stream_get_aubio(src):
# for item in data_stream_get_librosa(src):
    # print(f'')
    samples = item[0]
    read = item[1]
    # print(f'samples.shape {samples.shape}, read = {read})
    if read < frame_size: break
    for onset_detector in onset_detectors:
        od = onset_detectors[onset_detector]['instance']
        od(samples)
        onset_detectors[onset_detector]['onsets_thresholded'].append(od.get_thresholded_descriptor())
        onset_detectors[onset_detector]['onsets'].append(od.get_descriptor())

print('Computed {0} frames'.format(len(onset_detectors[onset_detector]['onsets'])))

fig = plt.figure()
fig.suptitle('onsets (aubio {0}) for {1}'.format('onsets', input_file))

for i, onset_detector in enumerate(onset_detectors):
    ax = fig.add_subplot(len(onset_detectors),2,2*i+1)
    ax.set_title(onset_detector)
    ax.plot(np.array(onset_detectors[onset_detector]['onsets']), label='onsets')
    ax.legend()
    ax = fig.add_subplot(len(onset_detectors),2,2*i+2)
    ax.set_title(onset_detector)
    # ax.plot(np.array(onset_detectors[onset_detector]['onsets_thresholded']), alpha=0.5, label='thresholded onsets')
    ax.plot(np.array(onset_detectors[onset_detector]['onsets_thresholded']) > 0, label='thresholded onsets')
    ax.legend()

plt.show()
