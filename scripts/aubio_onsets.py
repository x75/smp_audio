"""aubio_onsets

Oswald Berthold, 2019

aubio load and calling pattern deduced from ipy session
""" 
import aubio
import numpy as np
import matplotlib.pyplot as plt

input_file = '/home/lib/audio/work/arglaaa-mini/24.wav'
onset_method = 'specdiff'
onset_detectors = {
    "energy": {}, "hfc": {}, "complex": {}, "phase": {}, "wphase": {},
    "specdiff": {}, "kl": {}, "mkl": {}, "specflux": {},
}

src = aubio.source(input_file, channels=1)
src.seek(0)

for onset_detector in onset_detectors:
    onset_detectors[onset_detector]['onsets'] = []
    onset_detectors[onset_detector]['onsets_thresholded'] = []
    onset_detectors[onset_detector]['instance'] = aubio.onset(method=onset_detector, samplerate=src.samplerate)
    onset_detectors[onset_detector]['instance'].set_threshold(1.0)
    # onsets = []
    # onsets_thresholded = []
    # onset_detectors = []

# for func in onset_detection_functions:
#     od = aubio.onset(method=onset_method, samplerate=src.samplerate)
#     onset_detectors.append(od)
#     od.set_threshold(1.0)

while True:
    samples, read = src()
    # print(samples.shape)
    if read < src.hop_size:
        break

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
