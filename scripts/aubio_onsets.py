"""aubio_onsets

Oswald Berthold, 2019

aubio load and calling pattern deduced from ipy session
""" 
import aubio
import numpy as np
import matplotlib.pyplot as plt

src = aubio.source('/home/src/QK/data/sound-arglaaa-2018-10-25/24.wav', channels=1)
src.seek(0)
onsets = []

od = aubio.onset(method='kl', samplerate=src.samplerate)
# od.set_threshold(1.0)

while True:
    samples, read = src()
    print(samples.shape)
    if read < src.hop_size:
        break
    od(samples)
    onsets.append(od.get_thresholded_descriptor())
    
plt.plot(np.array(onsets))
plt.show()
