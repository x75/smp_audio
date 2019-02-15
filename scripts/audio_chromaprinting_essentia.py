import essentia.standard as es
import essentia.streaming as ess



import matplotlib.pyplot as plt


loader = ess.MonoLoader(filename = '/home/src/QK/data/sound-arglaaa-2018-10-25/22.wav')
fps = ess.Chromaprinter(analysisTime=20, concatenate=True)
pool = ess.essentia.Pool()

# Conecting the algorithms
loader.audio >> fps.signal
fps.fingerprint >> (pool, 'chromaprint')

ess.essentia.run(loader)

fp = pool['chromaprint'][0]

print(('fp = {0}'.format(fp)))

import acoustid as ai
# import acoustid.chromaprint

fp_int = ai.chromaprint.decode_fingerprint(fp)[0]
# fp_int = chromaprint.decode_fingerprint(fp)[0]

fb_bin = [list('{:032b}'.format(abs(x))) for x  in fp_int] # Int to unsigned 32-bit array

arr = np.zeros([len(fb_bin), len(fb_bin[0])])

for i in range(arr.shape[0]):
    arr[i,0] = int(fp_int[i] > 0) # The sign is added to the first bit
    for j in range(1, arr.shape[1]):
        arr[i,j] = float(fb_bin[i][j])

plt.imshow(arr.T, aspect='auto', origin='lower')
plt.title('Binary representation of a Chromaprint ')

plt.Text(0.5,1,'Binary representation of a Chromaprint ')


plt.show()
