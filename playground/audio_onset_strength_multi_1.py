# First, load some audio and plot the spectrogram

import matplotlib.pyplot as plt
y, sr = librosa.load(librosa.util.example_audio_file(),
                     duration=10.0)
D = np.abs(librosa.stft(y))
plt.figure()
plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='log')
plt.title('Power spectrogram')

# Construct a standard onset function over four sub-bands

onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr,
                                                    channels=[0, 32, 64, 96, 128])
plt.subplot(2, 1, 2)
librosa.display.specshow(onset_subbands, x_axis='time')
plt.ylabel('Sub-bands')
plt.title('Sub-band onset strength')
