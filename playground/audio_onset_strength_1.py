# First, load some audio and plot the spectrogram
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# y, sr = librosa.load(librosa.util.example_audio_file(),
#                      duration=10.0)
y, sr = librosa.load('/home/x75/Downloads/BEAT1R-mono.wav',
                     duration=10.0)
D = np.abs(librosa.stft(y))
times = librosa.frames_to_time(np.arange(D.shape[1]))
plt.figure()
ax1 = plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('Power spectrogram')

# Construct a standard onset function

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(times, 2 + onset_env / onset_env.max(), alpha=0.8,
         label='Mean (mel)')

# Median aggregation, and custom mel options

onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         aggregate=np.median,
                                         fmax=8000, n_mels=256)
plt.plot(times, 1 + onset_env / onset_env.max(), alpha=0.8,
         label='Median (custom mel)')

# Constant-Q spectrogram instead of Mel

onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         feature=librosa.cqt)
plt.plot(times, onset_env / onset_env.max(), alpha=0.8,
         label='Mean (CQT)')
plt.legend(frameon=True, framealpha=0.75)
plt.ylabel('Normalized strength')
plt.yticks([])
plt.axis('tight')
plt.tight_layout()

plt.show()

