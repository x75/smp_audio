import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# from util
def args_to_dict(args):
    cls=dict
    # l.info('geconf.from_args(args = {0})'.format(args))
    args_attrs = [attr_ for attr_ in dir(args) if not attr_.startswith('_')]
    args_attr_vals = dict([(attr_, getattr(args, attr_)) for attr_ in dir(args) if not attr_.startswith('_')])
    # l.info('geconf.from_args: dir(args) = {0})'.format(args_attr_vals))
    return cls(**args_attr_vals)

def compute_tempogram_and_tempo(filename):
    # Compute local onset autocorrelation
    # filename = librosa.util.example_audio_file()
    y, sr = librosa.load(filename)
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                          hop_length=hop_length)
    # Compute global onset autocorrelation
    ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    ac_global = librosa.util.normalize(ac_global)
    # Estimate the global tempo for display purposes
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                               hop_length=hop_length)[0]

    return {'tempo': tempo, 'tempogram': tempogram, 'oenv': oenv,
            'sr': sr, 'y': y, 'ac_global': ac_global, 'hop_length': hop_length}

def plot_tempogram_and_tempo(data):
    plt.figure(figsize=(8, 8))

    oenv = data['oenv']
    tempo = data['tempo']
    tempogram = data['tempogram']
    ac_global = data['ac_global']
    sr = data['sr']
    hop_length = data['hop_length']
    
    plt.subplot(4, 1, 1)
    plt.plot(oenv, label='Onset strength')
    plt.xticks([])
    plt.legend(frameon=True)
    plt.axis('tight')
    
    plt.subplot(4, 1, 2)
    # We'll truncate the display to a narrower range of tempi
    librosa.display.specshow(
        tempogram, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='tempo')
    plt.axhline(tempo, color='w', linestyle='--', alpha=1,
                label='Estimated tempo={:g}'.format(tempo))
    plt.legend(frameon=True, framealpha=0.75)
    
    plt.subplot(4, 1, 3)
    x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
                    num=tempogram.shape[0])
    plt.plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
    plt.plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
    plt.xlabel('Lag (seconds)')
    plt.axis('tight')
    plt.legend(frameon=True)
    
    plt.subplot(4,1,4)
    # We can also plot on a BPM axis
    freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
    plt.semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
                 label='Mean local autocorrelation', basex=2)
    plt.semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
                 label='Global autocorrelation', basex=2)
    plt.axvline(tempo, color='black', linestyle='--', alpha=.8,
                label='Estimated tempo={:g}'.format(tempo))
    plt.legend(frameon=True)
    plt.xlabel('BPM')
    plt.axis('tight')
    plt.grid()
    plt.tight_layout()



def main(args):
    # convert args to dict
    kwargs = args_to_dict(args)

    data = compute_tempogram_and_tempo(kwargs['filenames'][0])
    plot_tempogram_and_tempo(data)

    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filenames", action = 'append', dest = 'filenames', help="Input file(s) []", nargs = '+', default = [])

    args = parser.parse_args()

    args.filenames = args.filenames[0]
    
    main(args)

