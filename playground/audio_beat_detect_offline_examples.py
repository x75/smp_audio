"""code snippets from
https://www.analyticsvidhya.com/blog/2018/02/audio-beat-tracking-for-music-information-retrieval/

Oswald Berthold, 2018
"""

# import modules
import librosa 
import IPython.display as ipd 

# read audio file 
x, sr = librosa.load('Media-103515.wav') 
ipd.Audio(x, rate=sr)


# approach 1 - onset detection and dynamic programming
tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=60, units='time')

clicks = librosa.clicks(beat_times, sr=sr, length=len(x))
ipd.Audio(x + clicks, rate=sr)

# import modules
import madmom 

# approach 2 - dbn tracker
# https://musicinformationretrieval.files.wordpress.com/2017/02/multimodel.png

proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()('train/train1.wav')

beat_times = proc(act)

clicks = librosa.clicks(beat_times, sr=sr, length=len(x))
ipd.Audio(x + clicks, rate=sr)
