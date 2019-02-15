smp\_audio aka smp_audio
====================

Slurp comes from *smp_audioing* in music data and creating an incremental model based representation of the audio. This can be then be used for support or automation of different audio editing and production tasks such as segmentation, editing, score reconstruction, etc.

The library wraps several different existing MIR libraries, currently essentia, librosa, madmom, aubio, pyAudioAnalysis.

Install
-------

On Linux / Ubuntu

``` example
sudo pip3 install -r requirements-simple.txt essentia librosa madmon aubio pyAudioAnalysis
```

On Mac OSX

Install essentia with their homebrew formula <https://github.com/MTG/homebrew-essentia>, <https://essentia.upf.edu/documentation/installing.html>

Otherwise

``` example
sudo pip3 install -r requirements-simple.txt librosa madmon aubio pyAudioAnalysis
```

Modules and workflows
---------------------

The scripts folder contains prototypes for different music related workflows, e.g. automixing (quantitative measure based playlist sequencing), or autoedit (beat-tracking and segmentation based automatic editing and arrangement to duration of choice from raw audio session that is shorter or longer than output duration).

| **tag**          |                                            | desc                                                                                  |
|------------------|--------------------------------------------|---------------------------------------------------------------------------------------|
| **scripts**      |                                            | smp\_audio scripts                                                                    |
| **tools**        |                                            | additional tools to support larger scopes and additional processing steps             |
| silence-splitter |                                            | split a large audio file (&gt; 1h) into smaller parts (e.g. 10') based on silence     |
|                  |                                            | detection                                                                             |
|                  | sox                                        | silence plugin command line                                                           |
|                  |                                            | : sox -V3 audiofile.wav audiofile\_part\_.wav silence -l 0 1 2.0 0.1%                 |
|                  | aubio quiet                                | aubio quiet - analyze audio and print timestamps w/ onsets of silence and noise parts |
|                  |                                            | needs to be converted to input for a slicer or aubiocut                               |
|                  |                                            | : aubio quiet filename.wav                                                            |
|                  | aubio cut                                  | aubiocut cuts audio at every onset incl. option for beat alignment                    |
|                  |                                            |                                                                                       |
| downloaders      |                                            |                                                                                       |
|                  | soundscrape                                | soundcloud and bandcamp downloader                                                    |
|                  |                                            | <https://github.com/Miserlou/SoundScrape>                                             |
|                  |                                            | : sudo pip3 install SoundScrape                                                       |
|                  | youtube-dl                                 | versatile youtube downloader                                                          |
|                  |                                            |                                                                                       |
|                  | OBSOLETE below                             |                                                                                       |
|                  | playground/music\_beats.py                 | stub                                                                                  |
|                  | playground/music\_features\_print\_list.py |                                                                                       |
|                  | playground/music\_features.py              | collection of different sound parsing experiments                                     |
|                  | librosa-onset-detect-1.py                  |                                                                                       |
|                  | librosa-onset-onset\_detect-1.py           | final version using librosa/madmon/essentia for beat tracking and segmentation        |
|                  | librosa-onset-onset\_strength-1.py         |                                                                                       |
|                  | librosa-onset-onset\_strength\_multi-1.py  |                                                                                       |
|                  |                                            |                                                                                       |
|                  | moved all files to start with music\_      |                                                                                       |
|                  | copied all files to smp_audio/scripts      |                                                                                       |

Process prototype
-----------------

-   read file
-   apply iterative analyses to dynamically build graph of music data

### Caching

Using dict of functions and joblib.Memory to cache all compute intensive funtions. This is done at the calling level.

Librosa has its own caching mechanism, which is used in the librosa specific code.

``` example
LIBROSA_CACHE_DIR
```

``` example
$ export LIBROSA_CACHE_DIR=/tmp/librosa_cache
$ ipython
```

``` example
import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
import librosa
```
