smp\_audio aka smp\_audio
=========================

A playground for designing audio tools, *smp\_audio* is part of the smp project, a set of conventions about sensorimotor learning and developmental models used in robotics research. Auditory perception is an important function for robots, but there are clear side-effects. For example, the modules a robot needs to "hear" things, can used to build audio- and music tools for consumers and pros with novel properties. That is the idea, that this project tries to validate.

The smp approach is shaped by rapid-prototyping and functional proofing. Optimization issues are defered entirely to later stages, generally assuming them possible-if-justified. The approach is fundamentally graphical and only informally defined beyond this fact. Algorithms and processing flows are modelled as graphs whose nodes comprise function calls on impinging data. This project consists of a set of library functions in the \`smp\_audio\` directory, and experimental processing flows in the \`scripts\` directory, making use of library calls to build apps and usage examples.

Within smp, third party libraries are liberally used to realize the *rapid* aspect, at the cost of stability and ease of installation. The library wraps different MIR functions from several existing implementations, currently essentia, librosa, madmom, aubio, pyAudioAnalysis. These wrappers are complemented with additional glue and integration functions.

Install
-------

Part one of the dependency pile is just the usual Python scientific computing stack and can usually be installed as distribution packages.

On Linux / Ubuntu

``` example
sudo apt install python3-numpy python3-scipy python3-matplotlib python3-pandas python3-sklearn
```

or just using pip

``` example
sudo pip3 install numpy scipy matplotlib pandas sklearn
```

Part two is the audio and MIR specific packages, which I install with pip directly

``` example
sudo pip3 install -r requirements-simple.txt essentia librosa madmom aubio pyAudioAnalysis
```

On Mac OSX

Install essentia with their homebrew formula <https://github.com/MTG/homebrew-essentia>, <https://essentia.upf.edu/documentation/installing.html>

Then followed by

``` example
sudo pip3 install -r requirements-simple.txt librosa madmon aubio pyAudioAnalysis
```

To finish, add the module path to your PYTHONPATH by running

``` example
export PYTHONPATH=/path/to/cloned/smp_audio:$PYTHONPATH
```

In case you come back, this can be added to your shell profile.

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
| OBSOLETE         |                                            |                                                                                       |
|                  | playground/music\_beats.py                 | stub                                                                                  |
|                  | playground/music\_features\_print\_list.py |                                                                                       |
|                  | playground/music\_features.py              | collection of different sound parsing experiments                                     |
|                  | librosa-onset-detect-1.py                  |                                                                                       |
|                  | librosa-onset-onset\_detect-1.py           | final version using librosa/madmon/essentia for beat tracking and segmentation        |
|                  | librosa-onset-onset\_strength-1.py         |                                                                                       |
|                  | librosa-onset-onset\_strength\_multi-1.py  |                                                                                       |
|                  |                                            |                                                                                       |
|                  | moved all files to start with music\_      |                                                                                       |
|                  | copied all files to smp\_audio/scripts     |                                                                                       |

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
