# smp\_audio

The audio tooling playground within the **sensorimotor primitives**
(smp) family of packages \[1\]. They come out of a sensorimotor approach
to learning and development on robots.

Auditory perception is an extremely important modality that has not seen
the attention it deserves in robotics. There is a large overlap in
modeling robot perception and computer music processes. smp\_audio is a
collection of batch algorithms for audio processing tasks based on
techniques pulled from MIR and machine listening.

This is research code and was created on a functional proofing mindset
bent to radically rapid prototyping. Processing flows are modeled
graphically where nodes comprise function calls on impinging data. There
exist several versions of the graphical language strewn across different
smp projects with different levels of language rigor. Still WIP, but
hey.

This means we are using third party libraries without limit in the
project, which is easy working from within a well used Python
environment \[2\]. smp\_audio is wrapping around a number of established
audio processing libraries: aubio, essentia, librosa, pyAudioAnalysis,
madmom, and a few more specialized ones. These are all well maintained
and generally easy to install.

The \`smp\_audio\` directory contains the wrapping code and glue to
provide the underlying API to the processing flows defined in the files
in the \`scripts\` directory.

## Install

Part one of the dependency pile is just the usual Python scientific
computing stack and can usually be installed as distribution packages.

On Linux / Ubuntu

``` example
sudo apt install python3-numpy python3-scipy python3-matplotlib python3-pandas python3-sklearn
```

or just using pip

``` example
sudo pip3 install numpy scipy matplotlib pandas sklearn
```

Part two is the audio and MIR specific packages, which I install using
pip directly

``` example
sudo pip3 install -r requirements.txt essentia librosa madmom aubio pyAudioAnalysis
```

On Mac OSX

Install essentia with their homebrew formula
<https://github.com/MTG/homebrew-essentia>,
<https://essentia.upf.edu/documentation/installing.html>

Then followed by

``` example
sudo pip3 install -r requirements.txt librosa madmon aubio pyAudioAnalysis
```

To finish, add the module path to your PYTHONPATH by running

``` example
export PYTHONPATH=/path/to/cloned/smp_audio:$PYTHONPATH
```

In case you come back to using the package, the export line above can be
added to your login profile (\~/.bashrc or similar).

## Quick start

The most interesting modes are **autoedit** and **automix**, two
variations of a similar task. Autoedit is designed as a tool for
session-based production styles. It takes a stereo recording of a
session on the input, computes a beat aligned segmentation into
different parts, and assembles these parts into an edit. It is run as

``` example
python3 scripts/automain.py autoedit --filenames /path/to/single/file.wav
```

Automaster is run with

``` example
python3 scripts/automain.py automaster --bitdepth 16 \
--filenames /file/that/should/be/mastered.wav \
--references /file/to/use/as/reference.wav
```

Calling the script with \`–help\` argument prints the parameters with
docstring and default values

``` example
python3 scripts/automain --help
```

The most important parameter is the approximate number of segments given
by \`–numsegs N\`. More segments in total make individual segments
shorter, making the edit more dynamic at the risk of being jumpy at a
sub-bar resolution. There's no limit to file sizes, durations etc in
principle but large files will exhaust your memory at some point. Can be
improved later.

The automix mode also compiles a set of audio segments into a single
large file. While autoedit works at track level durations of order \< 10
minutes, automix does so at mix level ones of order 1 hour.

``` example
python3 audio_sort_features.py --mode automix --sorter extractor_lowlevel.spectral_complexity.mean --filenames /path/to/textfile_one_wavfile_path_per_line.txt
```

Current assembly modes are primitive at best, autoedit uses random and
sequential segment order so far. With automix the segments are sorted by
particular feature value, which can be selected with the \`–sorter\`
argument.

TODO: Consolidation of randomly disparate functionality is
work-in-progress.

## Workflows

The scripts folder contains several workflows. Those that work are the
in `audio_sort_features.py` file that you can run and which has a help
to tell you how to call it.

**autoedit** (beat-tracking and segmentation based automatic editing and
arrangement to duration of choice from raw audio session that is shorter
or longer than output duration).

**autofeatures**, load file and return a set of features computed for
this file.

**autocover**, get autofeatures and map the feature values to an image
generation process to create a hi-res square graphics file that can be
used as a cover image for music on the internet, or whatever.

**automaster**, is wrapping the grandiose matchering library. matchering
transforms your target file according to the mastering signature
extracted from the reference file.

WIP **autovoice**

WIP **automix** (quantitative measure based playlist sequencing)

### slicing audio

Depending on your resources, it is often convenient to split very long
files into shorter parts of maybe an hour length.

sox command line w/ silence plugin

``` example
sox -V3 audiofile.wav audiofile_part_.wav silence -l  0 1 2.0 0.1%
```

aubio quiet

analyze audio and print the onset timestamps of silence and noise parts

``` example
aubio quiet filename.wav
```

aubio cut

aubiocut analyzes audio for different onset functions and can optionally
cut the file at each onset and save into a separate file each.

### downloading audio

**youtube-dl**, versatile youtube downloader, one of the best and nicest
computer programs in the world.

**ffmpeg**, another indispensable and super versatile tool for working
with audio, media, streams, and containers. it can do everything.

**soundscrape**, soundcloud and bandcamp downloader, similar to
youtube-dl but more narrow in scope and maybe less well maintained
recently \[3\]

``` example
sudo pip3 install SoundScrape
```

## Notes

### 2021-01-08 refactor api

refactoring for api integration

python3 *home/src/QK/smp\_audio/scripts/audio\_sort\_features.py
automaster –bitdepth 24 –filenames
data/GuitarRiff2\_50bpm-autoedit-10.wav –references
..*../../nextcloud/gt/work/automaster/refs/rae-sremmurd-notype.wav

### <span class="todo TODO">TODO</span> thumbnailing

provided by pyAudioAnalysis, running

``` example
python3 audioAnalysis.py thumbnail --input /path/to/file.wav
```

produces a thumbnail image and two thumbnail wav snippets in
/path/to/file\_thumbnail\*

### <span class="todo TODO">TODO</span> stream processing

switch the entire internal data flow to stream based processing and
implement batch versions as a separate option.

### Caching

Using dict of functions and joblib.Memory to cache all compute intensive
funtions. This is done at the calling level.

Librosa has its own caching mechanism, which is used in the librosa
specific code.

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

python3 scripts/aubio\_cut.py –mode scan –quiet –bufsize 2048 –hopsize
256 /home/lib/audio/work/tsx\_recur\_5/sco-base.wav

# footnotes

1.  citation needed

2.  The price is that it's a pain to install. My strategy is to not
    install all the requirements but run the script i want to run, and
    fix out the issues one by one until free.

3.  <https://github.com/Miserlou/SoundScrape>
