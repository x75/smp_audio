# smp\_audio
## smp\_audio.autopapi

quick start from scratch

`git clone https://github.com/x75/smp_audio`

change into smp_audio directory

`cd smp_audio`

initialize and setup your config

`cp config.py.dist config.py`

then open `config.py` with your favorite editor and paste your api_key (TODO: test key)

get test files in place, we providing one that you can get on the command line with wget

`wget https://github.com/x75/smp_data/blob/master/data/smp_audio/tests/TESTFILE.mp3?raw=true`

or curl

`curl https://github.com/x75/smp_data/blob/master/data/smp_audio/tests/TESTFILE.mp3?raw=true --output TESTFILE.mp3`

with that file in place you can run the tool suite

starting with **autoedit**, we type / paste

`python3 scripts/autoapi.py autoedit --filenames TESTFILE.mp3 --duration 40 --numsegs 60`

this will check your api_key with the server, upload the input file,
then run the requested processing (autoedit) on the uploaded copy of
your input file.

the processing returns immediately with a task url that is polled
for completion. on completion, a file url is returned. that is a json
file with the session description. the 'output_files' field of the
session description is a list of output files produced. these are
downloaded and the script exits. now the output files should be in
your local work directory.

next could be **automaster**, which needs a target input file (here it's
autoedit-14 converted to mp3 from a previous autoedit run) and a
reference input file TESTFILE_ref.mp3. the processing is run with

`python3 scripts/autoapi.py automaster --filenames TESTFILE-autoedit-14.wav --references TESTFILE_ref.mp3`

that returns another json session description with one output file,
the input filename with -automaster-N.wav appended. check and listen.

when you like it, you can create a cover pic for the track with
autocover. autocover has a double role as autofeatures. the task is to
run a standard frame-based analysis on the input file, returning a
fixed / configurable dictionary of features with a variable number of
analysis frames for each key. a simple autocover theme is to plot the
feature matrix as an image working with plotting parameters like the
colormap and so on.

run autofeatures / autocover like

`python3 scripts/autoapi.py autocover --filenames TESTFILE-autoedit-14-automaster-5.wav --outputs jpg`

this should return the json session description, with another json
output file that contains the feature matrix. open in your favorite
environment

`open TESTFILE-autoedit-14-automaster-5-autcover-N.json`

including the `--outputs jpg` option produces a the image plotted
feature matrix in jpg format, examine with

`open TESTFILE-autoedit-14-automaster-5-autcover-N.jpg`

this was just the TESTFILE. if any/all of this worked, then just move
on into the directory containing your real data. make sure to backup.

then call autoapi.py like before, only with different input files

`python3 /path/to/smp_audio/scripts/autoapi.py autoedit --filenames mywonkyfilename.wav`

thanks for checking by. feedback welcome, write a mail or even better,
leave an issue.
