"""smp_audio.common

Oswald Berthold, 2018

common data, embedding, transform functions for offline music and
multivariate timeseries processing

Abstract interface
- data_get_filename (librosa)
- data_load (aubio, librosa, essentia)
- data_stream (aubio, librosa, essentia)

- compute_onsets (aubio, librosa)
- compute_tempo_beats
- compute_music_extractor
- compute_segments (librosa, essentia)
- compute_chroma_librosa
- compute_tempogram_librosa
- compute_onsets_librosa
- compute_beats_librosa
- compute_beats_mix

- beats_to_bpm
"""
import os
from smp_base.impl import smpi
# import logging
logging = smpi('logging')

DEBUG=True

def myprint(*args, **kwargs):
    if not DEBUG: return
    print(*args, **kwargs)

def get_module_logger(modulename = 'experiment', loglevel = logging.INFO):
    """get a logging.logger instance with reasonable defaults

    Create a new logger and configure its name, loglevel, formatter
    and output stream handling.
    1. initialize a logger with name from arg 'modulename'
    2. set loglevel from arg 'loglevel'
    3. configure matching streamhandler
    4. set formatting swag
    5. return the logger
    """
    loglevels = {'debug': logging.DEBUG, 'info': logging.INFO, 'warn': logging.WARNING}
    if type(loglevel) is str:
        try:
            loglevel = loglevels[loglevel]
        except:
            loglevel = logging.INFO
            
    if modulename.startswith('smp_graphs'):
        modulename = '.'.join(modulename.split('.')[1:])
        
    if len(modulename) > 20:
        modulename = modulename[-20:]
    
    # create logger
    logger = logging.getLogger(modulename)
    logger.setLevel(loglevel)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)

    # create formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(levelname)8s: %(name)20s: %(message)s')
    # formatter = logging.Formatter('{levelname:8}s: %(name)20s: %(message)s')
    # formatter = logging.Formatter('%(name)s: %(levelname)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    # suppress double log output 
    logger.propagate = False
    return logger

def autoedit_get_count(rootdir='./', verbose=False):
    """autoedit get count

    load autoedit count from file, if it does not exist, init zero and
    save to file.
    """
    autoedit_count_datadir = os.path.join(rootdir, 'data/autoedit')
    autoedit_count_filename = os.path.join(autoedit_count_datadir, 'autoedit-count.txt')
    if os.path.exists(autoedit_count_filename):
        autoedit_count = int(open(autoedit_count_filename, 'r').read().strip())
        if verbose:
            print(f'autoedit_get_count from file {autoedit_count}, {type(autoedit_count)}')
    else:
        if verbose:
            print(f'autoedit_get_count file not found, initializing')
        autoedit_count = 0
        makedirs_ok = os.makedirs(autoedit_count_datadir, exist_ok=True)
        if verbose:
            print(f'autoedit_get_count autoedit_count = {autoedit_count}')
            print(f'autoedit_get_count autoedit_datadir = {autoedit_count_datadir}')
            print(f'autoedit_get_count autoedit_datadir created {makedirs_ok}')
        
    autoedit_count_new = autoedit_count + 1
    f = open(autoedit_count_filename, 'w')
    f.write(f'{autoedit_count_new}\n')
    f.flush()
    return autoedit_count

def autocount(basedir='./', autoname='autoedit', verbose=False):
    """autocount

    Get a count on how many times a mode has been run in a given
    context.

    Load the session count for a given session name from file, if it
    does not exist, initialize and create the path to the file.
    """
    # datadir = os.path.join(basedir, 'data', autoname)
    datadir = os.path.join(basedir, autoname)
    filename = os.path.join(datadir, 'count.txt')
    if os.path.exists(filename):
        count = int(open(filename, 'r').read().strip())
        if verbose:
            print(f'autocount = {count}, {type(count)}')
    else:
        if verbose:
            print(f'autocount does not exist, initializing')
        count = 0
        makedirs_ok = os.makedirs(datadir, exist_ok=True)
        if verbose:
            print(f'autocount count = {count}')
            print(f'autocount datadir = {datadir}')
            print(f'autocount datadir created {makedirs_ok}')
        
    count += 1
    f = open(filename, 'w')
    f.write(f'{count}\n')
    f.flush()
    return count

def autofilename(args):
    """autofilename

    Create an output filename from the first input file, the mode, the
    count and the seed.

    Return output filename stub without extension.
    """
    dirname = os.path.dirname(args.filenames[0])
    filename = os.path.basename(args.filenames[0])
    filename_noext = ".".join(filename.split('.')[:-1])
    count = autocount(basedir=args.rootdir, autoname=args.mode)
    # if args.verbose:
    #     print(f"smp_audio.common.autofilename filenames {args.filenames}")
    #     print(f"smp_audio.common.autofilename dirname {dirname}")
    #     print(f"smp_audio.common.autofilename filename {filename}")
    #     print(f"smp_audio.common.autofilename filename_noext {filename_noext}")
    #     print(f"smp_audio.common.autofilename count {count}")
    filename_export = os.path.join(
        dirname,
        f'{filename_noext}-{args.mode}-{count}'
    )
    if args.verbose:
        print(f"smp_audio.common.autofilename = {filename_export}")
    return filename_export
