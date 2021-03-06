"""assemble_pydub

Assemble track from a list of input segments using pydub audio library.

Current assembly modes:
- track_assemble_from_segments: select segment randomly until song length greater than desired length
- track_assemble_from_segments_sequential: create song from concatenating segments in list order sequentially
- track_assemble_from_segments_sequential_scale: assemble song sequentially squeezing into desired length
"""
from pprint import pformat
import os, random
import pydub
import numpy as np

files_default = [
    "default.wav"
]

def segfiles_to_segs(files):
    """segfiles_to_segs

    Convert list of input filenames into list of loaded pydub AudioSegments
    """
    segs = []
    # print(f'segfiles_to_segs files {pformat(files)}')
    for file_ in files:
        # print(f'    segfiles_to_segs file {file_}')
        if file_.endswith('.mp3'):
            try:
                seg_ = pydub.AudioSegment.from_mp3(file_)
                # segs.append(seg_)
            except Exception as e:
                print('Exception', e)
        elif file_.endswith('.ogg'):
            try:
                seg_ = pydub.AudioSegment.from_ogg(file_)
                # segs.append(seg_)
            except Exception as e:
                print('Exception', e)
        elif file_.endswith('.wav'):
            try:
                seg_ = pydub.AudioSegment.from_wav(file_)
                # segs.append(seg_)
            except Exception as e:
                print('Exception', e)
        else:
            try:
                seg_ = pydub.AudioSegment.from_file(file_)
                # segs.append(seg_)
            except Exception as e:
                print('Exception', e)

        # seg loaded, append
        segs.append(seg_)

    return segs

def track_assemble_kwargs_to_args(**kwargs):
    """track_assemble_kwargs_to_args

    Convert track_assemble kwargs to in-function variables
    """
    if 'filename_export' in kwargs:
        filename_export = kwargs['filename_export'] # '22-assembled.wav'
    else:
        filename_export = "track_assemble_from_segments_sequential.wav"

    if kwargs['files'] is None:
        files = files_default
    else:
        files = kwargs['files']
    # print('track_assemble_kwargs_to_args files = {0}'.format(pformat(files)))
          
    # parameter
    if kwargs['duration'] is None:
        duration = 180
    else:
        duration = kwargs['duration']

    # parameter
    if 'crossfade' in kwargs:
        crossfade = kwargs['crossfade']
    else:
        crossfade = 10
        
    return files, filename_export, duration, crossfade
        
def track_assemble_from_segments(**kwargs):
    """track_assemble_from_segments

    Assemble a *track* of *duration* in seconds from a list of
    segment-*files* using pydub and export the result as a wav.

    Sequence is sampled randomly from list of segments.
    """
    files, filename_export, desired_duration, crossfade = track_assemble_kwargs_to_args(**kwargs)
    rootdir = kwargs['rootdir']
    verbose = kwargs['verbose']
    
    # make pydub segment list
    # segs = [pydub.AudioSegment.from_wav(file_) for file_ in files]

    segs = segfiles_to_segs(files)

    # init empty song
    song = pydub.AudioSegment.empty()
    # song = random.randrange(0, len(files))

    # export
    filename_export_txt = os.path.join(
        rootdir,
        filename_export + ".txt")
    f = open(filename_export_txt, 'w')
    f.write('{0},{1},{2}\n'.format('index', 'duration', 'file'))

    # while song duration is less than desired duration, select random
    # segment and append to song
    seg_s = []
    i = 0
    while song.duration_seconds < desired_duration:
        seg_ = random.randrange(0, len(segs))
        # print('seg_ {0}'.format(seg_))
        f.write('{0},{1:.2f},{2}\n'.format(
            seg_, segs[seg_].duration_seconds, files[seg_]))

        if i > 1 and song.duration_seconds > 0.01 and segs[seg_].duration_seconds > 0.01:
            # crossfade_eff = 10
            crossfade_eff = crossfade
        else:
            crossfade_eff = 0

        if verbose:
            print('    assemble rnd crossfade {0}'.format(crossfade_eff))
            print('    assemble rnd song duration {0}'.format(song.duration_seconds))
            print('    assemble rnd seg_ duration {0}\n    {1}'.format(segs[seg_].duration_seconds, files[seg_]))
        
        # if len(seg_s) > 0:
        #     # crossfade_eff = 10
        #     crossfade_eff = crossfade
        # else:
        #     crossfade_eff = 0
        
        song = song.append(segs[seg_], crossfade=crossfade_eff)
        if verbose:
            print('    assemble rnd song duration {0:.2f}'.format(song.duration_seconds))
        seg_s.append(seg_)
        
        # increase counter
        i = i + 1
        
    # export the edit sequence list
    f.flush()
    f.close()
    
    # export the assembled song
    filename_export_wav = os.path.join(
        rootdir,
        filename_export + ".wav")
    if verbose:
        print('assemble song duration {0}'.format(song.duration_seconds))
        print('assemble song export to {0}'.format(filename_export_wav))
    song.export(filename_export_wav, format='wav')

    # FIXME: change this to export_filename, duration, segs, numsegs
    return {
        'filename_export_wav': filename_export_wav,
        'filename_export_txt': filename_export_txt,
        'final_duration': song.duration_seconds,
        'seg_s': seg_s
    }

def track_assemble_from_segments_sequential_scale(**kwargs):
    """track_assemble_from_segments_sequential_scale

    Assemble a *track* from a list of segment-*files* in sequential
    order but *contracting* using pydub and export the result as a wav.

    Sequence is sampled in input-list order.
    """
    # wrapper
    files, filename_export, duration, crossfade = track_assemble_kwargs_to_args(**kwargs)
    rootdir = kwargs['rootdir']
    verbose = kwargs['verbose']

    # make pydub segment list
    segs = segfiles_to_segs(files)

    # duratio foo
    segs_duration_seconds = [seg_.duration_seconds for seg_ in segs]
    scale = 0.5
    scale = duration / np.sum(segs_duration_seconds)
    
    # init empty song
    song = pydub.AudioSegment.empty()

    # loop all segements
    seg_s = []

    # track bpm, maxlen2beatmultiples
    filename_export_txt = os.path.join(
        rootdir,
        filename_export + ".txt")
    f = open(filename_export_txt, 'w')
    f.write('{0},{1},{2}\n'.format('index', 'duration', 'file'))
        
    # while song.duration_seconds < desired_duration:
    for i, seg_ in enumerate(segs):
        if np.random.uniform(0, 1) > scale:
            # print('skipping')
            continue
        
        if i > 1 and song.duration_seconds > 0.01 and seg_.duration_seconds > 0.01:
            # crossfade_eff = 10
            crossfade_eff = crossfade
        else:
            crossfade_eff = 0

        if verbose:
            print('    assemble seq scl crossfade {0}'.format(crossfade_eff))
            print('    assemble seq scl song duration {0}'.format(song.duration_seconds))
            print('    assemble seq scl seg_ duration {0:.2f} {1}'.format(seg_.duration_seconds, files[i]))
        
        # song = song.append(seg_.apply_gain_stereo(-1, -1), crossfade=crossfade_eff)
        f.write('{0},{1:.2f},{2}\n'.format(i, seg_.duration_seconds, files[i]))
        song = song.append(seg_, crossfade=crossfade_eff)
        seg_s.append(i)
                
    f.flush()
    f.close()
    
    # files_snips = [pydub.AudioSegment.from_file(_.rstrip()) for _ in open('/home/lib/audio/work/fm_2019_sendspaace/data/files_snips.txt', 'r').readlines()]
    # insert_pos = np.random.normal(np.cumsum([[song.duration_seconds / len(files_snips)] * len(files_snips)]), scale=100).tolist()

    # random.shuffle(files_snips)
    
    # for i, insert_pos_ in enumerate(insert_pos):
    #     print('inserts seg {1} at pos {0}'.format(files_snips[i], insert_pos_))
    #     song = song.overlay(files_snips[i], position=insert_pos_*1000) # , gain_during_overlay=-3
    # song_ = song.apply_gain_stereo(-0.5, -0.5)
    
    # export the assembled song
    filename_export_wav = os.path.join(
        rootdir,
        filename_export + ".wav")
    if verbose:
        print(f"assemble exporting song duration {song.duration_seconds:.2f} to {filename_export_wav}")

    song.export(filename_export_wav, format='wav')

    return {
        'filename_export_wav': filename_export_wav,
        'filename_export_txt': filename_export_txt,
        'final_duration': song.duration_seconds,
        'seg_s': seg_s
    }

# DEPRECATED
def track_assemble_from_segments_sequential(**kwargs):
    """track_assemble_from_segments_sequential

    Assemble a *track* from a list of segment-*files* in sequential
    order using pydub and export the result as a wav.

    Sequence is sampled in input-list order.
    """
    # wrapper
    files, filename_export, duration, crossfade = track_assemble_kwargs_to_args(**kwargs)
    rootdir = kwargs['rootdir']
        
    # make pydub segment list
    segs = segfiles_to_segs(files)
    
    # init empty song
    song = pydub.AudioSegment.empty()

    # loop all segements
    seg_s = []
    #    while song.duration_seconds < desired_duration:
    for i,seg_ in enumerate(segs):
        print('seg_ {0} / {1}'.format(seg_.duration_seconds, files[i]))
        if i > 1:
            # crossfade_eff = 10
            crossfade_eff = crossfade
        else:
            crossfade_eff = 0
        # song = song.append(seg_.apply_gain_stereo(-1, -1), crossfade=crossfade_eff)
        song = song.append(seg_, crossfade=crossfade_eff)
        seg_s.append(seg_)
        # seg_ = random.randrange(0, len(segs))
        # print('seg_ {0}'.format(seg_))
        # Song = song.append(segs[seg_], crossfade=0)
        print('song duration {0}'.format(song.duration_seconds))
        
    files_snips = [pydub.AudioSegment.from_file(_.rstrip()) for _ in open('/home/lib/audio/work/fm_2019_sendspaace/data/files_snips.txt', 'r').readlines()]
    insert_pos = np.random.normal(np.cumsum([[song.duration_seconds / len(files_snips)] * len(files_snips)]), scale=100).tolist()

    random.shuffle(files_snips)
    
    for i, insert_pos_ in enumerate(insert_pos):
        print('inserts seg {1} at pos {0}'.format(files_snips[i], insert_pos_))
        song = song.overlay(files_snips[i], position=insert_pos_*1000) # , gain_during_overlay=-3
    
    # export the song
    print('song duration {0}'.format(song.duration_seconds))
    # song_ = song.apply_gain_stereo(-0.5, -0.5)
    print('song export to {0}'.format(filename_export))
    song.export(filename_export, format='wav')

    return {'filename_export': filename_export, 'final_duration': song.duration_seconds, 'seg_s': seg_s}

if __name__ == '__main__':
    print('smp_music.smp_audio.assemble_pydub')

    # track_assemble_from_segments()

