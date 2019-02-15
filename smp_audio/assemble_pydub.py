from pprint import pformat
import pydub
import random
import numpy as np

def track_assemble_from_segments(**kwargs):
    if kwargs['files'] is None:
        files = [
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg21.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg22.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg23.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg24.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg25.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg26.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg27.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg28.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg2.wav",
        ]
    else:
        files = kwargs['files']
    print('files = {0}'.format(pformat(files)))
          
    # parameter
    if kwargs['duration'] is None:
        desired_duration = 180
    else:
        desired_duration = kwargs['duration']
          
    # make pydub segment list
    # segs = [pydub.AudioSegment.from_wav(file_) for file_ in files]
    segs = []
    for file_ in files:
        if file_.endswith('.mp3'):
            try:
                seg_ = pydub.AudioSegment.from_mp3(file_)
                segs.append(seg_)
            except Exception as e:
                print('Exception', e)
        elif file_.endswith('.ogg'):
            try:
                seg_ = pydub.AudioSegment.from_ogg(file_)
                segs.append(seg_)
            except Exception as e:
                print('Exception', e)
        elif file_.endswith('.wav'):
            try:
                seg_ = pydub.AudioSegment.from_wav(file_)
                segs.append(seg_)
            except Exception as e:
                print('Exception', e)
            
    # init empty song
    song = pydub.AudioSegment.empty()
    # song = random.randrange(0, len(files))

    # while song duration is less than desired duration, select random segment and append to song
    seg_s = []
    while song.duration_seconds < desired_duration:
        seg_ = random.randrange(0, len(segs))
        print('seg_ {0}'.format(seg_))
        song = song.append(segs[seg_], crossfade=0)
        print('song duration {0}'.format(song.duration_seconds))
        seg_s.append(seg_)
    #
    filename_export = '22-assembled.wav'
        
    # export the song
    song.export(filename_export, format='wav')

    return {'filename_export': filename_export, 'final_duration': song.duration_seconds, 'seg_s': seg_s}

def track_assemble_from_segments_sequential(**kwargs):
    if kwargs['files'] is None:
        files = [
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg21.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg22.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg23.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg24.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg25.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg26.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg27.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg28.wav",
            "/home/src/QK/data/sound-arglaaa-2018-10-25/22-seg2.wav",
        ]
    else:
        files = kwargs['files']
    print('files = {0}'.format(pformat(files)))
          
    # parameter
    if kwargs['duration'] is None:
        desired_duration = 180
    else:
        desired_duration = kwargs['duration']
          
    # make pydub segment list
    # segs = [pydub.AudioSegment.from_wav(file_) for file_ in files]

    
    segs = []
    for file_ in files:
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
                
        segs.append(seg_)
            
    # init empty song
    song = pydub.AudioSegment.empty()
    # song = random.randrange(0, len(files))

    # while song duration is less than desired duration, select random segment and append to song
    seg_s = []
    #    while song.duration_seconds < desired_duration:
    for i,seg_ in enumerate(segs):
        print('seg_ {0} / {1}'.format(seg_.duration_seconds, files[i]))
        if i > 1:
            crossfade = 0 # 10
        else:
            crossfade = 0
        # song = song.append(seg_.apply_gain_stereo(-1, -1), crossfade=crossfade)
        song = song.append(seg_, crossfade=crossfade)
        seg_s.append(seg_)
        # seg_ = random.randrange(0, len(segs))
        # print('seg_ {0}'.format(seg_))
        # song = song.append(segs[seg_], crossfade=0)
        print('song duration {0}'.format(song.duration_seconds))
    #
    filename_export = 'sendspaace-assembled.wav'

    files_snips = [pydub.AudioSegment.from_file(_.rstrip()) for _ in open('/home/lib/audio/work/fm_2019_sendspaace/data/files_snips.txt', 'r').readlines()]
    insert_pos = np.random.normal(np.cumsum([[song.duration_seconds / len(files_snips)] * len(files_snips)]), scale=100).tolist()

    random.shuffle(files_snips)
    
    for i, insert_pos_ in enumerate(insert_pos):
        print('inserts seg {1} at pos {0}'.format(files_snips[i], insert_pos_))
        song = song.overlay(files_snips[i], position=insert_pos_*1000) # , gain_during_overlay=-3
    
    # export the song
    print('song duration {0}'.format(song.duration_seconds))
    # song_ = song.apply_gain_stereo(-0.5, -0.5)
    song.export(filename_export, format='wav')

    return {'filename_export': filename_export, 'final_duration': song.duration_seconds, 'seg_s': seg_s}

if __name__ == '__main__':
    print('smp_music.slurp.assemble_pydub')

    # track_assemble_from_segments()

