"""smp_audio.common_api

define the api and map calls to one or more implementation libraries,
merge results if required and return final prediction

### data_load

fix streaming vs. buffered load issue

### data_stream

### etc
"""
from smp_base.impl import smpi
# import logging
logging = smpi('logging')
common_aubio = smpi('common_aubio')
common_librosa = smpi('common_librosa')
common_essentia = smpi('common_essentia')

DEBUG=True

def data_load(**kwargs):
    if common_aubio is not None:
        print('aubio loaded {0}'.format(common_aubio))
        return common_aubio.data_load_aubio(**kwargs)
    if common_librosa is not None:
        print('librosa loaded {0}'.format(common_librosa))
        return common_librosa.data_load_librosa(**kwargs)
    if common_essentia is not None:
        print('essentia loaded {0}'.format(common_essentia))
        return common_essentia.data_load_essentia(**kwargs)

if __name__ == '__main__':
    src = data_load(filename='out.wav')
    print('src = {0}'.format(src))
