import argparse

from smp_audio.util import args_to_dict
from smp_audio.assemble_pydub import track_assemble_from_segments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default=None, help="Textfile with list of segments one per line")

    args = parser.parse_args()
    kwargs = args_to_dict(args)

    if kwargs['filename'] is not None:
        kwargs['files'] = [_.rstrip() for _  in open(kwargs['filename'], 'r').readlines()]
    else:
        kwargs['files'] = None
        
    ret = track_assemble_from_segments(**kwargs)

    print(('returned {0}'.format(ret)))
