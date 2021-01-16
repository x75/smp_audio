import argparse

import matchering as mg

# Sending all log messages to the default print function
# Just delete the following line to work silently
# mg.log(print)

targets = {
    'trk006-5-voice-4': {'filename': 'trk006-5-voice-4.wav',},
    'trk008-6-voice-4': {'filename': 'trk008-6-voice-4.wav',},
    'speedtest': {'filename': 'speedtest.wav',},
    '11.84.0.-1.0-1.1-1_5072.884286-autoedit-3': {'filename': '11.84.0.-1.0-1.1-1_5072.884286-autoedit-3.wav'},
    '11.84.0.-1.0-1.1-1_5072.884286-autoedit-11': {'filename': '11.84.0.-1.0-1.1-1_5072.884286-autoedit-11.wav'},
    '11.84.0.-1.0-1.1-1_5072.884286-autoedit-12': {'filename': '11.84.0.-1.0-1.1-1_5072.884286-autoedit-12.wav'},
}

references = {
    'shluff2_comp': {'filename': '/home/lib/audio/work/tsx_recur_2/shluff2_comp.wav'},
    'sampa_daratofly': {'filename': '/home/lib/audio/work/tsx_4_sco_1/mastering-ref-1-sampa-daretofly.wav'},
    'dj-stingray-cryptic': {'filename': 'matchering-dj-stingray-cryptic.wav'},
    '11.84.0.-1.0-1.1-1': {'filename': '/media/x75/swurl/lib/audio/work/fm_2020_11.84.0.-1.0-1.1-1/master/export/11.84.0-1.0-1.1-1_2133-autoedit-1.wav'}
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default=None, help='Target file name, which one to master [None]')
    parser.add_argument('-r', '--reference', type=str, default=None, help='Reference file name, which one to extract master from [None]')
    parser.add_argument('-b', '--bitdepth', type=int, default=16, help='Bit depth to work at [16]')
    
    # target_k = 'trk006-5-voice-4'
    # target_k = 'trk008-6-voice-4'
    # target_k = 'speedtest'
    args = parser.parse_args()
    if args.target is None:
        target_k_default = '11.84.0.-1.0-1.1-1_5072.884286-autoedit-3'
        args.target = targets[target_k_def]['filename']
    if args.reference is None:
        ref_k_default = '11.84.0.-1.0-1.1-1'
        args.reference = references[ref_k_default]['filename']

    target_k = args.target[:-4]
    ref_k = args.reference[:-4]

    print(f'processing')
    print(f'    target {args.target}')
    print(f'       ref {args.reference}')
        
    # reference = references['sampa_daratofly']['filename']
    # reference = references['dj-stingray-cryptic']['filename']

    results = []
    if args.bitdepth == 24:
        results.append(mg.pcm24('{0}_master_24bit.wav'.format(target_k)))
    else:
        results.append(mg.pcm16('{0}_master_16bit.wav'.format(target_k)))
    
    mg.process(
        # The track you want to master
        target=args.target,

        # Some "wet" reference track
        # reference=,
        reference=args.reference,

        # Where and how to save your results
        results=results,
    )
