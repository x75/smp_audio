import random, os
import matchering as mg

# TODO: create automaster_conf_default
automaster_conf = {
    'default': {
        'filenames': [],
        'mode': 'automaster',
        'references': [],
        'rootdir': './',
        'seed': 1234,
        'bitdepth': 24,
        'verbose': False,
        'outputs': ['wav']
    },
}
automaster_conf_default = automaster_conf['default']

def main_automaster(args):
    # convert from auto args template
    # TODO: if len(targets) == len(references) then match up, otherwise random choice, otherwise single

    # print(f"main_automaster args {args}")
    # if type(args.references) in [list]:
    #     args.references = args.references[0]
    
    if len(args.references) == 1:
        references = [args.references[0] for _ in range(len(args.filenames))]
    elif len(args.references) == len(args.filenames):
        references = args.references
    else:
        references = [random.choice(args.references) for _ in range(len(args.filenames))]
    if args.verbose:
        print(f"main_automaster references {references}")

    automaster_results = {
        'targets': [],
        'references': [],
        'results': [],
    }

    # print(f"main_automaster filenames {args.filenames}")
    # print(f"main_automaster references {references}")
    
    for target_i, target in enumerate(args.filenames):
        reference = references[target_i]
        # result_filename = f"{target[:-4]}_master{args.bitdepth}.wav"
        result_filename = os.path.join(
            args.rootdir,
            os.path.basename(args.filename_export) + ".wav")
        
        if args.verbose:
            print(f"main_automaster target {target}, reference {reference}, bitdepth {args.bitdepth}")
            print(f"main_automaster outputs {result_filename}")

        if not os.path.exists(target) or not os.path.exists(reference):
            print(f"main_automaster target {target} or reference {reference} doesnt exist")
            continue
        
        if args.bitdepth == 16:
            result = mg.pcm16(result_filename)
        else:
            result = mg.pcm24(result_filename)
            
        mg.process(
            # The track you want to master
            target=target,
            # Some "wet" reference track
            reference=reference,
            # Where and how to save your results
            results=[result],
        )

        automaster_results['targets'].append(target)
        automaster_results['references'].append(reference)
        automaster_results['results'].append(result)
