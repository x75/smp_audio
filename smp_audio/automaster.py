import random, os, json, codecs
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

def main_automaster(args, **kwargs):
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
        # filename_export = f"{target[:-4]}_master{args.bitdepth}.wav"
        filename_export = os.path.join(
            args.rootdir,
            os.path.basename(args.filename_export) + ".wav")
        
        if args.verbose:
            print(f"main_automaster target {target}, reference {reference}, bitdepth {args.bitdepth}")
            print(f"main_automaster outputs {filename_export}")

        if not os.path.exists(target) or not os.path.exists(reference):
            print(f"main_automaster target {target} or reference {reference} doesnt exist")
            continue
        
        if args.bitdepth == 16:
            result = mg.pcm16(filename_export)
        else:
            result = mg.pcm24(filename_export)
            
        mg.process(
            # The track you want to master
            target=target,
            # Some "wet" reference track
            reference=reference,
            # Where and how to save your results
            results=[result],
        )

        # print(f"automaster result = {result}")
        
        automaster_results['targets'].append(target)
        automaster_results['references'].append(reference)
        automaster_results['results'].append(filename_export)

    results = {
        'data': {
            'output_files': [
                {'format': args.outputs[0],
                 'filename': os.path.basename(automaster_results['results'][0])}
            ],
        }
    }
        
    filename_result = os.path.join(
        args.rootdir,
        os.path.basename(args.filename_export) + ".json")

    # this saves the array in .json format
    json.dump(
        results,
        codecs.open(filename_result, 'w', encoding='utf-8'),
        # separators=(',', ':'),
        # sort_keys=True,
        # indent=4,
        # cls=NumpyEncoder,
    )

    if 'task' in kwargs:
        kwargs['task'].set_done(result_location=os.path.basename(args.filename_export) + ".json")
    
    return results
