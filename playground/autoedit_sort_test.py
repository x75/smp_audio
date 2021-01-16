
import joblib

from smp_audio.segments import track_assemble_from_segments

def get_features_sort_features_assemble(d):
    print(f'd {d.keys()}')

def main():
    d = joblib.load(open('/media/x75/swurl/lib/audio/work/slopper/slop_07.00/data/1_Audio_Track-autoedit-0.pkl', 'rb'))

    seq = get_features_sort_features_assemble(d)

    track_assemble_from_segments(seq)
    
    # print(f'{d["l6_merge"]["files"]}')
    for f in d["l6_merge"]["files"]:
        get_features()

if __name__ == '__main__':
    main()
