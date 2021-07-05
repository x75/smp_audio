import soundfile as sf

def data_load_soundfile():
    y_48, sr_48 = soundfile.read(filename_48, always_2d=True)
    y_48 = y_48.T
    return 
