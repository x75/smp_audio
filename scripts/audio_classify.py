"""smp_audio.scripts.audio_classify

first prototype of frame based audio classification

- take frames from file stream reader or from the network via OSC
- classify frame timbre and genre
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import librosa

# MLP for Pima Indians Dataset Serialize to JSON and HDF5
# from keras.models import Sequential
# from keras.layers import Dense
from keras.models import model_from_json

from smp_audio.common_librosa import data_stream_librosa, data_stream_get_librosa
from smp_audio.common_aubio import data_stream_aubio, data_stream_get_aubio

def model_serialize(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def model_unserialize(modelfile='model.json'):
    # load json and create model
    json_file = open(modelfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(modelfile[:-5] + ".h5")
    print("Loaded model from disk")
    return loaded_model

def model_evaluate(loaded_model, X, Y):
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

def main(args):
    print(f'args {args}')

    # load model
    model = model_unserialize(args.modelfile)
    print(f'model {model}')
    
    # load model training data
    data = pd.read_csv('../notebooks/data-stream.csv')
    data = data.drop(['filename'],axis=1)

    genre_list = data.iloc[:, -1]
    # encoder = LabelEncoder()
    
    # scaler = StandardScaler()
    # X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
    scaler = pickle.load(open('../notebooks/scaler.pkl', 'rb'))
    
    # load audio data
    framesize = 1024
    src, sr = data_stream_librosa(
        filename=args.filename,
        frame_length=framesize,
        hop_length=framesize
    )
    X = []
    Y = []
    for blk_i, blk_y in enumerate(src):
        print(f'i {blk_i}, y {blk_y.shape}')

        if len(blk_y) < framesize: continue
        # D_block = librosa.stft(block_y, n_fft=framesize, hop_length=framesize, center=False)
        # D.append(D_block[:,0])
        y = blk_y
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        rmse = librosa.feature.rms(y=y, frame_length=framesize, hop_length=framesize, center=False)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=framesize, hop_length=framesize, center=False)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=framesize, hop_length=framesize, center=False)
            
        # D = np.array(D) #.reshape((-1, framesize/2 + 1))

        to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        # to_append += f' {g}'

        X = np.atleast_2d(to_append.split())
        # print(f'    X = {X}')
        X_s = scaler.transform(X)
        # print(f'    X_s = {X_s}')
        # y = encoder.fit_transform(genre_list)

        y_ = model.predict(X_s)

        # print(f'frame {blk_i} prediction {y_}')
        # print(f'frame {blk_i} prediction {np.argmax(y_)}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help="Audio file to analyze [None]", default=None)
    parser.add_argument('-m', '--modelfile', type=str, help="Model file to load [model.json]", default='model.json')
    args = parser.parse_args()

    main(args)
