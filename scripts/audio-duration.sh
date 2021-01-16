#!/bin/bash

# compute total duration of audio in a directory in milliseconds

workdir="."
if [ "$1" != "" ]; then
    workdir=$1
    echo "audio_duration directory ${workdir}"
else
    echo "audio_duration, no directory given, using default ${workdir}"
fi

shopt -s nullglob
let playlist_duration_ms=0
for song_file in ${workdir}/*.{mp3,mp4,ogg,m4a,flac,wav,aiff,aif}; do
    # echo enter \#\#\#\#\#
    echo file `basename $song_file`
    # echo duration accumulated $playlist_duration_ms pre
    # playlist_duration_ms=$(expr $playlist_duration_ms + $(mediainfo --Inform="Audio;%Duration%" "$song_file"))
    song_duration=$(mediainfo --Inform="Audio;%Duration%" "$song_file")
    song_duration_int=${song_duration%.*}
    echo "    duration file  = ${song_duration_int}ms"
    playlist_duration_ms=$(expr $playlist_duration_ms + $song_duration_int)
    echo "    duration total = ${playlist_duration_ms}ms"
done
shopt -u nullglob

let playlist_duration_secs=$(expr $playlist_duration_ms / 1000)
let playlist_duration_mins=$(expr $playlist_duration_ms / 60000)
let playlist_duration_remaining_secs=$(expr $playlist_duration_secs - $(expr $playlist_duration_mins \* 60))

echo $playlist_duration_mins minutes, $playlist_duration_remaining_secs seconds

