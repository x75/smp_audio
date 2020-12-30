shopt -s nullglob
let playlist_duration_ms=0
for song_file in *.{mp3,mp4,ogg,m4a,flac,wav,aiff,aif}; do
	echo enter \#\#\#\#\#
	echo     song file         $song_file
	echo duration accumulated $playlist_duration_ms pre
  	# playlist_duration_ms=$(expr $playlist_duration_ms + $(mediainfo --Inform="Audio;%Duration%" "$song_file"))
	song_duration=$(mediainfo --Inform="Audio;%Duration%" "$song_file")
	song_duration_int=${song_duration%.*}
	echo     song duration dur $song_duration_int
  	playlist_duration_ms=$(expr $playlist_duration_ms + $song_duration_int)
	echo duration accumulated $playlist_duration_ms post
done
shopt -u nullglob

let playlist_duration_secs=$(expr $playlist_duration_ms / 1000)
let playlist_duration_mins=$(expr $playlist_duration_ms / 60000)
let playlist_duration_remaining_secs=$(expr $playlist_duration_secs - $(expr $playlist_duration_mins \* 60))

echo $playlist_duration_mins minutes, $playlist_duration_remaining_secs seconds

