shopt -s nullglob
let playlist_duration_ms=0
for song_file in *.{mp3,ogg,m4a,flac,wav}; do
	echo $playlist_duration_ms pre
	echo $song_file
  	# playlist_duration_ms=$(expr $playlist_duration_ms + $(mediainfo --Inform="Audio;%Duration%" "$song_file"))
	echo dur $(mediainfo --Inform="Audio;%Duration%" "$song_file")
  	playlist_duration_ms=$(expr $playlist_duration_ms + $(mediainfo --Inform="Audio;%Duration%" "$song_file"))
	echo $playlist_duration_ms post
done
shopt -u nullglob

let playlist_duration_secs=$(expr $playlist_duration_ms / 1000)
let playlist_duration_mins=$(expr $playlist_duration_ms / 60000)
let playlist_duration_remaining_secs=$(expr $playlist_duration_secs - $(expr $playlist_duration_mins \* 60))

echo $playlist_duration_mins minutes, $playlist_duration_remaining_secs seconds

