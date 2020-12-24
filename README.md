# music-viz
Fast music visualization to aid with transcription

### Dependencies
Python dependencies are listed in requirements.txt. `ffmpeg` is the only other dependency.

### Computing the wavelet transform
```
cericson@DESKTOP:~/music-viz$ python gabor_transform.py audio/Light\ My\ Fire.mp3 lightmyfire.pkl --pitch-oversampling 5
Loading audio data...
Computing octaves of transform: 100%|████████████████████████████████| 8/8 [00:12<00:00,  1.52s/it]
Elapsed: 12.153 s
```

### Rendering the transform
```
cericson@DESKTOP:~/music-viz$ python render_transform.py lightmyfire.pkl lightmyfire.png --horizontal --height 400 --width 1500 --end-time 10 --pitch-tick-origin F#0 --min-freq F#4 --max-freq F#7 --time-grid 0.83,130,4 --color-pitch
```

The resulting image:
![lightmyfire.png](/lightmyfire.png)

### Miscellanea
CLI documentation for each of the transform computation and rendering scripts can be accessed via `python <script name>.py -h`
