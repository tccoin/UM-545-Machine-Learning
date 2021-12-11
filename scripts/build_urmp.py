import os
import sys
import ffmpeg

dataset_path = 'data/urmp/video/'
frames_path = 'data/urmp/frames/'
audio_path = 'data/urmp/audio/'
dirs = os.listdir(dataset_path)
dirs = [x for x in dirs if os.path.isdir(dataset_path+x)]

for dir in dirs:
    file_names = os.listdir(dataset_path+dir)
    sep_audio_files = [x for x in file_names if x[:5] == 'AuSep']
    if not os.path.exists(frames_path+dir+'.mp4'):
        os.mkdir(frames_path+dir+'.mp4')
    video_path = '{}{}/Vid_{}.mp4'.format(dataset_path, dir, dir)
    mixaudio_path = '{}{}/AuMix_{}.wav'.format(dataset_path, dir, dir)
    (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=1)
        .filter('scale', -1, 630)
        .trim(start=0, end=45)
        .setpts('PTS-STARTPTS')
        .output(frames_path+dir+'.mp4/%d.jpg',
                start_number=0)
        .overwrite_output()
        .run(quiet=False)
    )
    (
        ffmpeg
        .input(mixaudio_path)
        .filter('atrim', duration=45)
        .output(audio_path+dir+'.wav', format='wav', ac=1, ar=11025)
        .overwrite_output()
        .run(quiet=False)
    )
    for i in range(len(sep_audio_files)):
        (
            ffmpeg
            .input(dataset_path+dir+'/'+sep_audio_files[i])
            .filter('atrim', duration=45)
            .output(audio_path+dir+'_{}.wav'.format(i), format='wav', ac=1, ar=11025)
            .overwrite_output()
            .run(quiet=False)
        )
