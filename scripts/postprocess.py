import json
import os
import sys
import ffmpeg


class VideoSpliter(object):
    """split video into frames and audio files """

    def __init__(self):
        self.fps = '8'

    def fromjson(self, dataset_dir, frames_dir, audio_dir, json_path):
        dataset = json.load(open(json_path))

        for instrument in dataset.keys():
            instrument_frames_dir = os.path.join(frames_dir, instrument)
            instrument_audio_dir = os.path.join(audio_dir, instrument)
            if not os.path.exists(instrument_frames_dir):
                os.makedirs(instrument_frames_dir)
            if not os.path.exists(instrument_audio_dir):
                os.makedirs(instrument_audio_dir)
            for i, video_id in enumerate(dataset[instrument]):
                video_path = os.path.join(
                    dataset_dir, instrument, video_id)+'.mp4'
                audio_path = os.path.join(
                    audio_dir, instrument, video_id+'.wav')
                frames_dir_path = os.path.join(
                    frames_dir, instrument, video_id+'.mp4')
                frame_existed = os.path.exists(frames_dir_path+'/0.jpg')
                audio_existed = os.path.exists(audio_path)
                video_existed = os.path.exists(video_path)
                if not frame_existed and video_existed:
                    os.makedirs(frames_dir_path)
                    try:
                        (
                            ffmpeg
                            .input(video_path)
                            .filter('fps', fps=self.fps)
                            .filter('scale', -1, 224)
                            .trim(start=0, end=45)
                            .setpts('PTS-STARTPTS')
                            .output(frames_dir_path+'/%d.jpg',
                                    start_number=0)
                            .overwrite_output()
                            .run(quiet=False)
                        )
                    except KeyboardInterrupt:
                        sys.exit()
                if not audio_existed and video_existed:
                    try:
                        (
                            ffmpeg
                            .input(video_path)
                            .filter('atrim', duration=45)
                            .output(audio_path, format='wav', ac=1, ar=11025)
                            .overwrite_output()
                            .run(quiet=False)
                        )
                    except KeyboardInterrupt:
                        sys.exit()


if __name__ == '__main__':
    spliter = VideoSpliter()
    spliter.fromjson('data1/video', 'data1/frames', 'data1/audio',
                     os.path.split(sys.argv[0])[0]+'/solos.json')
