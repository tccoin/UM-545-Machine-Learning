from PIL import Image, UnidentifiedImageError
import json
import os
import shutil
import sys
import librosa
import warnings
warnings.filterwarnings('ignore')


class VideoSpliter():
    """split video into frames and audio files """

    def __init__(self):
        self.fps = '8'

    def fromjson(self, dataset_dir, frames_dir, audio_dir, json_path):
        dataset = json.load(open(json_path))

        for instrument in dataset.keys():
            for i, video_id in enumerate(dataset[instrument]):
                audio_path = os.path.join(
                    audio_dir, instrument, video_id+'.wav')
                if os.path.exists(audio_path):
                    audio_raw, rate = librosa.load(
                        audio_path, sr=None, mono=True)
                    assert(audio_raw.shape[0] > 0)
                    print('PASS '+audio_path)
        for instrument in dataset.keys():
            for i, video_id in enumerate(dataset[instrument]):
                j = 0
                while True:
                    frames_path = os.path.join(
                        frames_dir, instrument, video_id+'.mp4')
                    frame_path = os.path.join(frames_path, '{}.jpg'.format(j))
                    if not os.path.exists(frame_path):
                        break
                    try:
                        Image.open(frame_path).convert('RGB')
                    except UnidentifiedImageError:
                        print('ERROR: ', frame_path)
                        os.remove(frame_path)
                        shutil.copyfile(os.path.join(
                            frames_path, '{}.jpg'.format(j-1)), frame_path)
                        print('FIX: ', frame_path)
                    j += 1
                print('PASS ' + frames_path)


if __name__ == '__main__':
    spliter = VideoSpliter()
    spliter.fromjson('data/video', 'data/frames', 'data/audio',
                     os.path.split(sys.argv[0])[0]+'/solos.json')
