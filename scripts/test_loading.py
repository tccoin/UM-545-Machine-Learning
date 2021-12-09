import json
import os
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
                audio_path = os.path.join(audio_dir, instrument, video_id+'.wav')
                if os.path.exists(audio_path):
                    audio_raw, rate = librosa.load(audio_path, sr=None, mono=True)
                    assert(audio_raw.shape[0]>0)
                    print('PASS '+audio_path)


if __name__ == '__main__':
    spliter = VideoSpliter()
    spliter.fromjson('data/video', 'data/frames', 'data/audio', os.path.split(sys.argv[0])[0]+'/solos.json')
