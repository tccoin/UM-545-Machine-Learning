import json
import os
import sys
import yt_dlp as youtube_dl

class YouTubeSaver(object):
    """Load video from YouTube using an auditionDataset.json """

    def __init__(self):
        self.outtmpl = '%(id)s.%(ext)s'
        self.ydl_opts = {
            'format': 'mp4',
            'outtmpl': self.outtmpl,
            """
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            """
            'logger': None
        }

    def fromjson(self, dataset_dir, json_path):
        dataset = json.load(open(json_path))

        for instrument in dataset.keys():
            if not os.path.exists(os.path.join(dataset_dir, instrument)):
                os.makedirs(os.path.join(dataset_dir, instrument))
            self.ydl_opts['outtmpl'] = os.path.join(dataset_dir, instrument, self.outtmpl)
            with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
                for i, video_id in enumerate(dataset[instrument]):
                    try:
                        ydl.download(['https://www.youtube.com/watch?v=%s' % video_id])
                        # del dataset[video_id]
                    except OSError:
                        with open(os.path.join(dataset_dir, 'backup.json'), 'w') as dst_file:
                            json.dump(dataset, dst_file)
                        print('Process failed at video {0}, #{1}'.format(video_id, i))
                        print('Backup saved at {0}'.format(os.path.join(dataset_dir, 'backup.json')))
                        ydl.download(['https://www.youtube.com/watch?v=%s' % video_id])
                    except youtube_dl.utils.DownloadError:
                        print(video_id, 'not available')

                    except KeyboardInterrupt:
                        sys.exit()


if __name__ == '__main__':
    saver = YouTubeSaver()
    saver.fromjson('data', os.path.split(sys.argv[0])[0]+'/solos.json')
