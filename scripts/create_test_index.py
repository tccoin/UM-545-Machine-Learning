import os
import glob
import argparse
import random
import fnmatch


def find_recursive(root_dir, ext='.wav'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_audio', default='./data/urmp/audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='./data/urmp/frames',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=8, type=int,
                        help="fps of video frames")
    parser.add_argument('--path_output', default='./data',
                        help="path to output index files")
    parser.add_argument('--trainset_ratio', default=1.0, type=float,
                        help="80% for training, 20% for validation")
    args = parser.parse_args()

    # find all audio/frames pairs
    infos = []
    audio_files = find_recursive(args.root_audio, ext='.wav')
    for audio_path in audio_files:
        frame_path = audio_path.replace(args.root_audio, args.root_frame) \
                               .replace('.wav', '.mp4')
        frame_files = glob.glob(frame_path + '/*.jpg')
        if len(frame_files) > args.fps * 20:
            infos.append(
                ','.join([audio_path, frame_path, str(len(frame_files))]))
    print('{} audio/frames pairs found.'.format(len(infos)))

    # split train/val
    random.shuffle(infos)
    trainset = infos
    for name, subset in zip(['test'], [trainset]):
        filename = '{}.csv'.format(os.path.join(args.path_output, name))
        with open(filename, 'w') as f:
            for item in subset:
                f.write(item + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))

    print('Done!')
