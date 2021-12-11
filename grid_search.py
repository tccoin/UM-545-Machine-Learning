import torch
from main import *

args = {
    # general
    'mode': 'train',
    'seed': None,
    'batch_size': 80,
    'workers': 24,
    'print_interval_batch': 1,
    'evaluate_interval_epoch': 10,
    'ckeckpoint_path': 'ckpt/',
    # training
    'num_epoch': 100,
    'net_sound_lr': 1e-3,
    'net_frames_lr': 1e-3,
    'net_synthesizer_lr': 1e-3,
    'weight_decay': 1e-3,
    'resnet_pretrained': True,
    # dataset
    'mix_num': 2,
    'train_samples_num': 400,
    'validation_samples_num': 40,
    'test_samples_num': 44,
    'train_sample_list_path': 'data/train.csv',
    'validation_sample_list_path': 'data/val.csv',
    'test_sample_list_path': 'data/test.csv',
    # frames
    'frame_size': 224,
    'frames_per_video': 3,
    'frames_stride': 24,
    'frame_rate': 8,
    # audio
    'audio_rate': 11025,
    'audio_length': 65535,
    'use_binary_mask': True,
    # STFT
    'log_freq': True,
    'stft_frame': 1022,
    'stft_hop': 256,
}
for i in range(6):
    # grid search
    args['resnet_pretrained'] = True
    args['net_sound_lr'] = 3e-4
    args['net_frames_lr'] = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2][i % 6]
    args['net_synthesizer_lr'] = 3e-4
    print('=== Search {}: pretrained={}, net_frames_lr={} ==='.format(
        i,
        args['resnet_pretrained'],
        args['net_frames_lr']
    ))

    # states
    args['device'] = torch.device('cuda')
    args['writer'] = SummaryWriter()
    args['current_epoch'] = 0
    # args['seed'] = random.randint(0, 10000)
    args['seed'] = 6480
    args['history'] = {'train_loss': [],
                       'validation_loss': [], 'test_loss': [], 'metrics': []}

    # nets
    nets = build_nets(args)
    model = NetWrapper(nets, args).to(args['device'])
    optimizer = build_optimizer(nets, args)

    # load checkpoint
    # load_checkpoint('ckpt/latest.pth', model, optimizer, args)

    # dataset and loader
    dataset_train = SolosMixDataset(args, 'train')
    dataset_validation = SolosMixDataset(args, 'validation')
    dataset_test = UrmpDataset(args, 'test')
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=args['workers'],
        drop_last=True)
    loader_validation = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=args['validation_samples_num'],
        shuffle=False,
        num_workers=args['workers'],
        drop_last=False)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args['test_samples_num'],
        shuffle=False,
        num_workers=args['workers'],
        drop_last=False)

    evaluate(model, loader_validation, args)
    test(model, loader_test, args)
    if args['mode'] == 'evaluate':
        exit()
    elif args['mode'] == 'train':
        # train mode
        epoch_iters = len(dataset_train)
        print('1 Epoch = {} iters'.format(epoch_iters))
        for epoch in range(args['current_epoch'], args['num_epoch']):
            args['current_epoch'] = epoch
            print('Epoch {}'.format(epoch+1))
            train(model, loader_train, optimizer, args)
            evaluate(model, loader_validation, args)
            if (epoch+1) % args['evaluate_interval_epoch'] == 0:
                test(model, loader_test, args)
                save_checkpoint(model, optimizer, args)
