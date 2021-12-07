import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms


class FrameDataset:

    def __init__(self, batch_size=4, dataset_path='data_sample/frames', if_resize=True):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.if_resize = if_resize
        self.train_dataset = self.get_train_numpy()
        self.x_mean, self.x_std = self.compute_train_statistics()
        self.transform = self.get_transforms()
        # self.train_loader, self.val_loader = self.get_dataloaders()
        self.train_loader = self.get_dataloaders()

    def get_train_numpy(self):
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'train'))
        train_x = np.zeros((len(train_dataset), 224, 398, 3))
        # train_x = np.zeros((len(train_dataset), 64, 64, 3))
        for i, (img, _) in enumerate(train_dataset):
            train_x[i] = img
        return train_x / 255.0

    def compute_train_statistics(self):
        # compute per-channel mean and std with respect to self.train_dataset
        x_mean = np.mean(self.train_dataset, axis=(0, 1, 2))  # per-channel mean
        x_std = np.std(self.train_dataset, axis=(0, 1, 2))  # per-channel std
        return x_mean, x_std

    def get_transforms(self):
        if self.if_resize:
            transform_list = [
                transforms.Resize((28, 50)),
                transforms.ToTensor(),
                transforms.Normalize(self.x_mean, self.x_std)
            ]
        else:
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(self.x_mean, self.x_std)
            ]
        transform = transforms.Compose(transform_list)
        return transform

    def get_dataloaders(self):
        # train set
        train_set = torchvision.datasets.ImageFolder(os.path.join(
            self.dataset_path, 'train'), transform=self.transform)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)

        # validation set
        # val_set = torchvision.datasets.ImageFolder(os.path.join(
        #     self.dataset_path, 'val'), transform=self.transform)
        # val_loader = torch.utils.data.DataLoader(
        #     val_set, batch_size=self.batch_size, shuffle=False)

        # return train_loader, val_loader
        return train_loader
