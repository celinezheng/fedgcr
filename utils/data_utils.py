import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

class DigitsDataset(Dataset):
    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None, split_test=False):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.paths, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            paths, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.paths = np.concatenate([self.paths,paths], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.paths, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.paths.shape[0] * percent*10)
                    self.paths = self.paths[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.paths, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.paths, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.longlong).squeeze()
        if split_test:
            self.paths, self.labels = self.shuffle_order()
    
    def shuffle_order(self):
        randomize = np.arange(len(self.labels))
        np.random.shuffle(randomize)
        return self.paths[randomize], self.labels[randomize]

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        image = self.paths[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class PACSDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('../../data/PACS/pkls/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('../../data/PACS/pkls/{}_test.pkl'.format(site), allow_pickle=True)
            
        self.paths = np.asarray(self.paths)
        self.labels = np.asarray(self.text_labels).astype(np.longlong).squeeze()
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'
        self.paths, self.labels = self.shuffle_order()
    
    def shuffle_order(self):
        randomize = np.arange(len(self.labels))
        np.random.shuffle(randomize)
        return self.paths[randomize], self.labels[randomize]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None, split_test=False):
        if train:
            self.paths, self.text_labels = np.load('../../data/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('../../data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
        
        self.labels = [label_dict[text] for text in self.text_labels]
        self.labels = np.asarray(self.labels).astype(np.longlong).squeeze()
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../../data'
        if split_test:
            self.paths, self.labels = self.shuffle_order()
    
    def shuffle_order(self):
        randomize = np.arange(len(self.labels))
        np.random.shuffle(randomize)
        return self.paths[randomize], self.labels[randomize]
 
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
