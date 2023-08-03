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
    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.longlong).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
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
            
        self.labels = np.asarray(self.text_labels).astype(np.longlong).squeeze()
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

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
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('../../data/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('../../data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
        
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../../data'

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

class GeoNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load(f'{base_path}/{site}_train.pkl', allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load(f'{base_path}/{site}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict = {'alley':0, 'ballroom':1, 'bridge':2, 'cafeteria':3, 'castle':4, 'coast':5, 'highway':6, 'living_room':7, 'market_outdoor':8, 'shopfront':9}
        # label_dict = {'assembly_hall':0, 'beblack_rhinocerosacon':1, 'food_court':2, 'cab':3, 'banquet':4, 'sidewalk':5, 'bakery':6, 'billboard':7, 'buffet':8, 'bungalow':9}
        
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../../data'

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

class FairFaceIIDDataset(Dataset):
    def __init__(self, args, base_path, site, gender_label=False, train=True, transform=None):
        if train:
            self.paths, self.gender, self.age = np.load(f'{base_path}/pkl/{args.gender_dis}/{site}_train.pkl', allow_pickle=True)
        else:
            self.paths, self.gender, self.age = np.load(f'{base_path}/pkl/{args.gender_dis}/{site}_test.pkl', allow_pickle=True)
        
        self.path = np.asarray(self.paths)
        gender_dict = {'Male':0, 'Female':1}     
        self.gender = [gender_dict[text] for text in self.gender]
        if gender_label:
            self.labels = np.asarray(self.gender).astype(np.float16)
        else: 
            self.labels = np.asarray(self.age).astype(np.float16)
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../../data'
       
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

class FairFaceBinaryDataset(Dataset):
    def __init__(self, base_path, site, client_idx, gender_label=False, train=True, transform=None):
        if gender_label: distribution = 'binary_race_gender'
        else: distribution  = 'binary_race_mix'
        if train:
            self.paths, self.gender, self.age = np.load(f'{base_path}/pkl/{distribution}/train_{site}_{client_idx}.pkl', allow_pickle=True)
        else:
            self.paths, self.gender, self.age = np.load(f'{base_path}/pkl/{distribution}/test_{site}_{client_idx}.pkl', allow_pickle=True)
        
        self.path = np.asarray(self.paths)
        gender_dict = {'Male':0, 'Female':1}     
        self.gender = [gender_dict[text] for text in self.gender]
        if gender_label:
            self.labels = np.asarray(self.gender).astype(np.float16)
        else: 
            self.labels = np.asarray(self.age).astype(np.float16)
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../../data'
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


class FairFaceGenderDataset(Dataset):
    def __init__(self, distribution_mode, base_path, site, client_idx, gender_label=False, train=True, transform=None):
        if train:
            self.paths, self.gender, self.age = np.load(f'../../data/FairFace/pkl/{distribution_mode}/{site}_train_{client_idx}.pkl', allow_pickle=True)
        else:
            self.paths, self.gender, self.age = np.load(f'../../data/FairFace/pkl/{distribution_mode}/{site}_test_{client_idx}.pkl', allow_pickle=True)
        
        self.path = np.asarray(self.paths)
        gender_dict = {'Male':0, 'Female':1}     
        self.gender = [gender_dict[text] for text in self.gender]
        if gender_label:
            self.labels = np.asarray(self.gender).astype(np.float16)
        else: 
            self.labels = np.asarray(self.age).astype(np.float16)
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../../data'
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
