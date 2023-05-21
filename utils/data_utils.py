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
        print(len(self.labels))
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


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('../../data/office_caltech_10/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('../../data/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
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

class FairFaceDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.gender, self.labels = np.load('../../data/FairFace/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.gender, self.labels = np.load('../../data/FairFace/{}_test.pkl'.format(site), allow_pickle=True)
        
        self.path = np.asarray(self.paths)
        print(len(self.path))
        self.labels = np.asarray(self.labels).astype(np.longlong)
        # self.labels = np.asarray(self.labels).astype(np.float16)
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

class FairFaceAmpDataset(Dataset):
    def __init__(self, base_path, site, amp_loader=None, train=True, transform=None):
        if train:
            self.paths, self.gender, self.labels = np.load('../../data/FairFace/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.gender, self.labels = np.load('../../data/FairFace/{}_test.pkl'.format(site), allow_pickle=True)
        
        self.path = np.asarray(self.paths)
        self.labels = np.asarray(self.labels).astype(np.longlong)
        self.transform = transform
        self.base_path = '../../data/FairFace'
        self.amp_loader = amp_loader
        # print(os.path.join(self.base_path, self.paths[0]))
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        img = Image.open(img_path)
        if self.amp_loader is not None:
            img_np = np.asarray(img, dtype=np.float32)
            tar_freq = self.amp_loader.get_amp()
            img = source_to_target_freq(img_np, tar_freq, L=0, ratio=1.0)
            # image_tar_freq = source_to_target_freq(img_np, tar_freq[:3,...], L=0,ratio=1.0)
            img = np.clip(img, 0, 255)

            img = Image.fromarray(img.astype(np.uint8))

        if len(img.split()) != 3:
            img = transforms.Grayscale(num_output_channels=3)(img)

        if self.transform is not None:
            img = self.transform(img)


        return img, label

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ,ratio=1.0):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    # print (b)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    ratio = np.random.randint(1,10)/10

    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    # a_src[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    # a_trg[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2]
    # a_trg = np.fft.ifftshift( a_trg, axes=(-2, -1) )
    return a_src

def source_to_target_freq( src_img, amp_trg, L=0.1 ,ratio=1.0):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img.transpose((2, 0, 1))
    amp_trg = amp_trg.transpose((2, 0, 1))
    amp_trg = amp_trg[:3,...]
    src_img_np = src_img #.cpu().numpy()
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L ,ratio=1.0)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg.transpose(1, 2, 0)

class FairFace_AMP():
    def __init__(self, sites):
        self.sites = sites
        self.amp_paths = []
        self.base_path = '../../data/'
        dataset = 'FairFace'
        for dname in sites:
            # name_train, labels_train, = _dataset_info(os.path.join(os.path.dirname(__file__), 'txt_lists', args.dataset, '%s_train.txt' % dname))
            name_train, _, _ = np.load(f'../../data/{dataset}/{dname}_train.pkl', allow_pickle=True)
            self.amp_paths.append(os.path.join(self.base_path, dataset, 'amp/train', f"{dname}_mean.npy"))
        self.all_amp = os.path.join(self.base_path, dataset, 'amp/train', f"all_mean.npy")
    def get_amp(self):
        site_idx = np.random.choice(len(self.sites))
        tar_freq_path = self.amp_paths[site_idx]
        tar_freq = np.load(os.path.join(tar_freq_path))
        return tar_freq