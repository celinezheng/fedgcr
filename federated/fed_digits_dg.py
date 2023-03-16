"""
federated learning with different aggregation strategy on office dataset
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import data_utils

import pickle as pkl
from utils.data_utils import DomainNetDataset
import argparse
import time
import copy
import torchvision.transforms as transforms
import random
import numpy as np
from utils.doprompt import DoPrompt
from domainbed import hparams_registry, misc
import json
from utils.util import train, train_doprompt, test, communication

def write_log(msg):
    log_path = '../logs/digits_dg/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(os.path.join(log_path, f'{args.mode}_lsim={args.lsim}_target_idx={args.target_idx}.log'), 'a') as logfile:
        logfile.write(msg)

def train_prox(args, model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
                
            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff
                        
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total
     
def prepare_data(args):
    img_size = 224
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Resize([img_size,img_size]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([img_size,img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([img_size,img_size]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([img_size,img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.Resize([img_size,img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    base_path = "../../data/digit"
    # MNIST
    mnist_trainset     = data_utils.DigitsDataset(data_path=os.path.join(base_path, "MNIST"), channels=1, percent=args.percent, train=True,  transform=transform_mnist)
    mnist_testset      = data_utils.DigitsDataset(data_path=os.path.join(base_path, "MNIST"), channels=1, percent=args.percent, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset      = data_utils.DigitsDataset(data_path=os.path.join(base_path, "SVHN"), channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
    svhn_testset       = data_utils.DigitsDataset(data_path=os.path.join(base_path, "SVHN"), channels=3, percent=args.percent,  train=False, transform=transform_svhn)

    # USPS
    usps_trainset      = data_utils.DigitsDataset(data_path=os.path.join(base_path, "USPS"), channels=1, percent=args.percent,  train=True,  transform=transform_usps)
    usps_testset       = data_utils.DigitsDataset(data_path=os.path.join(base_path, "USPS"), channels=1, percent=args.percent,  train=False, transform=transform_usps)

    # Synth Digits
    synth_trainset     = data_utils.DigitsDataset(data_path=os.path.join(base_path, "SynthDigits"), channels=3, percent=args.percent,  train=True,  transform=transform_synth)
    synth_testset      = data_utils.DigitsDataset(data_path=os.path.join(base_path, "SynthDigits"), channels=3, percent=args.percent,  train=False, transform=transform_synth)

    # MNIST-M
    mnistm_trainset     = data_utils.DigitsDataset(data_path=os.path.join(base_path, "MNIST_M"), channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
    mnistm_testset      = data_utils.DigitsDataset(data_path=os.path.join(base_path, "MNIST_M"), channels=3, percent=args.percent,  train=False, transform=transform_mnistm)

    min_data_len = min(len(mnist_trainset), len(svhn_trainset), len(usps_trainset), len(synth_trainset), len(mnistm_trainset))
    val_len = int(min_data_len * 0.2)
    min_data_len = int(min_data_len * 0.2)

    mnist_valset = torch.utils.data.Subset(mnist_trainset, list(range(len(mnist_trainset)))[-val_len:]) 
    mnist_trainset = torch.utils.data.Subset(mnist_trainset, list(range(min_data_len)))
    
    svhn_valset = torch.utils.data.Subset(svhn_trainset, list(range(len(svhn_trainset)))[-val_len:]) 
    svhn_trainset = torch.utils.data.Subset(svhn_trainset, list(range(min_data_len)))
    
    usps_valset = torch.utils.data.Subset(usps_trainset, list(range(len(usps_trainset)))[-val_len:]) 
    usps_trainset = torch.utils.data.Subset(usps_trainset, list(range(min_data_len)))
    
    synth_valset = torch.utils.data.Subset(synth_trainset, list(range(len(synth_trainset)))[-val_len:]) 
    synth_trainset = torch.utils.data.Subset(synth_trainset, list(range(min_data_len)))
    
    mnistm_valset = torch.utils.data.Subset(mnistm_trainset, list(range(len(mnist_trainset)))[-val_len:]) 
    mnistm_trainset = torch.utils.data.Subset(mnistm_trainset, list(range(min_data_len)))
    
    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    mnist_val_loader  = torch.utils.data.DataLoader(mnist_valset, batch_size=args.batch, shuffle=False)
    mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    
    svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
    svhn_val_loader = torch.utils.data.DataLoader(svhn_valset, batch_size=args.batch, shuffle=False)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    
    usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
    usps_val_loader = torch.utils.data.DataLoader(usps_valset, batch_size=args.batch, shuffle=False)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
    
    synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
    synth_val_loader = torch.utils.data.DataLoader(synth_valset, batch_size=args.batch, shuffle=False)
    synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    
    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
    mnistm_val_loader = torch.utils.data.DataLoader(mnistm_valset, batch_size=args.batch, shuffle=False)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

    

    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    val_loaders  = [mnist_val_loader, svhn_val_loader, usps_val_loader, synth_val_loader, mnistm_val_loader]
    test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]

    return train_loaders, val_loaders, test_loaders

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed=  4
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 16, help ='batch size')
    parser.add_argument('--iters', type = int, default=300, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='DoPrompt', help='fedavg | DoPrompt')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint/digits', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--deep', action='store_true', help ='deep prompt')
    parser.add_argument('--lsim', action='store_true', help ='lsim loss for adapter')
    parser.add_argument('--algorithm', type=str, default='DoPrompt')
    parser.add_argument('--dataset', type=str, default='digits')
    parser.add_argument('--num_classes', type = int, default=10, help ='number of classes')
    parser.add_argument('--model', type = str, default='prompt', help='prompt | vit-linear')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--target_idx', type=int, default=0,
        help='client idx for unseen target domain')
    args = parser.parse_args()

    exp_folder = 'fed_digits'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, f'{args.mode}_lsim={args.lsim}_target_idx={args.target_idx}')

    write_log('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    write_log('===Setting===\n')
    write_log('    lr: {}\n'.format(args.lr))
    write_log('    batch: {}\n'.format(args.batch))
    write_log('    iters: {}\n'.format(args.iters))
    write_log('    wk_iters: {}\n'.format(args.wk_iters))
    
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.mode, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.mode, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    train_loaders, val_loaders, test_loaders = prepare_data(args)
    
    # name of each datasets
    datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
    # federated client number
    datasets_num = len(datasets)
    client_num = len(datasets)-1
    client_weights = [1/client_num for i in range(client_num)]
    # setup model
    server_model = DoPrompt(num_classes=10, num_domains=client_num, hparams=hparams).to(device)
    loss_fun = nn.CrossEntropyLoss()

    # each local client model
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    best_changed = False

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        _, test_acc = test(server_model, test_loaders[args.target_idx], loss_fun, device)
        print(f' {datasets[args.target_idx]:<11s} | Test Acc On Target Domain: {test_acc:.4f}')
        write_log(f' {datasets[args.target_idx]:<11s} | Test Acc On Target Domain: {test_acc:.4f}\n')
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        for client_idx in range(client_num):
            models[client_idx].load_state_dict(checkpoint['server_model'])
        best_epoch, best_acc  = checkpoint['best_epoch'], checkpoint['best_acc']
        start_iter = int(checkpoint['a_iter']) + 1

        print('Resume training from epoch {}'.format(start_iter))
    else:
        best_epoch = 0
        best_acc = [0. for j in range(client_num)] 
        start_iter = 0

    # Start training
    for a_iter in range(start_iter, args.iters):
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            write_log("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 

            for dataset_idx in range(datasets_num):
                if dataset_idx == args.target_idx: continue
                if dataset_idx < args.target_idx: client_idx = dataset_idx
                else: client_idx = dataset_idx-1
                model = models[client_idx]
                if args.mode.lower() == 'fedprox':
                    # skip the first server model(random initialized)
                    if a_iter > 0:
                        train_loss, train_acc = train_prox(args, model, train_loaders[dataset_idx], optimizers[client_idx], loss_fun, device)
                    else:
                        train_loss, train_acc = train(model, train_loaders[dataset_idx], optimizers[client_idx], loss_fun, device)    
                else:
                    train_doprompt(args, model, train_loaders[dataset_idx], client_idx, device)
        
        with torch.no_grad():
            # aggregation
            server_model, models = communication(args, server_model, models, client_weights, client_num)
            # Report loss after aggregation
            for dataset_idx in range(datasets_num):
                if dataset_idx == args.target_idx: continue
                if dataset_idx < args.target_idx: client_idx = dataset_idx
                else: client_idx = dataset_idx-1
                model = models[client_idx]
                train_loss, train_acc = test(model, train_loaders[dataset_idx], loss_fun, device)
                print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[dataset_idx] ,train_loss, train_acc))
                write_log(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[dataset_idx] ,train_loss, train_acc))
            # Validation
            val_acc_list = [None for j in range(client_num)]
            for dataset_idx in range(datasets_num):
                if dataset_idx == args.target_idx: continue
                if dataset_idx < args.target_idx: client_idx = dataset_idx
                else: client_idx = dataset_idx-1
                model = models[client_idx]
                val_loss, val_acc = test(model, val_loaders[dataset_idx], loss_fun, device)
                val_acc_list[client_idx] = val_acc
                print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[dataset_idx], val_loss, val_acc), flush=True)
                write_log(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[dataset_idx], val_loss, val_acc))
            # Record best
            if np.mean(val_acc_list) > np.mean(best_acc):
                # for client_idx in range(client_num):
                for dataset_idx in range(datasets_num):
                    if dataset_idx == args.target_idx: continue
                    if dataset_idx < args.target_idx: client_idx = dataset_idx
                    else: client_idx = dataset_idx-1
                    best_acc[client_idx] = val_acc_list[client_idx]
                    best_epoch = a_iter
                    best_changed=True
                    print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(datasets[dataset_idx], best_epoch, best_acc[client_idx]))
                    write_log(' Best site-{:<10s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[dataset_idx], best_epoch, best_acc[client_idx]))
            if best_changed:     
                print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
                write_log(' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))
                torch.save({
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                best_changed = False
            _, test_acc = test(server_model, test_loaders[args.target_idx], loss_fun, device)
            print(f' {datasets[args.target_idx]:<11s} | Test Acc On Target Domain: {test_acc:.4f}')
            write_log(f' {datasets[args.target_idx]:<11s} | Test Acc On Target Domain: {test_acc:.4f}\n')
        