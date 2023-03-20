"""
federated learning with different aggregation strategy on domainnet dataset
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pickle as pkl
from utils.prompt_vit import PromptViT
import argparse
import time
import copy
import random
import numpy as np
from utils.doprompt import DoPrompt
from domainbed import hparams_registry, misc
import json
from utils.util import train, train_doprompt, test, communication, train_fedprox, prepare_data

def write_log(msg):
    log_path = f'../logs/{args.dataset}_{args.expname}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(os.path.join(log_path, f'{args.mode}_lsim={args.lsim}.log'), 'a') as logfile:
        logfile.write(msg)
      

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= 1
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch', type = int, default=32, help ='batch size')
    parser.add_argument('--iters', type = int, default=300, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='DoPrompt', help='[FedBN | FedAvg | DoPrompt]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--deep', action='store_true', help ='deep prompt')
    parser.add_argument('--lsim', action='store_true', help ='lsim loss for adapter')
    parser.add_argument('--dataset', type=str, default='digit')
    parser.add_argument('--percent', type = float, default= 0.2, help ='percentage of dataset to train')
    parser.add_argument('--num_classes', type = int, default=10, help ='number of classes')
    parser.add_argument('--model', type = str, default='prompt', help='prompt | vit-linear')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--expname', type=str, default='prompt-sim')
    args = parser.parse_args()

    exp_folder = f'fed_{args.dataset}_{args.expname}'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, f'{args.mode}_lsim={args.lsim}')

    write_log('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    write_log('===Setting===\n')
    write_log('    dataset: {}\n'.format(args.dataset))
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
    
    if args.dataset.lower()[:6] == 'domain':
        # name of each datasets
        datasets = ['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch']
    elif args.dataset.lower()[:5] == 'digit':
        datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
    elif args.dataset.lower()[:6] == 'office':
        datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
    else:
        import warnings
        warnings.warn("invalid args.dataset")
        exit(0)
    train_loaders, val_loaders, test_loaders = prepare_data(args)

    # federated client number
    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)]
    
    # setup model
    if args.mode.lower() == 'fedavg':
        model_type="sup_vitb16_imagenet21k"
        server_model = PromptViT(model_type=model_type, args=args).to(device)
        for name, param in server_model.named_parameters():
            if 'prompt' not in name and 'head' not in name:
                param.requires_grad = False
    else:
        server_model = DoPrompt(num_classes=args.num_classes, num_domains=client_num, hparams=hparams).to(device)

    loss_fun = nn.CrossEntropyLoss()

    # each local client model
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    best_changed = False


    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        test_accs = [0. for _ in range(client_num)]
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        for test_idx, test_loader in enumerate(test_loaders):
            _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            test_accs[test_idx] = test_acc
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
            write_log(' {:<11s}| Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_acc))
        write_log(f'Average Test Accuracy: {np.mean(test_accs):.4f}\n')
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        best_epoch, best_acc  = checkpoint['best_epoch'], checkpoint['best_acc']
        start_iter = int(checkpoint['a_iter']) + 1

        print('Resume training from epoch {}'.format(start_iter))
    else:
        # log the best for each model on all datasets
        best_epoch = 0
        best_acc = [0. for j in range(client_num)] 
        start_iter = 0

    # Start training
    for a_iter in range(start_iter, args.iters):
        if args.mode.lower() != 'doprompt':
            optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        else:
            optimizers = [None for _ in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            write_log("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 

            for client_idx, model in enumerate(models):
                if args.mode.lower() == 'fedprox':
                    # skip the first server model(random initialized)
                    if a_iter > 0:
                        train_loss, train_acc = train_fedprox(args, server_model, model ,train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
                    else:
                        train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)    
                elif args.mode.lower() == 'doprompt':
                    train_doprompt(args, model, train_loaders[client_idx], client_idx, device)
                else:
                    train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
        
        with torch.no_grad():
            # Aggregation
            server_model, models = communication(args, server_model, models, client_weights, client_num)

            # Report loss after aggregation
            for client_idx, model in enumerate(models):
                train_loss, train_acc = test(model, train_loaders[client_idx], loss_fun, device)
                print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
                write_log(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))

            # Validation
            val_acc_list = [None for j in range(client_num)]
            for client_idx, model in enumerate(models):
                val_loss, val_acc = test(model, val_loaders[client_idx], loss_fun, device)
                val_acc_list[client_idx] = val_acc
                print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss, val_acc))
                write_log(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], val_loss, val_acc))
            write_log(f'Average Valid Accuracy: {np.mean(val_acc_list):.4f}\n')
            print(f'Average Valid Accuracy: {np.mean(val_acc_list):.4f}')
            
            # Record best
            if np.mean(val_acc_list) > np.mean(best_acc):
                for client_idx in range(client_num):
                    best_acc[client_idx] = val_acc_list[client_idx]
                    best_epoch = a_iter
                    best_changed=True
                    print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(datasets[client_idx], best_epoch, best_acc[client_idx]))
                    write_log(' Best site-{:<10s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[client_idx], best_epoch, best_acc[client_idx]))

            test_accs = [0. for _ in range(client_num)]
            if best_changed:     
                print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
                write_log(' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))
                if args.mode.lower() == 'fedbn':
                    torch.save({
                        'model_0': models[0].state_dict(),
                        'model_1': models[1].state_dict(),
                        'model_2': models[2].state_dict(),
                        'model_3': models[3].state_dict(),
                        'model_4': models[4].state_dict(),
                        'model_5': models[5].state_dict(),
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                    best_changed = False
                    for client_idx, datasite in enumerate(datasets):
                        _, test_acc = test(models[client_idx], test_loaders[client_idx], loss_fun, device)
                        test_accs[client_idx] = test_acc
                        print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                        write_log(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                else:
                    torch.save({
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                    best_changed = False
                    for client_idx, datasite in enumerate(datasets):
                        _, test_acc = test(server_model, test_loaders[client_idx], loss_fun, device)
                        test_accs[client_idx] = test_acc
                        print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                        write_log(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

                print(f'Average Test Accuracy: {np.mean(test_accs):.4f}')
                write_log(f'Average Test Accuracy: {np.mean(test_accs):.4f}\n')
    write_log('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

