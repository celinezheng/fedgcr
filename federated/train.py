"""
federated learning with different aggregation strategy on domainnet dataset
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import math
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
from utils.doprompt import DoPrompt, FedPrompt, CoCoOP
from domainbed import hparams_registry, misc
import json
from utils.util import train, train_doprompt, test, communication, train_fedprox, prepare_data, train_fedprompt, write_log, train_CoCoOP
from utils import util
      

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= 1
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--tune', action='store_true', help='whether to tune hparams')
    parser.add_argument('--memory', action='store_true', help='whether to test memory usage of each algorithm')
    parser.add_argument('--dg', action='store_true', help='domain generalization')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--lambda_con', type=float, default=0.5, help='lambda for contrastive loss')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default=16, help ='batch size')
    parser.add_argument('--iters', type = int, default=10, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=5, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='DoPrompt', help='[FedBN | FedAvg | DoPrompt]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--deep', action='store_true', help ='deep prompt')
    parser.add_argument('--dataset', type=str, default='digit')
    parser.add_argument('--percent', type = float, default= 0.2, help ='percentage of dataset to train')
    parser.add_argument('--num_classes', type = int, default=10, help ='number of classes')
    parser.add_argument('--model', type = str, default='prompt', help='prompt | vit-linear')
    parser.add_argument('--target_domain', type = str, default='Clipart', help='Clipart, Infograph, ...')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--expname', type=str, default='prompt-sim')
    args = parser.parse_args()

    exp_folder = f'fed_{args.dataset}_{args.expname}'
    if args.tune:
        exp_folder += '_tune'
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, f'{args.mode}')
    if args.dg:
        SAVE_PATH = os.path.join(args.save_path, f'{args.mode}_{args.target_domain}')
    if args.tune:
        SAVE_PATH = os.path.join(args.save_path, f'{args.mode}_tune_{args.lambda_con}')

    write_log(args, '==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    write_log(args, '===Setting===\n')
    write_log(args, '    dataset: {}\n'.format(args.dataset))
    write_log(args, '    lr: {}\n'.format(args.lr))
    write_log(args, '    batch: {}\n'.format(args.batch))
    write_log(args, '    iters: {}\n'.format(args.iters))
    write_log(args, '    wk_iters: {}\n'.format(args.wk_iters))
    write_log(args, '    tune: {}\n'.format(args.tune))
    write_log(args, '    lambda for constrastive: {}\n'.format(args.lambda_con))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.mode, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.mode, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    
    if args.dataset.lower()[:6] == 'domain':
        # name of each datasets
        domain_num = 6
    elif args.dataset.lower()[:5] == 'digit':
        domain_num = 5
    elif args.dataset.lower()[:6] == 'office':
        domain_num = 4
    else:
        import warnings
        warnings.warn("invalid args.dataset")
        exit(0)
    train_loaders, val_loaders, test_loaders, datasets, target_loader = prepare_data(args)
    client_num = len(train_loaders)
    if args.dg:
        domain_num -= 1
    if 'sqrt' in args.expname.lower():
        # federated client number
        domain_num = int(math.sqrt(client_num))
    print(f"domain number = {domain_num}")
    write_log(args, f"domain number = {domain_num}\n")
    client_weights = [1/client_num for i in range(client_num)]
    prompt_bank = None
    # setup model
    if args.mode.lower() == 'doprompt':
        server_model = DoPrompt(num_classes=args.num_classes, num_domains=client_num, hparams=hparams).to(device)
    elif args.mode.lower() == 'fedprompt':
        server_model = FedPrompt(num_classes=args.num_classes, hparams=hparams, lambda_con=args.lambda_con).to(device)
        prompt_bank = nn.Parameter(
            torch.empty(domain_num, 4, 768, requires_grad=False).normal_(std=0.02)
        ).to(device)
        all_pi = None
        for client_idx in range(client_num):
            pi = server_model.state_dict()['prompt_tokens'].unsqueeze(0)
            if client_idx == 0:
                all_pi = pi
            else:
                all_pi = torch.concat((all_pi, pi))
        prompt_bank.detach_()
        print(prompt_bank.shape)
    elif args.mode.lower() == 'cocoop':
        server_model = CoCoOP(num_classes=args.num_classes, hparams=hparams).to(device)
    elif args.mode.lower() == 'full':
        model_type="sup_vitb16_imagenet21k"
        server_model = PromptViT(model_type=model_type, args=args).to(device)
    else:
        model_type="sup_vitb16_imagenet21k"
        server_model = PromptViT(model_type=model_type, args=args).to(device)
        for name, param in server_model.named_parameters():
            if 'prompt' not in name and 'head' not in name:
                param.requires_grad = False
    if args.memory:
        param_size = 0
        trained_param = 0
        for param in server_model.parameters():
            param_size += param.nelement() * param.element_size()
            if param.requires_grad:
                trained_param += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in server_model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            if buffer.requires_grad:
                trained_param += buffer.nelement() * buffer.element_size()
        if prompt_bank is not None:
            param_size += prompt_bank.nelement() * prompt_bank.element_size()
        size_all_train_mb = (trained_param) / 1024**2
        size_all_mb = (param_size + buffer_size) / 1024**2
        print('trained size: {:.3f}MB'.format(size_all_train_mb))
        print('model size: {:.3f}MB'.format(size_all_mb))
        print('trained percentage trained param: {:.6f}'.format(size_all_train_mb/size_all_mb))
        write_log(args, 'trained size: {:.3f}MB'.format(size_all_train_mb))
        write_log(args, 'model size: {:.3f}MB'.format(size_all_mb))
        write_log(args, 'trained percentage trained param: {:.6f}'.format(size_all_train_mb/size_all_mb))
        exit(0)

    loss_fun = nn.CrossEntropyLoss()

    # each local client model
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    best_changed = False

    if args.test:
        write_log(args, 'Loading snapshots...\n')
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        test_accs = [0. for _ in range(client_num)]
        if args.mode.lower() in ['fedbn', 'solo']:
            test_accs = [0. for _ in range(client_num)]
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
            if args.dg:
                for client_idx, datasite in enumerate(datasets):
                    _, test_acc = test(models[client_idx], target_loader, loss_fun, device, prompt_bank)
                    test_accs[client_idx] = test_acc
                    write_log(args, ' Test site-{:<10s} -> {}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, args.target_domain, checkpoint['best_epoch'], test_acc))
                    # print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
            else:
                for client_idx, datasite in enumerate(datasets):
                    _, test_acc = test(models[client_idx], test_loaders[client_idx], loss_fun, device, prompt_bank)
                    test_accs[client_idx] = test_acc
                    # print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                    write_log(args, ' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, checkpoint['best_epoch'], test_acc))
        else:
            test_accs = {}
            best_changed = False
            if args.dg:
                _, test_acc = test(server_model, target_loader, loss_fun, device, prompt_bank)
                write_log(args, f'Test Accuracy of {args.target_domain}: {test_acc:.4f}\n')
                test_accs[args.target_domain] = test_acc
            else:
                for client_idx, datasite in enumerate(datasets):
                    if datasite in test_accs: continue
                    _, test_acc = test(server_model, test_loaders[client_idx], loss_fun, device, prompt_bank)
                    test_accs[datasite] = test_acc
                    # print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                    write_log(args, ' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, checkpoint['best_epoch'], test_acc))
            test_accs = list(test_accs.values())
        write_log(args, f'Average Test Accuracy: {np.mean(test_accs):.4f}\n')
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() in ['fedbn', 'solo']:
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
    gmap = {}
    
    # Start training
    for a_iter in range(start_iter, args.iters):
        if args.mode.lower() not in ['doprompt', 'fedprompt', 'cocoop']:
            optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        else:
            optimizers = [None for _ in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            write_log(args, "============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 

            for client_idx, model in enumerate(models):
                if args.mode.lower() == 'fedprox':
                    # skip the first server model(random initialized)
                    if a_iter > 0:
                        train_loss, train_acc = train_fedprox(args, server_model, model ,train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
                    else:
                        train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)    
                elif args.mode.lower() == 'doprompt':
                    train_doprompt(args, model, train_loaders[client_idx], client_idx, device)
                elif args.mode.lower() == 'fedprompt':
                    if len(gmap) == 0: gidx = -1
                    else: 
                        gidx = gmap[client_idx]
                        if wi == args.wk_iters-1:
                            print(f"gidx of client-{client_idx} is {gidx}")
                    train_fedprompt(gidx, model, train_loaders[client_idx], prompt_bank, device)
                elif args.mode.lower() == 'cocoop':
                    train_CoCoOP(model, train_loaders[client_idx], device)
                else:
                    train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
        
        with torch.no_grad():
            # Aggregation
            if args.mode.lower() != 'solo':
                server_model, models, prompt_bank, gmap = communication(args, len(gmap), server_model, models, client_weights, client_num, domain_num, prompt_bank)

            # Report loss after aggregation
            for client_idx, model in enumerate(models):
                train_loss, train_acc = test(model, train_loaders[client_idx], loss_fun, device, prompt_bank)
                # print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
                write_log(args, ' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))

            # Validation
            val_acc_list = [None for j in range(client_num)]
            for client_idx, model in enumerate(models):
                val_loss, val_acc = test(model, val_loaders[client_idx], loss_fun, device, prompt_bank)
                val_acc_list[client_idx] = val_acc
                # print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss, val_acc))
                write_log(args, ' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], val_loss, val_acc))
            write_log(args, f'Average Valid Accuracy: {np.mean(val_acc_list):.4f}\n')
            # print(f'Average Valid Accuracy: {np.mean(val_acc_list):.4f}')
            
            # Record best
            if np.mean(val_acc_list) > np.mean(best_acc):
                for client_idx in range(client_num):
                    best_acc[client_idx] = val_acc_list[client_idx]
                    best_epoch = a_iter
                    best_changed=True
                    # print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(datasets[client_idx], best_epoch, best_acc[client_idx]))
                    write_log(args, ' Best site-{:<10s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[client_idx], best_epoch, best_acc[client_idx]))

            if best_changed:     
                # print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
                write_log(args, ' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))
                if args.mode.lower() in ['fedbn', 'solo'] :
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
                # todo save gmap
                else:
                    torch.save({
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                    best_changed = False
                    
                # print(f'Average Test Accuracy: {np.mean(test_accs):.4f}')
    
    write_log(args, '==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

