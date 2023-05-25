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
from utils.util import train, train_doprompt, test, communication, train_fedprox, prepare_data, train_fedprompt, write_log, train_CoCoOP, agg_rep, train_harmofl, train_sam
from utils import util
from utils.weight_perturbation import WPOptim
from utils.sam import SAM

def test_score(server_model, test_loaders, datasets, best_epoch):
    domain_test_accs = {}
    individual_test_acc = []
    for client_idx, datasite in enumerate(datasets):
        if datasite not in domain_test_accs:
            _, test_acc = test(server_model, test_loaders[client_idx], loss_fun, device, prompt_bank)
            domain_test_accs[datasite] = test_acc
            # print(' Test site-{:<25s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
            write_log(args, ' Test site-{:<25s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
        print(datasite)
        individual_test_acc.append(domain_test_accs[datasite])
    domain_test_accs = list(domain_test_accs.values())
    
    std_domain = np.std(domain_test_accs, dtype=np.float64)
    std_individual = np.std(individual_test_acc, dtype=np.float64)
    write_log(args, f'Average Test Accuracy: {np.mean(domain_test_accs):.4f}, domain std={std_domain:.4f}, individual std={std_individual:.4f}\n')
    # todo individual std
    return domain_test_accs

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--color_jitter', action='store_true', help='whether to color_jitter for fairface')
    parser.add_argument('--debug', action='store_true', help='whether to debug for inference/test')
    parser.add_argument('--small_test', action='store_true', help='whether to test small cluster')
    parser.add_argument('--tune', action='store_true', help='whether to tune hparams')
    parser.add_argument('--si', action='store_true', help='whether to use si only')
    parser.add_argument('--sam', action='store_true', help='whether to use sam optimizer')
    parser.add_argument('--memory', action='store_true', help='whether to test memory usage of each algorithm')
    parser.add_argument('--dg', action='store_true', help='domain generalization')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--lambda_con', type=float, default=0.5, help='lambda for contrastive loss')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--q', type = float, default=1, help ='q value for fairness')
    parser.add_argument('--batch', type = int, default=64, help ='batch size')
    parser.add_argument('--iters', type = int, default=10, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='DoPrompt', help='[FedBN | FedAvg | DoPrompt]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--deep', action='store_true', help ='deep prompt')
    parser.add_argument('--dataset', type=str, default='digit')
    parser.add_argument('--percent', type = float, default= 0.2, help ='percentage of dataset to train')
    parser.add_argument('--num_classes', type = int, default=10, help ='number of classes')
    parser.add_argument('--seed', type = int, default=1, help ='random seed')
    parser.add_argument('--model', type = str, default='prompt', help='prompt | vit-linear')
    parser.add_argument("--gender_dis", choices=['iid', 'gender', 'gender_age', 'random_dis'], default='iid', help="gender distribution of each client")
    parser.add_argument('--cluster_num', type = int, default=-1, help ='cluster number')
    parser.add_argument('--target_domain', type = str, default='Clipart', help='Clipart, Infograph, ...')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--expname', type=str, default='prompt-sim')
    parser.add_argument('--ratio', type=float, default=1.0)
    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)
    if args.dataset.lower()[:6] == 'domain':
        # name of each datasets
        domain_num = 6
    elif args.dataset.lower()[:5] == 'digit':
        domain_num = 5
    elif args.dataset.lower()[:9] == 'fairface':
        domain_num = 7
        args.num_classes = 9
    else:
        import warnings
        warnings.warn("invalid args.dataset")
        exit(0)
    exp_folder = f'fed_{args.dataset}_{args.expname}_{args.ratio}_{args.seed}'
    if args.gender_dis != 'iid':
        domain_num = args.cluster_num
        exp_folder += f"_{args.gender_dis}_cluster_{args.cluster_num}"
    elif args.cluster_num != -1:
        domain_num = args.cluster_num
        exp_folder += f"_cluster_{args.cluster_num}"
    else:
        args.cluster_num = domain_num
    
    if args.small_test:  exp_folder += f"_small_test"
    if args.sam: exp_folder += f"_sam"
    if args.color_jitter:  exp_folder += f"_color_jitter"

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, f'{args.mode}')
    if args.dg:
        SAVE_PATH = os.path.join(args.save_path, f'{args.mode}_{args.target_domain}')
    if args.sam:
        SAVE_PATH = os.path.join(args.save_path, f'{args.mode}_sam_{args.sam}')
    if 'ccop' in args.mode.lower():
        SAVE_PATH = os.path.join(args.save_path, f'{args.mode}_q={args.q}')

    write_log(args, '==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    write_log(args, '===Setting===\n')
    write_log(args, '    dataset: {}\n'.format(args.dataset))
    write_log(args, '    lr: {}\n'.format(args.lr))
    write_log(args, '    batch: {}\n'.format(args.batch))
    write_log(args, '    iters: {}\n'.format(args.iters))
    write_log(args, '    wk_iters: {}\n'.format(args.wk_iters))
    write_log(args, '    si: {}\n'.format(args.si))
    write_log(args, '    domain_num: {}\n'.format(domain_num))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.mode, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.mode, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    
    
    loss_fun = nn.CrossEntropyLoss()
    client_weights, train_loaders, val_loaders, test_loaders, datasets, target_loader = prepare_data(args)
    print(client_weights)
    client_num = len(train_loaders)
    if args.dg:
        domain_num -= 1
    
    print(f"domain number = {domain_num}")
    write_log(args, f"domain number = {domain_num}\n")
    prompt_bank = None
    # setup model
    feat_bank = nn.Parameter(
            torch.empty(domain_num, 768, requires_grad=False).normal_(std=0.02)
        ).to(device)
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
        prompt_bank = util.random_replace(all_pi, prompt_bank)
        print(prompt_bank.shape)
    elif args.mode.lower() in ['cocoop', 'nova', 'ccop']:
        server_model = CoCoOP(num_classes=args.num_classes, hparams=hparams).to(device)
    elif args.mode.lower() in ['full']:
        model_type="sup_vitb16_imagenet21k"
        server_model = PromptViT(model_type=model_type, args=args).to(device)
    # fedavg
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

    
    # each local client model
    models = [copy.deepcopy(server_model).to(device) for _ in range(client_num)]
    best_changed = False

    if args.test:
        write_log(args, 'Loading snapshots...\n')
        checkpoint = torch.load(SAVE_PATH)
        best_epoch, best_acc = checkpoint['best_epoch'], checkpoint['best_acc']
        server_model.load_state_dict(checkpoint['server_model'])
        test_accs = {}
        if args.mode.lower() in ['fedbn', 'solo']:
            for client_idx in range(domain_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
            for client_idx, datasite in enumerate(datasets):
                if datasite in test_accs: continue
                _, test_acc = test(models[client_idx], test_loaders[client_idx], loss_fun, device, prompt_bank)
                test_accs[datasite] = test_acc
                write_log(args, ' Test site-{:<25s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
            test_accs = list(test_accs.values())
            write_log(args, f'Average Test Accuracy: {np.mean(test_accs):.4f}\n')
        else:
            test_accs = test_score(server_model, test_loaders, datasets, best_epoch)
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() in ['fedbn', 'solo']:
            for client_idx in range(domain_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        best_epoch, best_acc = checkpoint['best_epoch'], checkpoint['best_acc']
        best_test = checkpoint['best_test']
        start_iter = int(checkpoint['a_iter']) + 1
        best_agg = np.mean(best_acc)

        print('Resume training from epoch {}'.format(start_iter))
    else:
        # log the best for each model on all datasets
        best_epoch = 0
        best_acc = [0. for j in range(client_num)] 
        best_test = [0. for j in range(domain_num)] 
        start_iter = 0
        best_agg = 0
    gmap = {}
    multi = 100
    if args.mode.lower() in ['nova', 'ccop']:
        write_log(args, f'multiply {multi} for Ea!\n')
        write_log(args, f'use train loss for Ea\n')
    Eas = [multi for _ in range(client_num)]
    train_losses = [1.0 for _ in range(client_num)]
    # Start training
    all_feat = None
    for a_iter in range(start_iter, args.iters):
        print(best_test)
        if args.mode.lower() in ['doprompt', 'fedprompt', 'cocoop', 'nova']:
            optimizers = [None for _ in range(client_num)]
        elif args.mode.lower() == 'harmo-fl':
            optimizers = [WPOptim(params=models[idx].parameters(), base_optimizer=optim.Adam, lr=args.lr, alpha=0.05, weight_decay=1e-4) for idx in range(client_num)]
        elif args.mode.lower() == 'ccop' and args.sam:
            optimizers = [SAM(params=models[idx].classifier.parameters(), base_optimizer=optim.Adam, lr=args.lr) for idx in range(client_num)]
            prompt_opts = [SAM(params=[models[idx].prompt_tokens], base_optimizer=optim.Adam, lr=args.lr) for idx in range(client_num)]
            project_opts = [SAM(params=models[idx].meta_net.parameters(), base_optimizer=optim.Adam, lr=args.lr) for idx in range(client_num)]
        else:
            optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            all_feat = None
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
                    else: gidx = gmap[client_idx]
                    train_fedprompt(gidx, model, train_loaders[client_idx], prompt_bank, device)
                elif args.mode.lower() in ['nova', 'ccop']:
                    if args.sam:
                        train_sam(model, train_loaders[client_idx], prompt_opts[client_idx], prompt_opts[client_idx], optimizers[client_idx], loss_fun, device)
                    else:
                        train_CoCoOP(args, model, train_loaders[client_idx], loss_fun, device)
                    feat_i = agg_rep(args, server_model, train_loaders[client_idx], device)
                    feat_i = feat_i.unsqueeze(0)
                    
                    if all_feat == None:
                        all_feat = feat_i
                    else:
                        all_feat = torch.concat((all_feat, feat_i)) 
                elif args.mode.lower() in ['cocoop']:
                    train_CoCoOP(args, model, train_loaders[client_idx], loss_fun, device)
                elif args.mode.lower() == 'harmo-fl':
                    train_harmofl(args, model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
                else:
                    train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
                    # for q-ffl
                    train_losses[client_idx] = train_loss
        
        with torch.no_grad():
            # Aggregation
            if args.mode.lower() != 'solo':
                if args.mode.lower() in ['nova', 'ccop']:
                    print(Eas)
                server_model, models, prompt_bank, gmap = communication(args, len(gmap), gmap, server_model, models, client_weights, client_num, domain_num, Eas, train_losses, a_iter, all_feat, prompt_bank, feat_bank)

            # Report loss after aggregation
            for client_idx, model in enumerate(models):
                train_loss, train_acc = test(model, train_loaders[client_idx], loss_fun, device, prompt_bank)
                Eas[client_idx] = int(multi / train_loss)
                write_log(args, ' Site-{:<25s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))

            # Validation
            val_acc_list = [None for j in range(client_num)]
            for client_idx, model in enumerate(models):
                val_loss, val_acc = test(model, val_loaders[client_idx], loss_fun, device, prompt_bank)
                val_acc_list[client_idx] = val_acc
                # train_accs[client_idx] = int(multi * val_acc)
                # print(' Site-{:<25s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss, val_acc))
                group_info = f"({gmap[client_idx]})" if args.mode.lower()=='ccop' else ""
                write_log(args, ' Site-{:<25s} {:<4s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], group_info, val_loss, val_acc))
            write_log(args, f'Average Valid Accuracy: {np.mean(val_acc_list):.4f}\n')
                    
            # Record best
            if args.mode.lower() in ['nova', 'ccop']:
                if args.sam: threshold = args.iters - 10
                else: threshold = args.iters - 5
                threshold = max(10, threshold)
                if args.debug: threshold = 0

                cnt = [0 for _ in range(domain_num)]
                accs = [0 for _ in range(domain_num)]
                agg = 0
                domain_cnt = 0
                for client_idx in range(client_num):
                    accs[gmap[client_idx]] += val_acc_list[client_idx]
                    cnt[gmap[client_idx]] += 1
                for di in range(domain_num):
                    if cnt[di]==0:continue
                    domain_cnt+=1
                    agg += (accs[di]/cnt[di])
                agg /= domain_cnt
                write_log(args, 'Aggregated Acc | Val Acc: {:.4f}\n'.format(agg))
                # if agg > best_agg:
                if np.mean(val_acc_list) > np.mean(best_acc):
                    best_agg = agg
                    best_epoch = a_iter
                    best_changed=True
                    for client_idx in range(client_num):
                        best_acc[client_idx] = val_acc_list[client_idx]
                        group_info = f"({gmap[client_idx]})" if args.mode.lower()=='ccop' else ""
                        write_log(args, ' Best site-{:<25s}{:<4s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[client_idx], group_info, best_epoch, best_acc[client_idx]))
                if ((a_iter+1)*(wi+1)) > threshold:
                    test_accs = test_score(server_model, test_loaders, datasets, best_epoch)
                    if np.mean(test_accs) > np.mean(best_test):
                        best_changed = True
                        best_epoch = a_iter
                        for i in range(len(test_accs)):
                            best_test[i] = test_accs[i]
                    else:
                        best_changed = False
                    write_log(args, f'Average Test Accuracy: {np.mean(test_accs):.4f}\n')        
            elif np.mean(val_acc_list) > np.mean(best_acc):
                for client_idx in range(client_num):
                    best_acc[client_idx] = val_acc_list[client_idx]
                    best_epoch = a_iter
                    best_changed=True
                    write_log(args, ' Best site-{:<25s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[client_idx], best_epoch, best_acc[client_idx]))
                best_agg = np.mean(best_acc)
                   
            if best_changed:  
                best_changed = False
                # print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
                write_log(args, ' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))
                if args.mode.lower() in ['fedbn', 'solo']:
                    idxs = [0 for _ in range(6)]
                    if args.dataset == 'digit':
                        if 'uneven-1' in args.expname:
                            idxs = [3, 6, 7, 8, 9]
                        elif 'unevn-2' in args.expname:
                            idxs = [4, 5, 6, 8, 9]
                        elif args.expname=='even':
                            idxs = [1, 3, 5, 7, 9]
                        else:
                            write_log(args, 'invalid expname!!!!\n')
                    else:
                        if 'uneven-4' in args.expname:
                            idxs = [3, 4, 8, 9, 10]
                        elif 'uneven-2' in args.expname:
                            idxs = [5, 8, 9, 10, 11]
                        elif args.expname == 'even':
                            idxs = [1, 3, 5, 7, 9]
                        else:
                            write_log(args, 'invalid expname!!!!\n')
                    print(idxs)
                    torch.save({
                        'model_0': models[idxs[0]].state_dict(),
                        'model_1': models[idxs[1]].state_dict(),
                        'model_2': models[idxs[2]].state_dict(),
                        'model_3': models[idxs[3]].state_dict(),
                        'model_4': models[idxs[4]].state_dict(),
                        'model_5': models[client_num-1].state_dict(),
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                # todo save gmap
                else:
                    if ((a_iter+1)*(wi+1)) % 10 == 0:
                        test_accs = test_score(server_model, test_loaders, datasets, best_epoch)
                        if np.mean(test_accs) > np.mean(best_test):
                            best_epoch = a_iter
                            for i in range(len(test_accs)):
                                best_test[i] = test_accs[i]
                   
                    torch.save({
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter, 
                        'best_test': best_test
                    }, SAVE_PATH)
                 
                # print(f'Average Test Accuracy: {np.mean(test_accs):.4f}')
    
    write_log(args, '==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

