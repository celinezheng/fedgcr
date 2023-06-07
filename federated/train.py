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

def test_score(server_model, test_loaders, datasets, best_epoch, gmap):
    domain_test_accs = {}
    cluster_test_accs = {}
    individual_test_acc = []
    if gmap:
        for client_idx in gmap:
            cluster_test_accs[gmap[client_idx]] = list()
    for datasite in datasets:
        domain_test_accs[datasite.split("-")[0]] = list()
    print(datasets)
    for client_idx, datasite in enumerate(datasets):
        domain_name = datasite.split("-")[0]
        if datasite not in domain_test_accs or len(domain_test_accs[domain_name])==0:
            print(datasite)
            _, test_acc = test(server_model, test_loaders[client_idx], loss_fun, device, prompt_bank)
            domain_test_accs[domain_name].append(test_acc)
            if gmap:
                cluster_test_accs[gmap[client_idx]].append(test_acc)
                group_info = f"({gmap[client_idx]})"
            else:
                group_info = f""
            # print(' Test site-{:<25s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
            write_log(args, ' Test site-{:<25s} {}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, group_info, best_epoch, test_acc))
        print(f"{datasite}, domain = {domain_name}")
        individual_test_acc.append(domain_test_accs[domain_name][-1])
    if gmap:
        for gidx in gmap.values():
            cluster_test_accs[gidx] = np.mean(cluster_test_accs[gidx], dtype=np.float64)
    for name in domain_test_accs:
        domain_test_accs[name] = np.mean(domain_test_accs[name], dtype=np.float64)
    for name, acc in domain_test_accs.items():
        write_log(args, f"{name}: {acc:.4f}, ")
    write_log(args, "\n")
    print(domain_test_accs)
    cluster_test_accs = list(cluster_test_accs.values())
    domain_test_accs = list(domain_test_accs.values())
    if gmap: std_cluster = np.std(cluster_test_accs, dtype=np.float64)
    else: std_cluster = -1
    std_domain = np.std(domain_test_accs, dtype=np.float64)
    std_individual = np.std(individual_test_acc, dtype=np.float64)
    msg = \
        f"Average Test Accuracy(group): {np.mean(domain_test_accs):.4f}, " \
        + f"Average Test Accuracy(individual): {np.mean(individual_test_acc):.4f},\n" \
        + f"domain std={std_domain:.4f}, " \
        + f"cluster std={std_cluster:.4f}, " \
        + f"individual std={std_individual:.4f}, " 
    write_log(args, f'{msg}\n')
    # todo individual std
    return domain_test_accs

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--color_jitter', action='store_true', help='whether to color_jitter for fairface')
    parser.add_argument('--relu', action='store_true', help='whether to relu for re-weighting')
    parser.add_argument('--debug', action='store_true', help='whether to debug for inference/test')
    parser.add_argument('--save_all_gmap', action='store_true', help='whether to save_all_gmap')
    parser.add_argument('--freeze_ckpt', action='store_true', help='whether to freeze_ckpt')
    parser.add_argument('--std_rw', action='store_true', help='divide ni with domain std over performance')
    parser.add_argument('--quan', type=float, default=0, help='whether to minimize client with loss smaller than 0.5 quantile')
    parser.add_argument('--small_test', action='store_true', help='whether to test small cluster')
    parser.add_argument('--split_test', action='store_true', help='whether to test split testing set')
    parser.add_argument('--binary_race', action='store_true', help='whether to test binary_race race distribution and find under-represented white people.')
    parser.add_argument('--gender_label', action='store_true', help='whether to predict gender')
    parser.add_argument('--tune', action='store_true', help='whether to tune hparams')
    parser.add_argument('--si', action='store_true', help='whether to use si only')
    parser.add_argument('--sam', action='store_true', help='whether to use sam optimizer')
    parser.add_argument('--memory', action='store_true', help='whether to test memory usage of each algorithm')
    parser.add_argument('--dg', action='store_true', help='domain generalization')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--lambda_con', type=float, default=0.5, help='lambda for contrastive loss')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--q', type = float, default=1, help ='q value for fairness')
    parser.add_argument('--save_iter', type = int, default=-1, help ='save_iter')
    parser.add_argument('--batch', type = int, default=32, help ='batch size')
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
    parser.add_argument("--gender_dis", choices=['iid', 'gender', 'gender_age', 'random_dis'], default='random_dis', help="gender distribution of each client")
    parser.add_argument('--cluster_num', type = int, default=-1, help ='cluster number')
    parser.add_argument('--target_domain', type = str, default='Clipart', help='Clipart, Infograph, ...')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--expname', type=str, default='prompt-sim')
    parser.add_argument('--gmap_path', type=str, default='none')
    parser.add_argument('--ratio', type=float, default=1.0)
    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)
    args.save_all_gmap = args.save_all_gmap and args.mode.lower()=='ccop'
    if args.dataset.lower()[:6] == 'domain':
        # name of each datasets
        domain_num = 6
    elif args.dataset.lower()[:5] == 'digit':
        domain_num = 5
    elif args.dataset.lower()[:9] == 'fairface':
        domain_num = 7
        if args.gender_label: args.num_classes = 2
        else: args.num_classes = 9
    else:
        import warnings
        warnings.warn("invalid args.dataset")
        exit(0)
    exp_folder = f'fed_{args.dataset}_{args.expname}_{args.ratio}_{args.seed}'
    if args.dataset.lower()[:8]=='fairface':
        cluster_num = args.cluster_num if args.mode.lower() in ['ccop', 'ablation'] else -1
        if args.gender_dis != 'iid':
            domain_num = args.cluster_num
            exp_folder += f"_{args.gender_dis}_cluster_{cluster_num}"
        elif args.cluster_num != -1:
            domain_num = args.cluster_num
            exp_folder += f"_cluster_{cluster_num}"
        else:
            args.cluster_num = domain_num
    
    if args.small_test:  exp_folder += f"_small_test"
    if args.gender_label: exp_folder += "_gender_label"
    if args.binary_race: exp_folder += "_binary_race"
    if args.sam: exp_folder += f"_sam"
    if args.color_jitter:  exp_folder += f"_color_jitter"
    if args.relu:  exp_folder += f"_relu"

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, f'{args.mode}')
    if args.sam:
        SAVE_PATH = os.path.join(args.save_path, f'{args.mode}_sam_{args.sam}')
    if 'ccop' in args.mode.lower():
        SAVE_PATH += f"_q={args.q}"
    GMAP_SAVE_PATH = args.gmap_path
    if GMAP_SAVE_PATH == 'none':
        GMAP_SAVE_PATH = f"{SAVE_PATH}_gmap"
        if args.mode.lower()!='ccop':
            GMAP_SAVE_PATH = GMAP_SAVE_PATH.replace(args.mode, f'ccop_q={args.q}')
        GMAP_SAVE_PATH = GMAP_SAVE_PATH.replace('cluster_-1', f"cluster_{args.cluster_num}")
        
    print(GMAP_SAVE_PATH)
    write_log(args, '==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    write_log(args, '===Setting===\n')
    write_log(args, '    dataset: {}\n'.format(args.dataset))
    write_log(args, '    lr: {}\n'.format(args.lr))
    write_log(args, '    batch: {}\n'.format(args.batch))
    write_log(args, '    iters: {}\n'.format(args.iters))
    write_log(args, '    wk_iters: {}\n'.format(args.wk_iters))
    write_log(args, '    si: {}\n'.format(args.si))
    write_log(args, '    domain_num: {}\n'.format(domain_num))
    write_log(args, '    quantile: {}\n'.format(args.quan))
    write_log(args, '    std_rw: {}\n'.format(args.std_rw))
    write_log(args, '    gender_label: {}\n'.format(args.gender_label))
    write_log(args, '    binary_race: {}\n'.format(args.binary_race))
    write_log(args, '    save_iter: {}\n'.format(args.save_iter))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.mode, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.mode, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    
    
    loss_fun = nn.CrossEntropyLoss()
    client_weights, sum_len, train_loaders, val_loaders, test_loaders, datasets, target_loader = prepare_data(args)
    print(datasets)
    client_num = len(train_loaders)
    if args.dg:
        domain_num -= 1
    
    print(f"domain number = {domain_num}")
    write_log(args, f"domain number = {domain_num}\n")
    prompt_bank = None
    # setup model
    if args.mode.lower() == 'doprompt':
        server_model = DoPrompt(num_classes=args.num_classes, num_domains=client_num, hparams=hparams)
    elif args.mode.lower() == 'fedprompt':
        server_model = FedPrompt(num_classes=args.num_classes, hparams=hparams, lambda_con=args.lambda_con)
        prompt_bank = nn.Parameter(
            torch.empty(domain_num, 4, 768, requires_grad=False).normal_(std=0.02)
        )
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
        server_model = CoCoOP(num_classes=args.num_classes, hparams=hparams)
    elif args.mode.lower() in ['full']:
        model_type="sup_vitb16_imagenet21k"
        server_model = PromptViT(model_type=model_type, args=args)
    # fedavg, ablation
    else:
        model_type="sup_vitb16_imagenet21k"
        server_model = PromptViT(model_type=model_type, args=args)
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
    models = [copy.deepcopy(server_model) for _ in range(client_num)]
    best_changed = False
    try:
        gmap_ckpt = torch.load(GMAP_SAVE_PATH)
        gmap = gmap_ckpt['gmap']
    except:
        gmap = None
    
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
            test_accs = test_score(server_model, test_loaders, datasets, best_epoch, gmap)
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
        print(f'Resume training from epoch {start_iter}, best_test={np.mean(best_test):.3f}')
    else:
        # log the best for each model on all datasets
        best_epoch = 0
        best_acc = [0. for j in range(client_num)] 
        best_test = [0. for j in range(domain_num)] 
        start_iter = 0

    # Start training
    gmap = {}
    multi = 100
    Eas = [multi for _ in range(client_num)]
    all_feat = None
    if args.mode.lower() in ['nova', 'ccop']:
        write_log(args, f'multiply {multi} for Ea!\n')
        write_log(args, f'use train loss for Ea\n')
    train_losses = [1.0 for _ in range(client_num)]
    if args.freeze_ckpt and start_iter>0: 
        args.iters = start_iter + 1
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
                elif args.mode.lower() == 'ablation':
                    train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
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
                    train_losses[client_idx] = train_loss
        
        with torch.no_grad():
            # Aggregation
            if args.mode.lower() != 'solo':
                if args.mode.lower() in ['nova', 'ccop', 'ablation']:
                    print(Eas)
                server_model, models, prompt_bank, gmap = communication(args, len(gmap), server_model, models, client_weights, sum_len, client_num, domain_num, Eas, train_losses, a_iter, all_feat, prompt_bank)

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
            if args.mode.lower() in ['nova', 'ccop', 'ablation']:
                if args.sam or args.dataset.lower() != 'digit': threshold = args.iters - 10
                else: threshold = args.iters - 5
                threshold = max(10, threshold)
                if args.debug: threshold = 0
                if np.mean(val_acc_list) > np.mean(best_acc):
                    best_epoch = a_iter
                    best_changed=True
                    for client_idx in range(client_num):
                        best_acc[client_idx] = val_acc_list[client_idx]
                        group_info = f"({gmap[client_idx]})" if args.mode.lower()=='ccop' else ""
                        write_log(args, ' Best site-{:<25s}{:<4s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[client_idx], group_info, best_epoch, best_acc[client_idx]))
                if ((a_iter+1)*(wi+1)) > threshold:
                    test_accs = test_score(server_model, test_loaders, datasets, best_epoch, gmap)
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
            if args.mode.lower()=='ccop' and GMAP_SAVE_PATH != 'none' and (best_changed or args.save_all_gmap):
                write_log(args, ' Saving the gmap checkpoint to {}...\n'.format(GMAP_SAVE_PATH))
                torch.save({
                    'a_iter': a_iter, 
                    'gmap': gmap
                }, GMAP_SAVE_PATH)
            best_changed = a_iter==args.save_iter or (best_changed and not args.freeze_ckpt)
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
                        test_accs = test_score(server_model, test_loaders, datasets, best_epoch, gmap)
                        if np.mean(test_accs) > np.mean(best_test):
                            best_epoch = a_iter
                            for i in range(len(test_accs)):
                                best_test[i] = test_accs[i]
                   
                    torch.save({
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter, 
                        'best_test': best_test,
                        'gmap': gmap
                    }, SAVE_PATH)
                
                 
                # print(f'Average Test Accuracy: {np.mean(test_accs):.4f}')
    
    write_log(args, '==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

