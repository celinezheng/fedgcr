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
from utils.doprompt import GCR
from domainbed import hparams_registry, misc
import json
from utils.util import train, test, communication, train_fedprox, prepare_data, train_fedprompt, write_log, train_GCR, agg_rep, train_harmofl, train_sam, train_fedmix
from utils.util import train_propfair, train_fedsam
from utils import util
from utils.weight_perturbation import WPOptim
from utils.sam import SAM
from utils.aggregate import agg_smash_data
from utils.fpl import train_fpl, proto_aggregation

def test_score(args, server_model, test_loaders, datasets, best_epoch, gmap):
    domain_test_accs = {}
    cluster_test_accs = {}
    individual_test_acc = []
    server_model.eval()
    if gmap:
        for client_idx in gmap:
            cluster_test_accs[gmap[client_idx]] = list()
    for datasite in datasets:
        if args.split_test:
            key = datasite
        else:
            key = datasite.split("-")[0]
        domain_test_accs[key] = list()
    print(datasets)
    for client_idx, datasite in enumerate(datasets):
        if args.split_test:
            domain_name = datasite
        else:
            domain_name = datasite.split("-")[0]
        if datasite not in domain_test_accs or len(domain_test_accs[domain_name])==0:
            print(datasite)
            _, test_acc = test(server_model, test_loaders[client_idx], loss_fun, device, prompt_bank)
            domain_test_accs[domain_name].append(test_acc)
            if gmap:
                group_info = f"({gmap[client_idx]})"
            else:
                group_info = f""
            # print(' Test site-{:<25s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
            write_log(args, ' Test site-{:<25s} {}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, group_info, best_epoch, test_acc))
        print(f"{datasite}, domain = {domain_name}")
        test_acc = domain_test_accs[domain_name][-1]
        if gmap:
            cluster_test_accs[gmap[client_idx]].append(test_acc)
        individual_test_acc.append(test_acc)
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
    mean_domain = np.mean(domain_test_accs)
    msg = \
        f"Average Test Accuracy(group): {np.mean(domain_test_accs):.4f}, " \
        + f"Average Test Accuracy(individual): {np.mean(individual_test_acc):.4f},\n" \
        + f"domain std={std_domain:.4f}, " \
        + f"cluster std={std_cluster:.4f}, " \
        + f"individual std={std_individual:.4f}, " 
    write_log(args, f'{msg}\n')

    return domain_test_accs

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--clscon', action='store_true', help='whether to pcon')
    parser.add_argument('--moon', action='store_true', help='whether to moon')
    parser.add_argument('--shuffle', action='store_true', help='whether to shuffle the order of majority/minority')
    parser.add_argument('--distinct', action='store_true', help='whether to distinct domain')
    parser.add_argument('--debug', action='store_true', help='whether to debug for inference/test')
    parser.add_argument('--save_all_gmap', action='store_true', help='whether to save_all_gmap')
    parser.add_argument('--split_test', action='store_true', help='whether to test split testing set')
    parser.add_argument('--si', action='store_true', help='whether to use si only')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--w_con', type=float, default=0.1, help='lambda for contrastive loss')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--q', type = float, default=1, help ='q value for fairness')
    parser.add_argument('--test_freq', type = int, default=10, help ='test_freq')
    parser.add_argument('--batch', type = int, default=32, help ='batch size')
    parser.add_argument('--iters', type = int, default=10, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='DoPrompt', help='[FedBN | FedAvg | DoPrompt]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--deep', action='store_true', help ='deep prompt')
    parser.add_argument('--dataset', type=str, default='digit')
    parser.add_argument('--cluster', type=str, default='gmm')
    parser.add_argument('--percent', type = float, default= 0.2, help ='percentage of dataset to train')
    parser.add_argument('--num_classes', type = int, default=10, help ='number of classes')
    parser.add_argument('--seed', type = int, default=1, help ='random seed')
    parser.add_argument('--model', type = str, default='prompt', help='prompt | vit-linear')
    parser.add_argument('--cluster_num', type = int, default=-1, help ='cluster number')
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
    elif args.dataset.lower()[:6] == 'pacs':
        # name of each datasets
        domain_num = 4
        args.num_classes = 7
    elif args.dataset.lower()[:5] == 'digit':
        domain_num = 5
    else:
        import warnings
        warnings.warn("invalid args.dataset")
        exit(0)
    
    exp_folder = f'fed_{args.dataset}_{args.expname}_{args.ratio}_{args.seed}'
    if args.cluster_num != -1:
        cluster_num = args.cluster_num if args.mode.lower() in ['ccop', 'ablation'] else -1
        domain_num = args.cluster_num
        exp_folder += f"_cluster_{cluster_num}"
    else:
        args.cluster_num = domain_num
    if 'finch' in args.cluster.lower():
        domain_num += 3
    if args.shuffle:  exp_folder += f"_shuffle"
    if args.distinct:  exp_folder += f"_distinct"

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, f'{args.mode}')
    
    if 'ccop' in args.mode.lower():
        SAVE_PATH += f"_q={args.q}"
        if 'gmm' not in args.cluster.lower(): 
            SAVE_PATH += f"_{args.cluster.lower()}"
    if 'only_dcnet' in args.mode.lower():
        if args.w_con!=0.1: SAVE_PATH += f"_wcon={args.w_con}"

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
    write_log(args, '    domain_num: {}\n'.format(domain_num))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.mode, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.mode, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    
    
    loss_fun = nn.CrossEntropyLoss()
    client_weights, sum_len, train_loaders, val_loaders, test_loaders, datasets = prepare_data(args)
    print(datasets)
    client_num = len(train_loaders)
    
    print(f"domain number = {domain_num}")
    write_log(args, f"domain number = {domain_num}\n")
    prompt_bank = None
    # setup model
    if args.mode.lower() in ['ccop', 'only_dcnet']:
        server_model = GCR(num_classes=args.num_classes, hparams=hparams)
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
    
    # each local client model
    models = [copy.deepcopy(server_model) for _ in range(client_num)]
    pre_locals = [copy.deepcopy(server_model) for _ in range(client_num)]
    pre_glob = copy.deepcopy(server_model)
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
        test_accs = test_score(args, server_model, test_loaders, datasets, best_epoch, gmap)
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
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
    
    test_freq = args.test_freq
    if args.mode.lower()=='fedmix':
        Xg, Yg = agg_smash_data(args, server_model, train_loaders)

    train_losses = [1.0 for _ in range(client_num)]
    # naively aggregate feat, pi
    all_feat = None
    all_pi = None
    # average along clusters
    cluster_pis = None
    for a_iter in range(start_iter, args.iters):
        print(best_test)
        if args.mode.lower() == 'harmo-fl':
            optimizers = [WPOptim(params=models[idx].parameters(), base_optimizer=optim.Adam, lr=args.lr, alpha=0.05, weight_decay=1e-4) for idx in range(client_num)]
        elif args.mode.lower() == 'fedsam':
            optimizers = [SAM(params=models[idx].parameters(), base_optimizer=optim.Adam, lr=args.lr) for idx in range(client_num)]
        else:
            optimizers = [optim.AdamW(params=models[idx].parameters(), lr=0.001) for idx in range(client_num)]
        pre_pis = copy.deepcopy(all_pi)
        pre_feats = copy.deepcopy(all_feat)
        for wi in range(args.wk_iters):
            all_feat = None
            all_pi = None
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            write_log(args, "============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 

            for client_idx, model in enumerate(models):
                if args.mode.lower() in ['ccop', 'only_dcnet']:
                    if len(gmap) == 0: 
                        gidx = -1
                        pre_feat = None
                        pre_pi = None
                    else:
                        gidx = gmap[client_idx]
                        pre_pi = pre_pis[client_idx]
                        pre_feat = pre_feats[client_idx]
                    train_GCR(args, model, train_loaders[client_idx], loss_fun, device, 
                    gidx, pre_feat, cluster_pis, pre_pi,
                    pre_glob=pre_glob, pre_local=pre_locals[client_idx],
                    a_iter=a_iter)
                    feat_i = agg_rep(args, server_model, train_loaders[client_idx], device).unsqueeze(0)
                    pi_i = agg_rep(args, server_model, train_loaders[client_idx], device, True).unsqueeze(0)
                    if all_feat == None:
                        all_feat = feat_i
                        all_pi = pi_i
                    else:
                        all_feat = torch.concat((all_feat, feat_i)) 
                        all_pi = torch.concat((all_pi, pi_i)) 
                elif args.mode.lower() == 'ablation':
                    train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
                    feat_i = agg_rep(args, server_model, train_loaders[client_idx], device)
                    feat_i = feat_i.unsqueeze(0)
                    if all_feat == None:
                        all_feat = feat_i
                    else:
                        all_feat = torch.concat((all_feat, feat_i))
                elif args.mode.lower() == 'harmo-fl':
                    train_harmofl(args, model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
                elif args.mode.lower() == 'fedmix':
                    lamb = 0.05
                    train_fedmix(model, Xg, Yg, lamb, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
                elif args.mode.lower() == 'propfair':
                    train_loss, train_acc = train_propfair(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
                elif args.mode.lower() == 'fedsam':
                    train_loss, train_acc = train_fedsam(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
                else:
                    train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
                    train_losses[client_idx] = train_loss
        if args.mode.lower() in ['only_dcnet', 'ccop']:
            for key in server_model.state_dict().keys():
                if 'prompt' in key or 'meta_net' in key:
                    pre_glob.state_dict()[key].data.copy_(server_model.state_dict()[key])
            for param in pre_glob.parameters():
                param.requires_grad = False
            for i in range(client_num):
                for key in pre_locals[i].state_dict().keys():
                    if 'prompt' in key or 'meta_net' in key:
                        pre_locals[i].state_dict()[key].data.copy_(models[i].state_dict()[key])
                for param in pre_locals[i].parameters():
                    param.requires_grad = False
        with torch.no_grad():
            # Aggregation
            if args.mode.lower() != 'solo':
                if args.mode.lower() in ['ccop', 'ablation']:
                    print(Eas)
                # server_model, models, prompt_bank, gmap, cluster_pis = communication(args, len(gmap), server_model, models, client_weights, sum_len, client_num, domain_num, Eas, train_losses, a_iter, datasets, pre_clusters, all_feat, all_pi, prompt_bank)
                server_model, models, prompt_bank, gmap, cluster_pis = communication(args, len(gmap), server_model, models, client_weights, sum_len, client_num, domain_num, Eas, train_losses, a_iter, datasets, all_feat, all_pi, prompt_bank)
                
            # Report loss after aggregation
            train_acc_list = [None for j in range(client_num)]
            for client_idx, model in enumerate(models):
                train_loss, train_acc = test(model, train_loaders[client_idx], loss_fun, device, prompt_bank)
                train_acc_list[client_idx] = train_acc
                Eas[client_idx] = int(multi / train_loss)
                group_info = f"({gmap[client_idx]})" if args.mode.lower() in ['ccop', 'ablation'] else ""
                write_log(args, ' Site-{:<25s} {:<4s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx], group_info, train_loss, train_acc))
            write_log(args, f'Average Train Accuracy: {np.mean(train_acc_list):.4f}\n')
            save_criteria = train_acc_list
            # Record best
            if np.mean(save_criteria) > np.mean(best_acc):
                for client_idx in range(min(len(best_acc), client_num)):
                    best_acc[client_idx] = save_criteria[client_idx]
                    best_epoch = a_iter
                    best_changed=True
                    write_log(args, ' Best site-{:<25s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[client_idx], best_epoch, best_acc[client_idx]))
            if args.mode.lower()=='ccop' and GMAP_SAVE_PATH != 'none' and (best_changed or args.save_all_gmap):
                write_log(args, ' Saving the gmap checkpoint to {}...\n'.format(GMAP_SAVE_PATH))
                torch.save({
                    'a_iter': a_iter, 
                    'gmap': gmap
                }, GMAP_SAVE_PATH)
            if ((a_iter+1)*(wi+1)) % test_freq == 0:
                test_accs = test_score(args, server_model, test_loaders, datasets, best_epoch, gmap)
                best_changed = best_changed
            if best_changed:  
                best_changed = False
                # print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
                write_log(args, ' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))
                if  ((a_iter+1)*(wi+1)) % test_freq == 0 and np.mean(test_accs) > np.mean(best_test):
                    best_epoch = a_iter
                    if len(best_test) < len(test_accs):
                        best_test = [0. for _ in range(len(test_accs))]
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
                
    write_log(args, '==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

