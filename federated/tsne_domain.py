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
from utils.util import agg_dc_tokens, prepare_data, agg_rep
from utils import util
from utils.sam import SAM
from utils.aggregate import agg_smash_data
from sklearn import manifold
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--pcon', action='store_true', help='whether to pcon')
    parser.add_argument('--clscon', action='store_true', help='whether to pcon')
    parser.add_argument('--moon', action='store_true', help='whether to moon')
    parser.add_argument('--shuffle', action='store_true', help='whether to shuffle the order of majority/minority')
    parser.add_argument('--distinct', action='store_true', help='whether to distinct domain')
    parser.add_argument('--Ea_val', action='store_true', help='whether to Ea_val')
    parser.add_argument('--w_con', type=float, default=0.1, help='threshold std')
    parser.add_argument('--power_relu', type=float, default=1, help='threshold std')
    parser.add_argument('--power_cs', type=float, default=1, help='threshold std')
    parser.add_argument('--save_mean', type=float, default=-1, help='threshold std')
    parser.add_argument('--save_std', type=float, default=-1, help='threshold std')
    parser.add_argument('--no_val', action='store_true', help='whether to disable validation set')
    parser.add_argument('--freeze_pi', action='store_true', help='whether to freeze_pi')
    parser.add_argument('--mix', action='store_true', help='whether to mix dataset for face')
    parser.add_argument('--mix2', action='store_true', help='whether to mix dataset for face')
    parser.add_argument('--mix3', action='store_true', help='whether to mix dataset for face')
    parser.add_argument('--mix4', action='store_true', help='whether to mix dataset for face')
    parser.add_argument('--mix5', action='store_true', help='whether to mix dataset for face')
    parser.add_argument('--netdb', action='store_true', help='whether is run at netdb gpu')
    parser.add_argument('--cs', action='store_true', help='whether to use std for re-weight')
    parser.add_argument('--cq', action='store_true', help='whether to use loss c as power for re-weight')
    parser.add_argument('--color_jitter', action='store_true', help='whether to color_jitter for fairface')
    parser.add_argument('--cb', action='store_true', help='whether to cb for re-weighting')
    parser.add_argument('--debug', action='store_true', help='whether to debug for inference/test')
    parser.add_argument('--save_all_gmap', action='store_true', help='whether to save_all_gmap')
    parser.add_argument('--freeze_ckpt', action='store_true', help='whether to freeze_ckpt')
    parser.add_argument('--std_rw', action='store_true', help='divide ni with domain std over performance')
    parser.add_argument('--quan', type=float, default=0, help='whether to minimize client with loss smaller than 0.5 quantile')
    parser.add_argument('--small_test', action='store_true', help='whether to test small cluster')
    parser.add_argument('--split_test', action='store_true', help='whether to test split testing set')
    parser.add_argument('--binary_race', action='store_true', help='whether to test binary_race race distribution and find under-represented white people.')
    parser.add_argument('--weak_white', action='store_true', help='whether to test binary_race race distribution and find under-represented white people.')
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
    parser.add_argument('--test_freq', type = int, default=10, help ='test_freq')
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
    elif args.dataset.lower()[:6] == 'geo':
        # name of each datasets
        domain_num = 2
    elif args.dataset.lower()[:4] == 'pacs':
        # name of each datasets
        domain_num = 4
        args.num_classes = 5
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

    client_weights, sum_len, train_loaders, val_loaders, test_loaders, datasets = prepare_data(args)
    print(datasets)
    client_num = len(train_loaders)

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
    
    if args.weak_white:  exp_folder += f"_weak_white"
    if args.split_test:  exp_folder += f"_split"
    if args.small_test:  exp_folder += f"_small_test"
    if args.gender_label: exp_folder += "_gender_label"
    if args.binary_race: exp_folder += "_binary_race"
    if args.sam: exp_folder += f"_sam"
    if args.color_jitter:  exp_folder += f"_color_jitter"
    if args.cb:  exp_folder += f"_cb"
    if args.cq:  exp_folder += f"_cq"
    if args.cs:  exp_folder += f"_cs"
    print(exp_folder)
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, f'{args.mode}')
    if args.sam:
        SAVE_PATH = os.path.join(args.save_path, f'{args.mode}_sam_{args.sam}')
    if 'ccop' in args.mode.lower():
        SAVE_PATH += f"_q={args.q}"
        if args.power_cs != 1: SAVE_PATH += f"_pcs={args.power_cs}"
    GMAP_SAVE_PATH = args.gmap_path
    if GMAP_SAVE_PATH == 'none':
        GMAP_SAVE_PATH = f"{SAVE_PATH}_gmap"
        if args.mode.lower()!='ccop':
            GMAP_SAVE_PATH = GMAP_SAVE_PATH.replace(args.mode, f'ccop_q={args.q}')
        GMAP_SAVE_PATH = GMAP_SAVE_PATH.replace('cluster_-1', f"cluster_{args.cluster_num}")
        
    print(GMAP_SAVE_PATH)
    # todo demonstrate the tsne of all the inter domains

    # loading gmap

    # aggregation rep
    
    if args.dg:
        domain_num -= 1
    
    print(f"domain number = {domain_num}")
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
        #todo: remove
        if args.freeze_pi:
            for name, param in server_model.named_parameters():
                if 'meta_net' in name:
                    param.requires_grad = False
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
    
    # checkpoint = torch.load(SAVE_PATH)
    # server_model.load_state_dict(checkpoint['server_model'])
    # Start training
    #todo: remove

    done = False
    all_dc = np.empty((0, 768))
    y = np.empty(0, dtype=int)
    exist_domain = set()
    domain_idx = 0
    for client_idx in range(client_num):
        if datasets[client_idx] in exist_domain:
            print(f"{datasets[client_idx]} exist!")
            continue
        print(datasets[client_idx])
        domain_idx+=1
        exist_domain.add(datasets[client_idx])
        dc_tokens, _ = agg_dc_tokens(args, server_model, test_loaders[client_idx], device)
        domain_label = torch.full((dc_tokens.shape[0],), domain_idx)
        all_dc = np.concatenate((all_dc, dc_tokens), axis=0)
        y = np.concatenate((y, domain_label), axis=0)
        
        print(all_dc.shape)
        # all_feat = agg_rep(args. )
    # process extracted features with t-SNE
    feat_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(all_dc)

    # Normalization the processed features 
    x_min, x_max = feat_tsne.min(0), feat_tsne.max(0)
    X_norm = (feat_tsne - x_min) / (x_max - x_min)
    print(X_norm.shape)
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = X_norm[:,0]
    df["comp-2"] = X_norm[:,1]

    plt.figure(figsize=(8, 8))
    sns_plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", domain_idx),
                    data=df)
    sns_plot.set(xlabel=None, ylabel=None, xticks=[], xticklabels=[], yticks=[], yticklabels=[])
    plt.title(label=f"DC-Net ouput for {args.dataset}",
          fontsize=16)
    # plt.xticks([])
    # plt.yticks([])
    sns_plot.get_figure().savefig(f"./img/tsne_{exp_folder}.png")
    # for i in range(X_norm.shape[0]):
    #     plt.plot(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(y[i]), 
    #              label=str(i))
    # plt.savefig(f'./img/tsne_{exp_folder}.png')

    # for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
    #     source_data = source_data.cuda()
    #     target_data = target_data.cuda()
        
    #     # Mixed the source data and target data, or it'll mislead the running params
    #     #   of batch_norm. (runnning mean/var of soucre and target data are different.)
    #     mixed_data = torch.cat([source_data, target_data], dim=0)
    #     domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
    #     # set domain label of source data to be 1.
    #     domain_label[:source_data.shape[0]] = 1

    # Step 1 : train domain classifier
    
    # X = np.concatenate((X, feature_extractor(mixed_data).detach().cpu().numpy()), axis=0)
    # y = np.concatenate((y, domain_label.detach().cpu().numpy().squeeze()), axis=0)
    
    
    
    
    print('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

