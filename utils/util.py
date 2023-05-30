from tqdm import tqdm
import sys, os
import torch
import torchvision.transforms as transforms
from utils.data_utils import DomainNetDataset, DigitsDataset, FairFaceIIDDataset, FairFaceGenderDataset
import math
import torch.nn.functional as tf
import numpy as np
import copy
def write_log(args, msg):
    log_path = f'../logs/{args.dataset}_{args.expname}_{args.ratio}_{args.seed}'
    log_fname = f'{args.mode}.log'
    if args.dg:
        log_path = f'../logs/{args.dataset}_{args.expname}_{args.target_domain}'
    if args.gender_dis != 'iid':
        log_path += f"_{args.gender_dis}_cluster_{args.cluster_num}"
    else:
        log_path += f"_{args.cluster_num}"
    if args.small_test: log_path += "_small_test"
    if args.gender_label: log_path += "_gender_label"
    if args.binary_race: log_path += "_binary_race"
    if args.sam: log_path += f"_sam"
    if args.color_jitter: log_path += f"_color_jitter"
    if args.q!=1: log_fname = f'{args.mode}_q={args.q}.log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(os.path.join(log_path, log_fname), 'a') as logfile:
        logfile.write(msg)

def prepare_digit_uneven(args):
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
    mnist_trainset     = DigitsDataset(data_path=os.path.join(base_path, "MNIST"), channels=1, percent=args.percent, train=True,  transform=transform_mnist)
    mnist_testset      = DigitsDataset(data_path=os.path.join(base_path, "MNIST"), channels=1, percent=args.percent, train=False, transform=transform_mnist)
    test_len = int(args.percent * len(mnist_testset))
    mnist_testset = torch.utils.data.Subset(mnist_testset, list(range(len(mnist_testset)))[:test_len])

    # SVHN
    svhn_trainset      = DigitsDataset(data_path=os.path.join(base_path, "SVHN"), channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
    svhn_testset       = DigitsDataset(data_path=os.path.join(base_path, "SVHN"), channels=3, percent=args.percent,  train=False, transform=transform_svhn)
    test_len = int(args.percent * len(svhn_testset))
    svhn_testset = torch.utils.data.Subset(svhn_testset, list(range(len(svhn_testset)))[:test_len])

    # USPS
    usps_trainset      = DigitsDataset(data_path=os.path.join(base_path, "USPS"), channels=1, percent=args.percent,  train=True,  transform=transform_usps)
    usps_testset       = DigitsDataset(data_path=os.path.join(base_path, "USPS"), channels=1, percent=args.percent,  train=False, transform=transform_usps)
    test_len = int(args.percent * len(usps_testset))
    usps_testset = torch.utils.data.Subset(usps_testset, list(range(len(usps_testset)))[:test_len])

    # Synth Digits
    synth_trainset     = DigitsDataset(data_path=os.path.join(base_path, "SynthDigits"), channels=3, percent=args.percent,  train=True,  transform=transform_synth)
    synth_testset      = DigitsDataset(data_path=os.path.join(base_path, "SynthDigits"), channels=3, percent=args.percent,  train=False, transform=transform_synth)
    test_len = int(args.percent * len(synth_testset))
    synth_testset = torch.utils.data.Subset(synth_testset, list(range(len(synth_testset)))[:test_len])

    # MNIST-M
    mnistm_trainset     = DigitsDataset(data_path=os.path.join(base_path, "MNIST_M"), channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
    mnistm_testset      = DigitsDataset(data_path=os.path.join(base_path, "MNIST_M"), channels=3, percent=args.percent,  train=False, transform=transform_mnistm)
    test_len = int(args.percent * len(mnistm_testset))
    mnistm_testset = torch.utils.data.Subset(mnistm_testset, list(range(len(mnistm_testset)))[:test_len])

    # min_data_len = min(len(dataset)) * args.persent
    # ori_data_len = min(len(mnist_trainset), len(svhn_trainset), len(usps_trainset), len(synth_trainset), len(mnistm_trainset))
    train_sets = {
        'MNIST': mnist_trainset,
        'SVHN': svhn_trainset,
        'USPS': usps_trainset,
        'SynthDigits': synth_trainset,
        'MNIST-M': mnistm_trainset,
        }
    test_sets = {
        'MNIST': mnist_testset,
        'SVHN': svhn_testset,
        'USPS': usps_testset,
        'SynthDigits': synth_testset,
        'MNIST-M': mnistm_testset,
        }
    # len_dataset = {
    #     'MNIST': int(0.8 * len(mnist_trainset) ),
    #     'SVHN': int(0.45 * len(svhn_trainset)),
    #     'USPS': int(0.4 * len(usps_trainset)),
    #     'SynthDigits': int(0.5 * len(synth_trainset)),
    #     'MNIST-M': int(0.3 * len(mnistm_trainset)),
    #     }
    len_dataset = {
        'MNIST': int(0.8 * len(mnist_trainset) ),
        'SVHN': int(0.4 * len(svhn_trainset)),
        'USPS': int(0.4 * len(usps_trainset)),
        'SynthDigits': int(0.5 * len(synth_trainset)),
        'MNIST-M': int(0.2 * len(mnistm_trainset)),
        }
    client_nums = {}
    if args.dg:
        if 'uneven-1' in args.expname.lower():
            data_len = [4, 4, 1, 1]
        elif 'uneven-2' in args.expname.lower():
            data_len = [7, 1, 1, 1]
        else:
            data_len = [3, 3, 3, 3]
    else:
        decay_order = ['MNIST', 'USPS', 'SynthDigits', 'MNIST-M', 'SVHN']
        if 'uneven' in args.expname.lower():
            # data_len = [4, 1, 1, 3, 1]
            decay_speed = args.ratio
            for i, name in enumerate(decay_order):
                client_nums[name] = round(np.float_power(decay_speed, len(decay_order)-i-1))
                if i==0:
                    len_dataset[name] = len(train_sets[name])
                else:
                    len_dataset[name] = int(len_dataset[decay_order[i-1]]/decay_speed)
        else:
            min_len = int(0.5 * len(mnist_trainset))
            decay_speed = 3
            for i, name in enumerate(decay_order):
                client_nums[name] = 2
                len_dataset[name] = min_len
    
    for name, val in len_dataset.items():
        print(f"{name}: {val * 0.6}")
    target_loader = None
    client_weights = []
    sum_len = 0
    if args.dg:
        print(f"target domain is {args.target_domain}")
        client_nums[args.target_domain] = 0
        target_loader = torch.utils.data.DataLoader(test_sets[args.target_domain], batch_size=args.batch, shuffle=False)
    print(client_nums)
    train_loaders, val_loaders, test_loaders = [], [], []
    datasets = []
    train_ratio = 0.8
    for key, value in client_nums.items():
        all_len = len_dataset[key]
        all_train_len = int(all_len * train_ratio)
        all_val_len = int(all_len * (1 - train_ratio))
        cur_dataset_len = len_dataset[key]
        train_begin = 0
        valid_begin = -all_val_len
        partition_num = (np.float_power(decay_speed, value)-1) / (decay_speed - 1)

        test_loader = torch.utils.data.DataLoader(test_sets[key], batch_size=1, shuffle=False)
        for j in range(value):
            train_len = int(all_train_len * np.float_power(decay_speed, j) / partition_num)
            val_len = int(all_val_len * np.float_power(decay_speed, j) / partition_num)
            datasets.append(key)
            cur_trainset = torch.utils.data.Subset(train_sets[key], list(range(all_train_len))[train_begin : train_begin+train_len])
            cur_valset = torch.utils.data.Subset(train_sets[key], list(range(cur_dataset_len))[-valid_begin : -valid_begin+val_len])
            train_loader = torch.utils.data.DataLoader(cur_trainset, batch_size=args.batch, shuffle=True)
            val_loader = torch.utils.data.DataLoader(cur_valset, batch_size=args.batch, shuffle=False)
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
            test_loaders.append(test_loader)
            client_weights.append(len(cur_trainset))
            sum_len += len(cur_trainset)
            train_begin += train_len
            valid_begin += val_len
            # print(len(cur_trainset), len(cur_valset))
    print(client_weights)
    write_log(args, f"data_number=[")
    for ni in client_weights:
        write_log(args, f"{ni},")
    write_log(args, f"\nclient_nums=[")
    for name in decay_order:
        write_log(args, f"{client_nums[name]},")
    write_log(args, f"]\nlen_dataset=[")
    for name in decay_order:
        write_log(args, f"{len_dataset[name]},")
    write_log(args, f"]\n")
    client_weights = [ci/sum_len for ci in client_weights]
    # check_labels(args, train_loaders)
    return client_weights, sum_len, train_loaders, val_loaders, test_loaders, datasets, target_loader

def prepare_domainnet_uneven(args):
    data_base_path = '../../data'
    transform_train = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
    ])

    # clipart
    clipart_trainset = DomainNetDataset(data_base_path, 'clipart', transform=transform_train)
    clipart_testset = DomainNetDataset(data_base_path, 'clipart', transform=transform_test, train=False)
    # infograph
    infograph_trainset = DomainNetDataset(data_base_path, 'infograph', transform=transform_train)
    infograph_testset = DomainNetDataset(data_base_path, 'infograph', transform=transform_test, train=False)
    # painting
    painting_trainset = DomainNetDataset(data_base_path, 'painting', transform=transform_train)
    painting_testset = DomainNetDataset(data_base_path, 'painting', transform=transform_test, train=False)
    # quickdraw
    quickdraw_trainset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_train)
    quickdraw_testset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_test, train=False)
    # real
    real_trainset = DomainNetDataset(data_base_path, 'real', transform=transform_train)
    real_testset = DomainNetDataset(data_base_path, 'real', transform=transform_test, train=False)
    # sketch
    sketch_trainset = DomainNetDataset(data_base_path, 'sketch', transform=transform_train)
    sketch_testset = DomainNetDataset(data_base_path, 'sketch', transform=transform_test, train=False)

    # min_data_len = int(min(len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset)))
    test_sets = {
        'Clipart': clipart_testset,
        'Infograph': infograph_testset,
        'Painting': painting_testset,
        'QuickDraw': quickdraw_testset,
        'Real': real_testset,
        'Sketch':sketch_testset
        }
    train_sets = {
        'Clipart': clipart_trainset,
        'Infograph': infograph_trainset,
        'Painting': painting_trainset,
        'QuickDraw': quickdraw_trainset,
        'Real': real_trainset,
        'Sketch':sketch_trainset
        }
    len_dataset = {}
    client_nums = {}
    decay_order = ['Clipart', 'Real', 'Painting', 'Sketch', 'QuickDraw', 'Infograph']
    if 'uneven' in args.expname.lower():
        # client number = 1.4^k, k=0~5
        # data_len = {5, 4, 3, 2, 1, 1}
        decay_speed = args.ratio
        for i, name in enumerate(decay_order):
            client_nums[name] = round(np.float_power(decay_speed, len(decay_order)-i-1))
            if i==0:
                len_dataset[name] = len(train_sets[name])
            else:
                len_dataset[name] = int(len_dataset[decay_order[i-1]]/decay_speed)
    else:
        decay_speed = 3
        # data_len = [2, 2, 2, 2, 2, 2]
        min_len = -1
        for _, train_set in train_sets.items():
            if min_len==-1:
                min_len = len(train_set)
            else:
                min_len = min(min_len, len(train_set))
        for i, name in enumerate(decay_order):
            client_nums[name] = 2
            len_dataset[name] = min_len
            
    print(client_nums)
    for name, val in len_dataset.items():
        print(f"{name}: {val * args.percent * 0.6}")

    target_loader = None
    client_weights = []
    if args.dg:
        print(f"target domain is {args.target_domain}")
        client_nums[args.target_domain] = 0
        target_loader = torch.utils.data.DataLoader(test_sets[args.target_domain], batch_size=args.batch, shuffle=False)

    train_loaders, val_loaders, test_loaders = [], [], []
    datasets = []
    sum_len = 0

    for key, value in client_nums.items():
        all_len = len_dataset[key] * args.percent
        all_train_len = int(all_len * 0.6)
        all_val_len = int(all_len * 0.4)
        cur_dataset_len = len_dataset[key]
        train_begin = 0
        valid_begin = -all_val_len
        partition_num = (np.float_power(decay_speed, value)-1) / (decay_speed - 1)
        
        test_loader = torch.utils.data.DataLoader(test_sets[key], batch_size=1, shuffle=False)
        for j in range(value):
            train_len = int(all_train_len * np.float_power(decay_speed, j) / partition_num)
            val_len = int(all_val_len * np.float_power(decay_speed, j) / partition_num)
            datasets.append(key)
            cur_trainset = torch.utils.data.Subset(train_sets[key], list(range(all_train_len))[train_begin : train_begin+train_len])
            cur_valset = torch.utils.data.Subset(train_sets[key], list(range(cur_dataset_len))[-valid_begin : -valid_begin+val_len])
            train_loader = torch.utils.data.DataLoader(cur_trainset, batch_size=args.batch, shuffle=True)
            val_loader = torch.utils.data.DataLoader(cur_valset, batch_size=args.batch, shuffle=False)
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
            test_loaders.append(test_loader)
            train_begin += train_len
            valid_begin += val_len
            client_weights.append(len(cur_trainset))
            sum_len += len(cur_trainset)
            # print(len(cur_trainset), len(cur_valset))
    print(client_weights)
    write_log(args, f"data_number=[")
    for ni in client_weights:
        write_log(args, f"{ni},")
    write_log(args, f"\nclient_nums=[")
    for name in decay_order:
        write_log(args, f"{client_nums[name]},")
    write_log(args, f"]\nlen_dataset=[")
    for name in decay_order:
        write_log(args, f"{len_dataset[name]},")
    write_log(args, f"]\n")
    client_weights = [ci/sum_len for ci in client_weights]
    check_labels(args, train_loaders)
    return client_weights, sum_len, train_loaders, val_loaders, test_loaders, datasets, target_loader

def prepare_fairface_iid_uneven(args):
    data_base_path = '../../data/FairFace'
    s = 1
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transform_train = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])
    if args.color_jitter:
        transform_train = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
        ])
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
    ])
    len_dataset = {}
    client_nums = {}
    train_sets = {}
    test_sets = {}
    decay_order = ['White', 'Latino_Hispanic', 'Black', 'East_Asian', 'Indian', 'Southeast_Asian', 'Middle_Eastern']
    if args.binary_race:
        decay_order = ['White', 'Black']
    for name in decay_order:
        train_sets[name] = FairFaceIIDDataset(args, data_base_path, name, transform=transform_train)
        test_sets[name] = FairFaceIIDDataset(args, data_base_path, name, transform=transform_test, train=False)
                
    if 'uneven' in args.expname.lower():
        # client number = 1.4^k, k=0~5
        # data_len = {5, 4, 3, 2, 1, 1}
        decay_speed = args.ratio
        if args.binary_race:
            client_nums = {"White": 10, "Black": 2}
            len_dataset[decay_order[0]] = len(train_sets[decay_order[0]])
            len_dataset[decay_order[1]] = int(len(train_sets[decay_order[0]]) * (client_nums[decay_order[1]] / client_nums[decay_order[0]]))
        else:
            for i, name in enumerate(decay_order):
                client_nums[name] = round(np.float_power(decay_speed, len(decay_order)-i-1))
                if i==0: 
                    len_dataset[name] = len(train_sets[name])
                else:
                    len_dataset[name] = int(len_dataset[decay_order[i-1]]/decay_speed)
    else:
        decay_speed = 3
        min_len = -1
        for _, train_set in train_sets.items():
            if min_len==-1:
                min_len = len(train_set)
            else:
                min_len = min(min_len, len(train_set))
        for i, name in enumerate(decay_order):
            client_nums[name] = 2
            len_dataset[name] = min_len
            
    print(client_nums)
    for name, val in len_dataset.items():
        print(f"{name}: {val * args.percent * 0.6}")

    target_loader = None
    client_weights = []
    if args.dg:
        print(f"target domain is {args.target_domain}")
        client_nums[args.target_domain] = 0
        target_loader = torch.utils.data.DataLoader(test_sets[args.target_domain], batch_size=args.batch, shuffle=False)

    train_loaders, val_loaders, test_loaders = [], [], []
    datasets = []
    sum_len = 0

    for key, value in client_nums.items():
        all_len = len_dataset[key] * args.percent
        all_train_len = int(all_len * 0.6)
        all_val_len = int(all_len * 0.4)
        cur_dataset_len = len_dataset[key]
        train_begin = 0
        valid_begin = -all_val_len
        partition_num = (np.float_power(decay_speed, value)-1) / (decay_speed - 1)
        
        test_loader = torch.utils.data.DataLoader(test_sets[key], batch_size=1, shuffle=False)
        for j in range(value):
            train_len = int(all_train_len * np.float_power(decay_speed, j) / partition_num)
            val_len = int(all_val_len * np.float_power(decay_speed, j) / partition_num)
            datasets.append(key)
            cur_trainset = torch.utils.data.Subset(train_sets[key], list(range(all_train_len))[train_begin : train_begin+train_len])
            cur_valset = torch.utils.data.Subset(train_sets[key], list(range(cur_dataset_len))[-valid_begin : -valid_begin+val_len])
            train_loader = torch.utils.data.DataLoader(cur_trainset, batch_size=args.batch, shuffle=True)
            val_loader = torch.utils.data.DataLoader(cur_valset, batch_size=args.batch, shuffle=False)
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
            test_loaders.append(test_loader)
            train_begin += train_len
            valid_begin += val_len
            client_weights.append(len(cur_trainset))
            sum_len += len(cur_trainset)
            # print(len(cur_trainset), len(cur_valset))
    print(client_weights)
    write_log(args, f"data_number=[")
    for ni in client_weights:
        write_log(args, f"{ni},")
    write_log(args, f"]\nclient_nums=[")
    for name in decay_order:
        write_log(args, f"{client_nums[name]},")
    write_log(args, f"]\nlen_dataset=[")
    for name in decay_order:
        write_log(args, f"{len_dataset[name]},")
    write_log(args, f"]\n")
    client_weights = [ci/sum_len for ci in client_weights]
    check_labels(args, train_loaders)
    return client_weights, sum_len, train_loaders, val_loaders, test_loaders, datasets, target_loader

def prepare_fairface_gender_uneven(args):
    data_base_path = '../../data/FairFace'
    transform_train = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
    ])
    client_nums = {}
    train_sets = {}
    test_sets = {}
    decay_order = ['White', 'Latino_Hispanic', 'Black', 'East_Asian', 'Indian', 'Southeast_Asian', 'Middle_Eastern']
    if args.small_test:
        decay_order = ['White', 'Black', 'Indian', 'Middle_Eastern']
    elif args.binary_race:
        decay_order = ['White', 'Black']

    if 'uneven' in args.expname.lower():
        # client number = 1.4^k, k=0~5
        # data_len = {5, 4, 3, 2, 1, 1}
        decay_speed = args.ratio
        max_clientnum = round(np.float_power(decay_speed, len(decay_order)-1))
        distribution_mode = f"imbalance{max_clientnum}_{args.gender_dis}"
        if args.small_test: distribution_mode += '_small'
        if args.binary_race:
            client_nums = {"White": 10, "Black": 2}
        else:
            for i, name in enumerate(decay_order):
                client_nums[name] = round(np.float_power(decay_speed, len(decay_order)-i-1))
    else:
        decay_speed = 3
        distribution_mode = f"balance_{args.gender_dis}"
        if args.small_test: distribution_mode += '_small'
        for i, name in enumerate(decay_order):
            client_nums[name] = 2
    print(client_nums)
    client_weights = []
    train_loaders, val_loaders, test_loaders = [], [], []
    datasets = []
    sum_len = 0
    genders = ['Male', 'Female']
    gidx = 0
    for name, value in client_nums.items():
        for j in range(value):
            train_set = FairFaceGenderDataset(distribution_mode, data_base_path, name, j, transform=transform_train)
            all_len = int(len(train_set) * args.percent)
            train_len = int(all_len * 0.6)
            val_len = int(all_len * 0.4)
            datasets.append(f'{name}_{genders[gidx]}')
            gidx = int(not gidx)
            val_set = torch.utils.data.Subset(train_set, list(range(all_len))[-val_len :])
            train_set = torch.utils.data.Subset(train_set, list(range(all_len))[: train_len])
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch, shuffle=False)
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
            test_set = FairFaceGenderDataset(distribution_mode, data_base_path, name, gidx, transform=transform_test, train=False)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
            test_loaders.append(test_loader)
            client_weights.append(len(train_set))
            sum_len += len(train_set)
            # print(len(cur_trainset), len(cur_valset))
    print(client_weights)
    write_log(args, f"data_number=[")
    for ni in client_weights:
        write_log(args, f"{ni},")
    write_log(args, f"]\nclient_nums=[")
    for name in decay_order:
        write_log(args, f"{client_nums[name]},")
    write_log(args, f"]\n")
    client_weights = [ci/sum_len for ci in client_weights]
    # check_labels(args, train_loaders)
    return client_weights, sum_len, train_loaders, val_loaders, test_loaders, datasets, None

def prepare_data(args):
    if args.dataset.lower()[:6] == 'domain':
        return prepare_domainnet_uneven(args)
    elif args.dataset.lower()[:5] == 'digit':
        return prepare_digit_uneven(args)
    elif args.dataset.lower()[:9] == 'fairface':
        if args.gender_dis in ['iid', 'random_dis']:
            return prepare_fairface_iid_uneven(args)
        else:
            return prepare_fairface_gender_uneven(args)


def check_labels(args, train_loaders):
    client_num = len(train_loaders)
    label_set = [set() for _ in range(client_num)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write_log(args, 'label set of clients: [')
    for i, train_loader in enumerate(train_loaders):
        train_iter = iter(train_loader)
        # for step in range(len(train_iter)):
        for step in tqdm(range(len(train_iter))):
            _, target = next(train_iter)
            target = target.to(device).long()
            for j in range(len(target)):
                label_set[i].add(target[j].item())
        print(len(label_set[i]))
        write_log(args, f'{len(label_set[i])}, ')
    write_log(args, ']\n')

def norm_grad_diff(weights_before, new_weights, lr):
    # input: nested gradients
    # output: square of the L-2 norm
    norms = 0
    for key in weights_before.state_dict().keys():
        if  'prompt' in key or 'classifier' in key or 'meta_net' in key:
            temp = (weights_before.state_dict()[key] - new_weights.state_dict()[key]) * 1.0 / lr
            norms += torch.norm(temp, p=2).cpu()
    return norms

def train(model, train_loader, optimizer, loss_fun, device):
    model.to(device)
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in tqdm(range(len(train_iter))):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        # optimizer.first_step(zero_grad=True)
        # loss_fun(model(x), y).backward()
        # optimizer.second_step(zero_grad=True)

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    model.to('cpu')
    return loss_all/len(train_iter), correct/num_data

def train_doprompt(args, model, train_loader, client_idx, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    # for step in range(len(train_iter)):
    for step in tqdm(range(len(train_iter))):
        x, y = next(train_iter)
        x = x.to(device).float()
        y = y.to(device).long()
        num_data += y.size(0)
        result = model.update_doprompt(x, y, client_idx, device)

        loss_all += result['loss']
        correct += result['correct']

    return loss_all/len(train_iter), correct/num_data

def train_fedprompt(gidx, model, train_loader, prompt_bank, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    # for step in range(len(train_iter)):
    for step in tqdm(range(len(train_iter))):
        x, y = next(train_iter)
        x = x.to(device).float()
        y = y.to(device).long()
        num_data += y.size(0)
        result = model.update(x, y, prompt_bank, gidx, device)

        loss_all += result['loss']
        correct += result['correct']

    return loss_all/len(train_iter), correct/num_data

def train_CoCoOP(args, model, train_loader, loss_fun, device):
    model.to(device)
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in tqdm(range(len(train_iter))):
        x, y = next(train_iter)
        x = x.to(device).float()
        y = y.to(device).long()
        num_data += y.size(0)
        result = model.update(loss_fun, x, y)
        loss_all += result['loss']
        correct += result['correct']
    model.to('cpu')
    return loss_all/len(train_iter), correct/num_data

import torch.nn.functional as F
def train_sam(model, train_loader, prompt_opt, project_opt, optimizer, loss_fun, device):
    model.to(device)
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    # for step in range(len(train_iter)):
    for step in tqdm(range(len(train_iter))):
        x, y = next(train_iter)
        x = x.to(device).float()
        y = y.to(device).long()
        num_data += y.size(0)
        # prompt
        all_logit = model.forward_prompt(x)
        loss_p = loss_fun(all_logit, y)
        loss_p.backward()
        prompt_opt.first_step(zero_grad=True)
        loss_fun(model.forward_prompt(x), y).backward()
        prompt_opt.second_step(zero_grad=True)
        # meta net
        model.network.eval()
        hint = model.forward_raw(x)
        model.network.train()
        logit = model.forward_proj(x, hint)
        loss_m = loss_fun(logit, y)      
        loss_m.backward()
        optimizer.first_step(zero_grad=True)
        project_opt.first_step(zero_grad=True)
        loss_fun(model.forward_proj(x, hint), y).backward()
        optimizer.second_step(zero_grad=True)
        project_opt.second_step(zero_grad=True)

        loss_all += (loss_m+loss_p).item()
        pred = logit.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    model.to('cpu')
    return loss_all/len(train_iter), correct/num_data

def train_harmofl(args, model, data_loader, optimizer, loss_fun, device):
    model.to(device)
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    train_acc = 0.

    for step, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()

        total += target.size(0)
        pred = output.data.max(1)[1]
        batch_correct = pred.eq(target.view(-1)).sum().item()
        correct += batch_correct
        if step % math.ceil(len(data_loader)*0.2) == 0:
            print(' [Step-{}|{}]| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(step, len(data_loader), loss.item(), batch_correct/target.size(0)), end='\r')

        loss.backward()
        optimizer.generate_delta(zero_grad=True)
        loss_fun(model(data), target).backward()
        optimizer.step(zero_grad=True)
        # optimizer.step()

    loss = loss_all / len(data_loader)
    acc = correct/total
    model.to('cpu')
    return loss, acc

def train_fedprox(args, server_model, model, train_loader, optimizer, loss_fun, device):
    model.to(device)
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    model.to('cpu')
    return loss_all/len(train_iter), correct/num_data

def test(model, test_loader, loss_fun, device, prompt_bank=None):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in tqdm(test_loader):
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        if prompt_bank is not None:
            output = model(data, prompt_bank)
        else:
            output = model(data)

        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
    model.to('cpu')
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

def is_personalized_param(name):
    if 'prompt' in name or 'head' in name or 'adapter' in name or 'bn' in name:
        return True
    else:
        return False

def get_domain_idx(pi, prompt_bank):
    #4*768
    reshape_pi = torch.flatten(pi).to('cuda')
    #6, (4*768)
    reshape_pompt_bank = torch.reshape(prompt_bank, (prompt_bank.shape[0], 4*768))
    domain_sim = torch.matmul(reshape_pompt_bank, reshape_pi).to(prompt_bank.device)

    return torch.argmax(domain_sim)

def agg_rep(args, model, test_loader, device):
    model.to(device)
    model.eval()
    agg_protos_label = {}
    cnt = {}
    mid = 3
    for idx, batch in enumerate(tqdm(test_loader)):
        if idx==mid: break
        data, target = batch
        data = data.to(device).float()
        target = target.to(device).long()
        if 'raw' in args.expname.lower():
            features = model.forward_raw(data)
        else:
            features = model.forward_feat(data)
        for i in range(len(target)):
            if target[i].item() in agg_protos_label:
                agg_protos_label[target[i].item()] += features[i]
                cnt[target[i].item()] += 1
            else:
                agg_protos_label[target[i].item()] = features[i]
                cnt[target[i].item()] = 1

    agg_protos = {}
    rep_list = []
    for [label, proto] in agg_protos_label.items():
        agg_protos[label] = proto / cnt[label]
        rep_list.append(agg_protos[label])
    all_rep = torch.stack(rep_list)
    avg_rep = torch.mean(all_rep, dim=0).data
    model.to('cpu')
    return avg_rep

from sklearn.cluster import AgglomerativeClustering as Agg
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
def cluster(args, all_pi, domain_num):
    all_pi_reshape = all_pi.cpu().reshape(all_pi.shape[0], -1)
    print(all_pi_reshape.shape)
    cluster = GMM(n_components=domain_num)
    # cluster = KMeans(n_clusters=domain_num, random_state=0)
    # cluster = SpectralClustering(n_clusters=domain_num,
    #      assign_labels='discretize',
    #      random_state=0)
    labels = cluster.fit_predict(all_pi_reshape)
    gmap = {}
    cnt = [0 for _ in range(domain_num)]
    for cidx, gidx in enumerate(labels):
        gmap[cidx] = gidx
        cnt[gidx] += 1
    write_log(args, f'cnt=[')
    for didx in range(domain_num):
        write_log(args, f'{cnt[didx]}, ')
    write_log(args, f']\n')
    print(cnt)
    return gmap, cnt

def agg_cluster(all_pi, prompt_bank):
    all_pi_reshape = all_pi.cpu().reshape(all_pi.shape[0], -1)
    print(all_pi_reshape.shape)
    # cluster = Agg(n_clusters=prompt_bank.shape[0]).fit(all_pi_reshape)
    # labels = cluster.labels_
    # cluster = KMeans(n_clusters=prompt_bank.shape[0], random_state=0)
    cluster = SpectralClustering(n_clusters=prompt_bank.shape[0],
         assign_labels='discretize',
         random_state=0)
    labels = cluster.fit_predict(all_pi_reshape)
    gmap = {}
    cnt = [0 for _ in range(prompt_bank.shape[0])]
    temp = torch.zeros_like(prompt_bank, dtype=torch.float32).cuda()
    for cidx, gidx in enumerate(labels):
        gmap[cidx] = gidx
        temp[gidx] += all_pi[cidx]
        cnt[gidx] += 1
    for gidx in range(prompt_bank.shape[0]):
        temp[gidx] /= cnt[gidx]
        if cnt[gidx]>0:
            prompt_bank[gidx].data.copy_(temp[gidx])
    return prompt_bank, gmap, cnt

def remap(all_pi, prompt_bank):
    # print(prompt_bank[0][0][0])
    # print(all_pi.shape, prompt_bank.shape)
    mi, stdi = torch.mean(all_pi, dim=0, keepdim=False), torch.std(all_pi, dim=0, keepdim=False)
    mb, stdb = torch.mean(prompt_bank, dim=0, keepdim=False), torch.std(prompt_bank, dim=0, keepdim=False)
    mi, stdi = mi.detach(), stdi.detach()
    mb, stdb = mb.detach(), stdb.detach()
    if torch.sum(stdb)==0:
        print(f"std is zero!! {torch.sum(stdb)}")
        return random_replace(all_pi, prompt_bank)
        # prompt_bank, _, _ = agg_cluster(all_pi, prompt_bank)
        # return prompt_bank
    else:
        print(f"std not zero: {torch.sum(stdb)}")
    prompt_bank = (mi + stdi * (prompt_bank - mb)/stdb)
    return prompt_bank

def random_replace(all_pi, prompt_bank):
    perm = torch.randperm(all_pi.size(0))[:prompt_bank.shape[0]]
    return all_pi[perm].detach().clone()

def domain_fairness(args, gmap, train_losses):
    domain_losses = {}
    for cidx, gidx in gmap.items():
        domain_losses[gidx] = list()
    for cidx, loss in enumerate(train_losses):
        gidx = gmap[cidx]
        domain_losses[gidx].append(loss)
    for gidx in domain_losses:
        mean = sum(domain_losses[gidx]) / len(domain_losses[gidx])
        domain_losses[gidx] = mean
    domain_losses = list(domain_losses.values())
    std = np.std(domain_losses, keepdims=False)
    print(f"domain fairness std = {std}")
    write_log(args, f"domain fairness std = {std}\n")
    return std

################# Key Function ########################
def communication(args, group_cnt, server_model, models, client_weights, sum_len, client_num, domain_num, Eas, train_losses, a_iter, all_feat=None, prompt_bank=None):
    gmap = {}
    alpha = 0.99
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode.lower() != 'fedprompt':
        prompt_bank = None
    for model in models:
        model.to(device)
    server_model.to(device)
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if  not is_personalized_param(key):
                # if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'fedper':
            for key in server_model.state_dict().keys():
                if  not is_personalized_param(key):
                # if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'doprompt':
            print(client_num, len(models))
            for key in server_model.state_dict().keys():
                if  'prompt' not in key and 'featurizer' not in key:
                # if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
                elif 'prompt' in key:
                    # todo: don't avg on the prompt of each domain!
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp[client_idx].data.copy_(models[client_idx].state_dict()[key][client_idx])
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
                    print(key)
        elif args.mode.lower() == 'harmo-fl':
            for model in models:
                model.to(device)

            for key in server_model.state_dict().keys():
                if not ('prompt' in key or 'head' in key or 'running_amp' in key):
                    continue
                print(key)
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
                if 'running_amp' in key:
                    # aggregate at first round only to save communication cost
                    server_model.amp_norm.fix_amp = True
                    for model in models:
                        model.amp_norm.fix_amp = True
        elif args.mode.lower() == 'cocoop':
            for key in server_model.state_dict().keys():
                if  'prompt' in key or 'classifier' in key or 'meta_net' in key:
                    print(key)
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'q-ffl':
            lr = 0.001
            hs = []
            q = 1.0
            for client_idx in range(client_num):
                loss = train_losses[client_idx] # compute loss on the whole training data, with respect to the starting point (the global model)
                new_weights = models[client_idx]
                temp = copy.deepcopy(server_model).to(device)
                # estimation of the local Lipchitz constant
                hs.append(q * np.float_power(loss+1e-10, (q-1)) * norm_grad_diff(temp, new_weights, lr) + (1.0/lr) * np.float_power(loss+1e-10, q))

            # aggregate using the dynamic step-size
            demominator = np.sum(np.asarray(hs))
            write_log(args, f'denominator: {demominator}\n')
            for key in server_model.state_dict().keys():
                if  'prompt' in key or 'head' in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    delta = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        grad_diff = (server_model.state_dict()[key] - models[client_idx].state_dict()[key]) * 1.0 / lr
                        delta += np.float_power(loss+1e-10, q) * grad_diff
                    temp = server_model.state_dict()[key] - delta / demominator
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'drfl':
            multi = 100
            q = 1
            all_w = 0
            new_weights = [0 for _ in range(client_num)]
            for client_idx in range(client_num):
                weight = multi / (Eas[client_idx])
                weight = client_weights[client_idx] * np.float_power(weight+1e-10, (q+1))
                new_weights[client_idx] = weight
                all_w += weight
            new_weights = [w/all_w for w in new_weights]
            print(new_weights)
            for key in server_model.state_dict().keys():
                if  'prompt' in key or 'head' in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        weight = new_weights[client_idx]
                        temp += weight * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'ccop':
            multi = 100
            
            q = args.q
            gmap, cnt = cluster(args, all_feat, domain_num)
            gsize = [0 for _ in range(domain_num)]
            gloss = [1e-10 for _ in range(domain_num)]
            for i in range(client_num):
                gsize[gmap[i]] += client_weights[i]
                loss = multi / (Eas[i])
                gloss[gmap[i]] += loss * client_weights[i]
            for i in range(domain_num):
                if gsize[i]>0:
                    gloss[i] /= gsize[i]
            all_weight = 0
            new_weights = [0 for _ in range(client_num)]
            power_decay = 0.9
            base = 0.5
            powerI = base + (1-base) * np.float_power(power_decay, a_iter+1)
            powerC = 1 - powerI
            print("========")
            print(gloss)
            write_log(args, f"power_decay: {power_decay}, powerI: {powerI:.4f}, powerC: {powerC:.4f}\n")
            print("========")
            losses = []
            for client_idx in range(client_num):
                losses.append(multi / (Eas[client_idx]))
            if args.std_rw:
                domain_std = domain_fairness(args, gmap, losses)
            else:
                domain_std = 0
            loss_i, loss_c = [], []
            for client_idx in range(client_num):
                loss = multi / (Eas[client_idx])
                Li  = loss
                Lc = gloss[gmap[client_idx]]
                loss_i.append(Li)
                loss_c.append(Lc)
                Lrb = np.float_power(Li, powerI) * np.float_power(Lc, powerC) + 1e-10
                Srb = client_weights[client_idx] / (1 + domain_std)
                weight = Srb * np.float_power(Lrb, (q+1))
                new_weights[client_idx] = weight
            if args.quan > 0:
                quan_i = np.quantile(np.asarray(loss_i), args.quan)
                quan_c = np.quantile(np.asarray(loss_c), args.quan)
                print(loss_i)
                print(quan_i)
                min_w = min(new_weights)
                write_log(args, f"min_w={min_w:.5f}, quan_i={quan_i:.2f}, quan_c={quan_c:.2f}\n")
                write_log(args, f"minimize weight of clients:[")
                for client_idx in range(client_num):
                    if loss_i[client_idx] < quan_i and loss_c[client_idx] < quan_c:
                        write_log(args, f"{client_idx}, ")
                        new_weights[client_idx] = min_w
                write_log(args, f"]\n")
            all_weight = sum(new_weights) 
            new_weights = [wi/all_weight for wi in new_weights]
            for key in server_model.state_dict().keys():
                if  'prompt' in key or 'classifier' in key or 'meta_net' in key:
                    print(key)
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        weight = new_weights[client_idx]
                        temp += weight * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'ablation':
            multi = 100
            q = args.q
            gmap, cnt = cluster(args, all_feat, domain_num)
            gsize = [0 for _ in range(domain_num)]
            gloss = [1e-10 for _ in range(domain_num)]
            for i in range(client_num):
                gsize[gmap[i]] += client_weights[i]
                loss = multi / (Eas[i])
                gloss[gmap[i]] += loss * client_weights[i]
            for i in range(domain_num):
                if gsize[i]>0:
                    gloss[i] /= gsize[i]
            all_weight = 0
            new_weights = [0 for _ in range(client_num)]
            power_decay = 0.9
            base = 0.5
            powerI = base + (1-base) * np.float_power(power_decay, a_iter+1)
            powerC = 1 - powerI
            print("========")
            print(gloss)
            write_log(args, f"power_decay: {power_decay}, powerI: {powerI:.4f}, powerC: {powerC:.4f}\n")
            print("========")

            for client_idx in range(client_num):
                loss = multi / (Eas[client_idx])
                Li  = loss
                Lc = gloss[gmap[client_idx]]
                Lrb = np.float_power(Li, powerI) * np.float_power(Lc, powerC) + 1e-10
                Srb = client_weights[client_idx]
                weight = Srb * np.float_power(Lrb, (q+1))
                new_weights[client_idx] = weight
                all_weight += weight
            new_weights = [wi/all_weight for wi in new_weights]
            write_log(args, f"sum of wi : {sum(new_weights):.4f}\n")
            for key in server_model.state_dict().keys():
                if  'prompt' in key or 'head' in key:
                    print(key)
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        weight = new_weights[client_idx]
                        temp += weight * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'fedprompt':
            for key in server_model.state_dict().keys():
                if  'prompt' not in key:
                # if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
                else:
                    print(key)
                    cnt = [0 for i in range(domain_num)]
                    all_pi = None
                    if group_cnt == 0:
                        for client_idx in range(client_num):
                            pi = models[client_idx].state_dict()[key].unsqueeze(0)
                            if client_idx == 0:
                                all_pi = pi
                            else:
                                all_pi = torch.concat((all_pi, pi))
                        prompt_bank = remap(all_pi, prompt_bank)
                    temp = torch.zeros_like(prompt_bank, dtype=torch.float32)
                    for client_idx in range(client_num):
                        didx = get_domain_idx(models[client_idx].state_dict()[key], prompt_bank)
                        gmap[client_idx] = didx
                        cnt[didx] += 1
                        temp[didx] += models[client_idx].state_dict()[key]
                    # # todo : EMA replace prompt
                    print(cnt)
                    write_log(args, f'clients=[')
                    for i in range(domain_num):
                        write_log(args, f'{cnt[i], }')
                        if cnt[i]>0:
                            temp[i] /= cnt[i]
                            prompt_bank[i].data.copy_(prompt_bank[i].data*(1-alpha) + alpha*temp[i])
                    write_log(args, f']\n')
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    if ('prompt' in key or 'head' in key) or 'full' in args.mode.lower():
                        print(key)
                        temp = torch.zeros_like(server_model.state_dict()[key])
                        for client_idx in range(len(client_weights)):
                            temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    for model in models:
        model.to('cpu')
    return server_model, models, prompt_bank, gmap