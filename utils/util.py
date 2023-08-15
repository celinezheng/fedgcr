from tqdm import tqdm
import sys, os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from utils.data_utils import DomainNetDataset, DigitsDataset, PACSDataset
import math
import torch.nn.functional as tf
import numpy as np
import copy
from utils.project import project, solve_centered_w
from utils.aggregate import aggregate, aggregate_lr, sum_models, zero_model, \
                            assign_models, avg_models, add_models, scale_model, sub_models, norm2_model
import torch.autograd as autograd
import torch.nn.functional as F

def write_log(args, msg):
    log_path = f'../logs/{args.dataset}_{args.expname}_{args.ratio}_{args.seed}'
    log_fname = f'{args.mode}'
    
    log_path += f"_{args.cluster_num}"
    if args.shuffle:  log_path += f"_shuffle"
    if args.moon:  log_path += f"_moon"
    if args.distinct:  log_path += f"_distinct"
    
    if args.q!=1: log_fname += f'_q={args.q}'
    if args.split_test:  log_fname += f"_split"
    if args.w_con!=0.1: log_fname += f'_w_con={args.w_con}'
    if args.clscon:  log_fname += f"_clscon"
    if 'gmm' not in args.cluster: log_fname += f"_{args.cluster}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(os.path.join(log_path, f'{log_fname}.log'), 'a') as logfile:
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
    mnist_testset      = DigitsDataset(split_test=args.split_test, data_path=os.path.join(base_path, "MNIST"), channels=1, percent=args.percent, train=False, transform=transform_mnist)
    test_len = int(args.percent * len(mnist_testset))
    mnist_testset = torch.utils.data.Subset(mnist_testset, list(range(len(mnist_testset)))[:test_len])

    # SVHN
    svhn_trainset      = DigitsDataset(data_path=os.path.join(base_path, "SVHN"), channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
    svhn_testset       = DigitsDataset(split_test=args.split_test, data_path=os.path.join(base_path, "SVHN"), channels=3, percent=args.percent,  train=False, transform=transform_svhn)
    test_len = int(args.percent * len(svhn_testset))
    svhn_testset = torch.utils.data.Subset(svhn_testset, list(range(len(svhn_testset)))[:test_len])

    # USPS
    usps_trainset      = DigitsDataset(data_path=os.path.join(base_path, "USPS"), channels=1, percent=args.percent,  train=True,  transform=transform_usps)
    usps_testset       = DigitsDataset(split_test=args.split_test, data_path=os.path.join(base_path, "USPS"), channels=1, percent=args.percent,  train=False, transform=transform_usps)
    test_len = int(args.percent * len(usps_testset))
    usps_testset = torch.utils.data.Subset(usps_testset, list(range(len(usps_testset)))[:test_len])

    # Synth Digits
    synth_trainset     = DigitsDataset(data_path=os.path.join(base_path, "SynthDigits"), channels=3, percent=args.percent,  train=True,  transform=transform_synth)
    synth_testset      = DigitsDataset(split_test=args.split_test, data_path=os.path.join(base_path, "SynthDigits"), channels=3, percent=args.percent,  train=False, transform=transform_synth)
    test_len = int(args.percent * len(synth_testset))
    synth_testset = torch.utils.data.Subset(synth_testset, list(range(len(synth_testset)))[:test_len])

    # MNIST-M
    mnistm_trainset     = DigitsDataset(data_path=os.path.join(base_path, "MNIST_M"), channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
    mnistm_testset      = DigitsDataset(split_test=args.split_test, data_path=os.path.join(base_path, "MNIST_M"), channels=3, percent=args.percent,  train=False, transform=transform_mnistm)
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
    len_dataset = {}
    client_nums = {}
    if args.shuffle:
        decay_order = ['USPS', 'MNIST', 'SynthDigits', 'SVHN', 'MNIST-M']
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
    
    client_weights = []
    sum_len = 0
    print(client_nums)
    train_loaders, val_loaders, test_loaders = [], [], []
    datasets = []
    train_ratio = 0.8
    if args.distinct:
        for i, name in enumerate(decay_order):
            client_nums[name] = 1

    for key, value in client_nums.items():
        all_len = len_dataset[key] * args.percent
        all_train_len = int(all_len * 0.6)
        all_val_len = int(all_len * 0.4)
        all_test_len = len(test_sets[key])
        cur_dataset_len = len_dataset[key]
        train_begin = 0
        test_begin = 0
        valid_begin = -all_val_len
        if args.distinct:
            partition_num = 1
        else:
            partition_num = (np.float_power(decay_speed, value)-1) / (decay_speed - 1)
        
        if not args.split_test:
            test_loader = torch.utils.data.DataLoader(test_sets[key], batch_size=1, shuffle=False)
        for j in range(value):
            train_len = int(all_train_len * np.float_power(decay_speed, j) / partition_num)
            val_len = int(all_val_len * np.float_power(decay_speed, j) / partition_num)
            if args.split_test:
                datasets.append(f'{key}-{j}')
                test_len = int(all_test_len * np.float_power(decay_speed, j) / partition_num)
                cur_testset = torch.utils.data.Subset(test_sets[key], list(range(all_test_len))[test_begin : test_begin+test_len])
                test_loader = torch.utils.data.DataLoader(cur_testset, batch_size=1, shuffle=False)
                test_begin += test_len
            else:
                datasets.append(key)
            test_loaders.append(test_loader)
            cur_trainset = torch.utils.data.Subset(train_sets[key], list(range(all_train_len))[train_begin : train_begin+train_len])
            cur_valset = torch.utils.data.Subset(train_sets[key], list(range(cur_dataset_len))[-valid_begin : -valid_begin+val_len])
            
            train_loader = torch.utils.data.DataLoader(cur_trainset, batch_size=args.batch, shuffle=True)
            val_loader = torch.utils.data.DataLoader(cur_valset, batch_size=args.batch, shuffle=False)
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
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
    # check_labels(args, train_loaders)
    return client_weights, sum_len, train_loaders, val_loaders, test_loaders, datasets

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
    clipart_testset = DomainNetDataset(split_test=args.split_test, base_path=data_base_path, site='clipart', transform=transform_test, train=False)
    # infograph
    infograph_trainset = DomainNetDataset(data_base_path, 'infograph', transform=transform_train)
    infograph_testset = DomainNetDataset(split_test=args.split_test, base_path=data_base_path, site='infograph', transform=transform_test, train=False)
    # painting
    painting_trainset = DomainNetDataset(data_base_path, 'painting', transform=transform_train)
    painting_testset = DomainNetDataset(split_test=args.split_test, base_path=data_base_path, site='painting', transform=transform_test, train=False)
    # quickdraw
    quickdraw_trainset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_train)
    quickdraw_testset = DomainNetDataset(split_test=args.split_test, base_path=data_base_path, site='quickdraw', transform=transform_test, train=False)
    # real
    real_trainset = DomainNetDataset(data_base_path, 'real', transform=transform_train)
    real_testset = DomainNetDataset(split_test=args.split_test, base_path=data_base_path, site='real', transform=transform_test, train=False)
    # sketch
    sketch_trainset = DomainNetDataset(data_base_path, 'sketch', transform=transform_train)
    sketch_testset = DomainNetDataset(split_test=args.split_test, base_path=data_base_path, site='sketch', transform=transform_test, train=False)

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
    if args.shuffle:
        decay_order = ['Real', 'Clipart', 'Painting', 'Sketch', 'Infograph', 'QuickDraw']
    else:
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

    client_weights = []
    
    train_loaders, val_loaders, test_loaders = [], [], []
    datasets = []
    sum_len = 0

    if args.distinct:
        for i, name in enumerate(decay_order):
            client_nums[name] = 1

    for key, value in client_nums.items():
        all_len = len_dataset[key] * args.percent
        all_train_len = int(all_len * 0.6)
        all_val_len = int(all_len * 0.4)
        all_test_len = len(test_sets[key])
        cur_dataset_len = len_dataset[key]
        train_begin = 0
        test_begin = 0
        valid_begin = -all_val_len
        if args.distinct:
            partition_num = 1
        else:
            partition_num = (np.float_power(decay_speed, value)-1) / (decay_speed - 1)
        
        if not args.split_test:
            test_loader = torch.utils.data.DataLoader(test_sets[key], batch_size=1, shuffle=False)
        for j in range(value):
            train_len = int(all_train_len * np.float_power(decay_speed, j) / partition_num)
            val_len = int(all_val_len * np.float_power(decay_speed, j) / partition_num)
            if args.split_test:
                datasets.append(f'{key}-{j}')
                test_len = int(all_test_len * np.float_power(decay_speed, j) / partition_num)
                cur_testset = torch.utils.data.Subset(test_sets[key], list(range(all_test_len))[test_begin : test_begin+test_len])
                test_loader = torch.utils.data.DataLoader(cur_testset, batch_size=1, shuffle=False)
                test_begin += test_len
            else:
                datasets.append(key)
            test_loaders.append(test_loader)
            cur_trainset = torch.utils.data.Subset(train_sets[key], list(range(all_train_len))[train_begin : train_begin+train_len])
            cur_valset = torch.utils.data.Subset(train_sets[key], list(range(cur_dataset_len))[-valid_begin : -valid_begin+val_len])
            
            train_loader = torch.utils.data.DataLoader(cur_trainset, batch_size=args.batch, shuffle=True)
            val_loader = torch.utils.data.DataLoader(cur_valset, batch_size=args.batch, shuffle=False)
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
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
    # check_labels(args, train_loaders)
    return client_weights, sum_len, train_loaders, val_loaders, test_loaders, datasets

def prepare_pacs_uneven(args):
    data_base_path = '../../data/PACS'
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

    # photo
    photo_trainset = PACSDataset(data_base_path, 'photo', transform=transform_train)
    photo_testset = PACSDataset(data_base_path, 'photo', transform=transform_test, train=False)
    # art_painting
    art_painting_trainset = PACSDataset(data_base_path, 'art_painting', transform=transform_train)
    art_painting_testset = PACSDataset(data_base_path, 'art_painting', transform=transform_test, train=False)
    # cartoon
    cartoon_trainset = PACSDataset(data_base_path, 'cartoon', transform=transform_train)
    cartoon_testset = PACSDataset(data_base_path, 'cartoon', transform=transform_test, train=False)
    # sketch
    sketch_trainset = PACSDataset(data_base_path, 'sketch', transform=transform_train)
    sketch_testset = PACSDataset(data_base_path, 'sketch', transform=transform_test, train=False)
    
    # min_data_len = int(min(len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset)))
    test_sets = {
        'photo': photo_testset,
        'art_painting': art_painting_testset,
        'cartoon': cartoon_testset,
        'sketch': sketch_testset
        }
    train_sets = {
        'photo': photo_trainset,
        'art_painting': art_painting_trainset,
        'cartoon': cartoon_trainset,
        'sketch': sketch_trainset
        }
    len_dataset = {}
    client_nums = {}
    if args.shuffle:
        decay_order = [
        'photo',
        'art_painting',
        'cartoon',
        'sketch'
        ]
    else:
        decay_order = [
        'photo',
        'art_painting',
        'cartoon',
        'sketch'
        ]

    if 'uneven' in args.expname.lower():
        # client number = 1.4^k, k=0~5
        # data_len = {5, 4, 3, 2, 1, 1}
        decay_speed = args.ratio
        for i, name in enumerate(decay_order):
            client_nums[name] = round(np.float_power(decay_speed, len(decay_order)-i-1))
            if i==0:
                len_dataset[name] = len(train_sets[name])
            else:
                len_dataset[name] = min(len(train_sets[name]), int(len_dataset[decay_order[i-1]]/decay_speed))
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

    client_weights = []
    
    train_loaders, val_loaders, test_loaders = [], [], []
    datasets = []
    sum_len = 0

    if args.distinct:
        for i, name in enumerate(decay_order):
            client_nums[name] = 1

    DIF = client_nums[decay_order[0]] // client_nums[decay_order[-1]]
    print(DIF)
    for key, value in client_nums.items():
        all_len = len_dataset[key] * args.percent
        all_train_len = int(all_len * 0.6)
        all_val_len = int(all_len * 0.4)
        cur_dataset_len = len_dataset[key]
        train_begin = 0
        valid_begin = -all_val_len
        if value==1 or 'uneven' not in args.expname: 
            client_decay = decay_speed
        else:
            client_decay = np.float_power(DIF, 1/(value-1))
        if args.distinct:
            partition_num = 1
        else:
            partition_num = (np.float_power(client_decay, value)-1) / (client_decay - 1)
        
        test_loader = torch.utils.data.DataLoader(test_sets[key], batch_size=1, shuffle=False)
        for j in range(value):
            train_len = int(all_train_len * np.float_power(client_decay, j) / partition_num)
            val_len = int(all_val_len * np.float_power(client_decay, j) / partition_num)
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
    # check_labels(args, train_loaders)
    return client_weights, sum_len, train_loaders, val_loaders, test_loaders, datasets

def prepare_data(args):
    if args.dataset.lower()[:6] == 'domain':
        return prepare_domainnet_uneven(args)
    elif args.dataset.lower()[:4] == 'pacs':
        return prepare_pacs_uneven(args)
    elif args.dataset.lower()[:5] == 'digit':
        return prepare_digit_uneven(args)

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
        if  'prompt' in key or 'head' in key or 'meta_net' in key:
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

def train_fedsam(model, train_loader, optimizer, loss_fun, device):
    model.to(device)
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    # for step in range(len(train_iter)):
    for step in tqdm(range(len(train_iter))):

        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.first_step(zero_grad=True)
        loss_fun(model(x), y).backward()
        optimizer.second_step(zero_grad=True)

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    model.to('cpu')
    return loss_all/len(train_iter), correct/num_data



def train_propfair(model, train_loader, optimizer, loss_fun, device):
    def log_loss(output, target, base=2.0):
        ce_loss = loss_fun(output, target)
        base = torch.tensor(base).to(device)
        epsilon = 0.2
        huber = False
        if base - ce_loss < epsilon:           
            # for the bad performing batches, we enforce a constant to avoid divergence
            if not huber:
                return ce_loss/base
            else:
                return ce_loss/epsilon
        else:
            return -torch.log(1 - ce_loss/base)
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

        loss = log_loss(output, y)
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

def train_GCR(args, model, train_loader, loss_fun, device, gidx=-1, pre_feat=None, cluster_pis=None, pre_pi=None, pre_glob=None, pre_local=None, a_iter=-1):
    model.to(device)
    pre_glob.to(device)
    pre_local.to(device)
    pre_glob.eval()
    pre_local.eval()
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    if cluster_pis is not None:
        cluster_pis = cluster_pis.to(device)
    for step in tqdm(range(len(train_iter))):
        x, y = next(train_iter)
        x = x.to(device).float()
        y = y.to(device).long()
        num_data += y.size(0)
        iter_ratio = a_iter/args.iters
        # if gidx!=-1 and (args.clscon or args.mode.lower()=='only_dcnet'):
        if gidx!=-1:
            result = model.update_clscon(args.w_con, loss_fun, x, y, cluster_pis, pre_pi, gidx, pre_glob=pre_glob, pre_local=pre_local, iter_ratio=iter_ratio)
        else:
            result = model.update(loss_fun, x, y)
        loss_all += result['loss']
        correct += result['correct']
    model.to('cpu')
    pre_glob.to('cpu')
    pre_local.to('cpu')
    return loss_all/len(train_iter), correct/num_data


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
        data = data.to(device).float()
        target = target.to(device).long()
        
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

def train_fedmix(model, Xg, Yg, lamb, train_loader, optimizer, loss_fun, device):
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
        inputX = (1 - lamb) * x
        inputX.requires_grad_()
        idg = torch.randint(len(Xg), (1, ))
        xg = Xg[idg].to(device)[0]
        yg = Yg[idg].to(device)[0]
        output = model(inputX)

        loss1 = (1 - lamb) * loss_fun(output, y)
        loss2 = lamb * loss_fun(output, y)
        loss = loss1 + loss2
        if x.size(0) == xg.size(0):
            gradients = autograd.grad(outputs=loss1, inputs=inputX,
                                            create_graph=True, retain_graph=True)[0]
                                            
            loss3 = lamb * torch.inner(gradients.flatten(start_dim=1), xg.flatten(start_dim=1))
            loss3 = torch.mean(loss3)
            loss += loss3
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

def agg_rep(args, model, test_loader, device, use_dc=False):
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
        if use_dc:
            features = model.get_dc(data)
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

def agg_feat(args, model, test_loader, device):
    model.to(device)
    model.eval()
    all_feat = np.empty((0, 768))  
    y = np.empty(0, dtype=int)
    mid = 800
    idx = 0
    for _, batch in enumerate(tqdm(test_loader)):
        if idx==mid: break
        data, target = batch
        data = data.to(device).float()
        feat = model.forward_feat(data)
        # if target[0].item() != 1: continue
        all_feat = np.concatenate((all_feat, feat.detach().cpu().numpy()), axis=0)
        y = np.concatenate((y, target.detach().cpu().numpy()), axis=0)
        idx += 1
    model.to('cpu')
    return all_feat, y

def agg_dc_tokens(args, model, test_loader, device):
    model.to(device)
    model.eval()
    dc_tokens = np.empty((0, 768))  
    y = np.empty(0, dtype=int)
    mid = 600
    idx = 0
    for _, batch in enumerate(tqdm(test_loader)):
        if idx==mid: break
        data, target = batch
        data = data.to(device).float()
        features = model.get_dc(data)
        # if target[0].item() != 1: continue
        dc_tokens = np.concatenate((dc_tokens, features.detach().cpu().numpy()), axis=0)
        y = np.concatenate((y, target.detach().cpu().numpy()), axis=0)
        idx += 1
    model.to('cpu')
    return dc_tokens, y

from sklearn.cluster import AgglomerativeClustering as Agg
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from utils.finch import FINCH
def cluster(args, all_pi, domain_num):
    all_pi_reshape = all_pi.cpu().reshape(all_pi.shape[0], -1)
    print(all_pi_reshape.shape)
    cluster_num = domain_num
    if args.distinct:
        cluster_num = domain_num - 1
    if 'finch' in args.cluster.lower():
        c, num_clust, req_c = FINCH(all_pi_reshape, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)
        labels = [g[0] for g in c]
        print('========= finch =========')
        print(labels)
        print(max(labels) + 1)
        cluster_num = max(labels) + 1
    elif 'agg' in args.cluster.lower():
        labels = Agg(n_clusters=domain_num).fit_predict(all_pi_reshape)
        print(labels)
    else:
        cluster = GMM(n_components=cluster_num)
        # cluster = KMeans(n_clusters=domain_num, random_state=0)
        # cluster = SpectralClustering(n_clusters=domain_num,
        #      assign_labels='discretize',
        #      random_state=0)
        labels = cluster.fit_predict(all_pi_reshape)

    gmap = {}
    cnt = [0 for _ in range(cluster_num)]
    for cidx, gidx in enumerate(labels):
        gmap[cidx] = gidx
        cnt[gidx] += 1
    write_log(args, f'cnt=[')
    for didx in range(cluster_num):
        write_log(args, f'{cnt[didx]}, ')
    write_log(args, f']\n')
    print(cnt)
    return gmap, cnt

def cluster_avg_feat(args, all_pi, domain_num):
    cluster_num = domain_num
    if args.distinct:
        cluster_num = domain_num - 1
    cluster_feats = torch.zeros((cluster_num, 768), requires_grad=False)
    all_pi_reshape = all_pi.cpu().reshape(all_pi.shape[0], -1)
    gmap={}
    gmap, cnt = cluster(args, all_pi, domain_num)
    for cidx in gmap:
        gidx = gmap[cidx]
        cluster_feats[gidx] += (1/cnt[gidx]) * all_pi_reshape[cidx] 

    return gmap, cnt, cluster_feats

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

################# Key Function ########################
# def communication(args, group_cnt, server_model, models, client_weights, sum_len, client_num, domain_num, Eas, train_losses, a_iter, datasets, pre_clusters, all_feat=None, all_pi=None, prompt_bank=None):
def communication(args, group_cnt, server_model, models, client_weights, sum_len, client_num, domain_num, Eas, train_losses, a_iter, datasets, all_feat=None, all_pi=None, prompt_bank=None):
    gmap = {}
    alpha = 0.99
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cluster_pis = None
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'harmo-fl':
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
        elif args.mode.lower() == 'q-ffl':
            lr = 0.001
            hs = []
            q = args.q
            for client_idx in range(client_num):
                loss = train_losses[client_idx] # compute loss on the whole training data, with respect to the starting point (the global model)
                new_weights = models[client_idx]
                temp = copy.deepcopy(server_model)
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
        elif args.mode.lower() == 'afl':
            y = np.array(client_weights) + 0.1 * np.array(train_losses)
            new_weights = project(y)
            sum_w = sum(new_weights)
            new_weights = [w / sum_w for w in new_weights]
            for key in server_model.state_dict().keys():
                if  'prompt' in key or 'head' in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        weight = new_weights[client_idx]
                        temp += weight * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'term':
            alpha = 0.01
            new_weights = [np.exp(alpha * train_losses[i]) * client_weights[i] \
                for i in range(client_num)]
            new_weights = list(new_weights / np.sum(new_weights))
            for key in server_model.state_dict().keys():
                if  'prompt' in key or 'head' in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        weight = new_weights[client_idx]
                        temp += weight * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() in ['fedgcr', 'ablation', 'only_dcnet']:
            multi = 100
            q = args.q
            if args.mode.lower() in ['only_dcnet', 'fedgcr']:
                gmap, cnt, cluster_pis = cluster_avg_feat(args, all_pi, domain_num)
            else:
                gmap, cnt = cluster(args, all_feat, domain_num)
            new_weights = [wi for wi in client_weights]
            if args.mode.lower() in ['fedgcr', 'ablation']:
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
                power_decay = 0.9
                base = 0.5
                powerI = base + (1-base) * np.float_power(power_decay, a_iter+1)
                powerC = 1 - powerI
                print("========")
                print(gloss)
                write_log(args, f"power_decay: {power_decay}, powerI: {powerI:.4f}, powerC: {powerC:.4f}\n")
                print("========")
                new_weights = [0 for _ in range(client_num)]
                losses = []
                for client_idx in range(client_num):
                    losses.append(multi / (Eas[client_idx]))
                loss_i, loss_c, loss_e = [], [], []
                for client_idx in range(client_num):
                    loss = multi / (Eas[client_idx])
                    Li  = loss
                    Lc = gloss[gmap[client_idx]]
                    Lrb = np.float_power(Li, powerI) * np.float_power(Lc, powerC) + 1e-10
                    loss_i.append(Li)
                    loss_c.append(Lc)
                    loss_e.append(Lrb)
                    Srb = client_weights[client_idx]
                    power = q + 1
                    weight = Srb * np.float_power(Lrb, power)
                    new_weights[client_idx] = weight
                all_weight = sum(new_weights) 
                new_weights = [wi/all_weight for wi in new_weights]
            print(new_weights)
            for key in server_model.state_dict().keys():
                if  'prompt' in key or 'head' in key or 'classifier' in key or 'meta_net' in key :
                    print(key)
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        weight = new_weights[client_idx]
                        temp += weight * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
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
    return server_model, models, prompt_bank, gmap, cluster_pis