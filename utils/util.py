import sys, os
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from utils.data_utils import DomainNetDataset, DigitsDataset, OfficeDataset


def prepare_domainnet(args):
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

    min_data_len = args.percent * min(len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset))
    # val_len = int(min_data_len * 0.1)
    val_len = int(min_data_len * 0.4)
    # min_data_len = int(min_data_len * 0.1)
    min_data_len = int(min_data_len * 0.6)

    clipart_valset   = torch.utils.data.Subset(clipart_trainset, list(range(len(clipart_trainset)))[-val_len:])
    clipart_trainset = torch.utils.data.Subset(clipart_trainset, list(range(min_data_len)))
    
    infograph_valset   = torch.utils.data.Subset(infograph_trainset, list(range(len(infograph_trainset)))[-val_len:])
    infograph_trainset = torch.utils.data.Subset(infograph_trainset, list(range(min_data_len)))
    
    painting_valset   = torch.utils.data.Subset(painting_trainset, list(range(len(painting_trainset)))[-val_len:])
    painting_trainset = torch.utils.data.Subset(painting_trainset, list(range(min_data_len)))

    quickdraw_valset   = torch.utils.data.Subset(quickdraw_trainset, list(range(len(quickdraw_trainset)))[-val_len:])
    quickdraw_trainset = torch.utils.data.Subset(quickdraw_trainset, list(range(min_data_len)))

    real_valset   = torch.utils.data.Subset(real_trainset, list(range(len(real_trainset)))[-val_len:])
    real_trainset = torch.utils.data.Subset(real_trainset, list(range(min_data_len)))

    sketch_valset   = torch.utils.data.Subset(sketch_trainset, list(range(len(sketch_trainset)))[-val_len:])
    sketch_trainset = torch.utils.data.Subset(sketch_trainset, list(range(min_data_len)))

    clipart_train_loader = torch.utils.data.DataLoader(clipart_trainset, batch_size=args.batch, shuffle=True)
    clipart_val_loader   = torch.utils.data.DataLoader(clipart_valset, batch_size=args.batch, shuffle=False)
    clipart_test_loader  = torch.utils.data.DataLoader(clipart_testset, batch_size=args.batch, shuffle=False)

    infograph_train_loader = torch.utils.data.DataLoader(infograph_trainset, batch_size=args.batch, shuffle=True)
    infograph_val_loader = torch.utils.data.DataLoader(infograph_valset, batch_size=args.batch, shuffle=False)
    infograph_test_loader = torch.utils.data.DataLoader(infograph_testset, batch_size=args.batch, shuffle=False)

    painting_train_loader = torch.utils.data.DataLoader(painting_trainset, batch_size=args.batch, shuffle=True)
    painting_val_loader = torch.utils.data.DataLoader(painting_valset, batch_size=args.batch, shuffle=False)
    painting_test_loader = torch.utils.data.DataLoader(painting_testset, batch_size=args.batch, shuffle=False)

    quickdraw_train_loader = torch.utils.data.DataLoader(quickdraw_trainset, batch_size=args.batch, shuffle=True)
    quickdraw_val_loader = torch.utils.data.DataLoader(quickdraw_valset, batch_size=args.batch, shuffle=False)
    quickdraw_test_loader = torch.utils.data.DataLoader(quickdraw_testset, batch_size=args.batch, shuffle=False)

    real_train_loader = torch.utils.data.DataLoader(real_trainset, batch_size=args.batch, shuffle=True)
    real_val_loader = torch.utils.data.DataLoader(real_valset, batch_size=args.batch, shuffle=False)
    real_test_loader = torch.utils.data.DataLoader(real_testset, batch_size=args.batch, shuffle=False)

    sketch_train_loader = torch.utils.data.DataLoader(sketch_trainset, batch_size=args.batch, shuffle=True)
    sketch_val_loader = torch.utils.data.DataLoader(sketch_valset, batch_size=args.batch, shuffle=False)
    sketch_test_loader = torch.utils.data.DataLoader(sketch_testset, batch_size=args.batch, shuffle=False)

    train_loaders = [clipart_train_loader, infograph_train_loader, painting_train_loader, quickdraw_train_loader, real_train_loader, sketch_train_loader]
    val_loaders = [clipart_val_loader, infograph_val_loader, painting_val_loader, quickdraw_val_loader, real_val_loader, sketch_val_loader]
    test_loaders = [clipart_test_loader, infograph_test_loader, painting_test_loader, quickdraw_test_loader, real_test_loader, sketch_test_loader]

    return train_loaders, val_loaders, test_loaders


def prepare_digit(args):
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
    min_data_len = min(len(mnist_trainset), len(svhn_trainset), len(usps_trainset), len(synth_trainset), len(mnistm_trainset))
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.6)
    print(min_data_len)

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

def prepare_office(args):
    data_base_path = '../../data'
    transform_office = transforms.Compose([
            transforms.Resize([224, 224]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),            
            transforms.ToTensor(),
    ])
    
    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office)
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office)
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    min_data_len = args.percent * min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.6)

    amazon_valset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)))[-val_len:]) 
    amazon_trainset = torch.utils.data.Subset(amazon_trainset, list(range(min_data_len)))

    caltech_valset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)))[-val_len:]) 
    caltech_trainset = torch.utils.data.Subset(caltech_trainset, list(range(min_data_len)))

    dslr_valset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)))[-val_len:]) 
    dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))

    webcam_valset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)))[-val_len:]) 
    webcam_trainset = torch.utils.data.Subset(webcam_trainset, list(range(min_data_len)))

    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=args.batch, shuffle=True)
    amazon_val_loader = torch.utils.data.DataLoader(amazon_valset, batch_size=args.batch, shuffle=False)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=args.batch, shuffle=False)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=args.batch, shuffle=True)
    caltech_val_loader = torch.utils.data.DataLoader(caltech_valset, batch_size=args.batch, shuffle=False)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=args.batch, shuffle=False)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=args.batch, shuffle=True)
    dslr_val_loader = torch.utils.data.DataLoader(dslr_valset, batch_size=args.batch, shuffle=False)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=args.batch, shuffle=False)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=args.batch, shuffle=True)
    webcam_val_loader = torch.utils.data.DataLoader(webcam_valset, batch_size=args.batch, shuffle=False)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=args.batch, shuffle=False)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    val_loaders = [amazon_val_loader, caltech_val_loader, dslr_val_loader, webcam_val_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]
    return train_loaders, val_loaders, test_loaders


def prepare_data(args):
    if args.dataset.lower()[:6] == 'domain':
        # name of each datasets
        return prepare_domainnet(args)
    elif args.dataset.lower()[:5] == 'digit':
        return prepare_digit(args)
    elif args.dataset.lower()[:6] == 'office':
        return prepare_office(args)

def train(model, train_loader, optimizer, loss_fun, device):
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

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
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
        if args.lsim:
            result = model.update(x, y, client_idx, device)
        else:
            result = model.update_doprompt(x, y, client_idx, device)

        loss_all += result['loss']
        correct += result['correct']

    return loss_all/len(train_iter), correct/num_data

def train_fedprox(args, server_model, model, train_loader, optimizer, loss_fun, device):
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
    return loss_all/len(train_iter), correct/num_data

def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in tqdm(test_loader):
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output = model(data)
        
        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
    
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

def is_personalized_param(name):
    if 'prompt' in name or 'head' in name or 'adapter' in name or 'bn' in name:
        return True
    else:
        return False    

################# Key Function ########################
def communication(args, server_model, models, client_weights, client_num):
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
        
        elif args.mode.lower() == 'doprompt':
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
                    # todo: don't avg on the prompt of each domain!
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp[client_idx].data.copy_(models[client_idx].state_dict()[key][client_idx])
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
                    print(key)
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    if 'prompt' in key or 'head' in key:
                        print(key)
                        temp = torch.zeros_like(server_model.state_dict()[key])
                        for client_idx in range(len(client_weights)):
                            temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models