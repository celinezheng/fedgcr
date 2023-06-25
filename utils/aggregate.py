'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
''' basic '''
def model_to_params(model):
    return [param.data for param in model.parameters()]

'''one model operation'''

def norm2_model(model):
    params = model_to_params(model)
    sum_ = 0.0
    for param in params:
        sum_ += torch.norm(param) ** 2
    return sum_

def zero_model(model):
    zero = deepcopy(model)
    for i, param in enumerate(zero.parameters()):
        param.data = torch.zeros_like(param.data)
    return zero

def scale_model(model, scale):
    scaled = deepcopy(model)
    for i, param in enumerate(scaled.parameters()):
        model_param = model_to_params(model)[i]
        param.data = scale * model_param.data
    return scaled

'''two model operation'''
def add_models(model1, model2, alpha=1.0):
    # obtain model1 + alpha * model2 for two models of the same size
    addition = deepcopy(model1)
    for i, param_add in enumerate(addition.parameters()):
        param1, param2 = model_to_params(model1)[i], model_to_params(model2)[i] 
        with torch.no_grad():
            param_add.data = param1.data + alpha * param2.data
    return addition

def sub_models(model1, model2):
    # obtain model1 - model2 for two models of the same size
    subtract = deepcopy(model1)
    for i, param_sub in enumerate(subtract.parameters()):
        param1, param2 = model_to_params(model1)[i], model_to_params(model2)[i] 
        with torch.no_grad():
            param_sub.data = param1.data - param2.data
    return subtract

def product_models(model1, model2):
    # obtain model1 - model2 for two models of the same size
    prod = 0.0
    for i, param in enumerate(model1.parameters()):
        param1, param2 = model_to_params(model1)[i], model_to_params(model2)[i] 
        with torch.no_grad():
            #print('param1.data: ', param1.data, 'param2.data: ', param2.data)
            prod += torch.dot(param1.data.view(-1), param2.data.view(-1))
    return prod

def assign_model(model1, model2):
    for i, param1 in enumerate(model1.parameters()):
        param2 = model_to_params(model2)[i] 
        with torch.no_grad():
            param1.data = deepcopy(param2.data)
    return

def assign_models(models, new_model):
    ''' assign the new_model into a list of models'''
    for model in models:
        assign_model(model, new_model)
    return


'''model list operation'''

def avg_models(models, weights=None):
    '''take a list of models and average, weights: a list of numbers summing up to 1'''
    if weights == None:
        total = len(models)
        weights = [1.0/total] * total
    avg = zero_model(models[0])
    for index, model in enumerate(models):
        for i, param in enumerate(avg.parameters()):
            model_param = model_to_params(model)[i]
            param.data += model_param * weights[index]
    return avg

def sum_models(models):
    '''take a list of models and average, weights: a list of numbers summing up to 1'''
    weights = [1.0] * len(models)
    return avg_models(models, weights=weights)


'''aggregation'''
def aggregate(models, weights=None):   # FedAvg
    avg = avg_models(models, weights=weights)
    assign_models(models, avg)
    return

def aggregate_lr(old_model, models, weights=None, global_lr=1.0): # FedAvg
    '''return old_model + global_lr * Delta, where Delta is aggregation of local updates'''
    with torch.no_grad():
        Delta = [sub_models(model, old_model) for model in models]
        avg_Delta = avg_models(Delta, weights=weights)
        new_model = add_models(old_model, avg_Delta, alpha=global_lr)
        assign_models(models, new_model)
    return

def aggregate_momentum(old_model, server_momentum, models, weights=None, global_lr=1.0, \
                 momentum_coeff=0.9): # FedAvg
    '''return old_model + global_lr * Delta + 0.9 momentum, where Delta is aggregation of local updates
    Polyak's momentum'''
    with torch.no_grad():
        Delta = [sub_models(model, old_model) for model in models]
        avg_Delta = avg_models(Delta, weights=weights)
        avg_Delta = scale_model(avg_Delta, global_lr)
        server_momentum = add_models(avg_Delta, server_momentum, momentum_coeff)
        new_model = add_models(old_model, avg_Delta)
        assign_models(new_model, models)
    return


def calculate_mean_data(args, model, data_loader):
    data, label = [], []
    mean_batch = 2
    last = len(data_loader)-1
    Xmean, ymean = [], []
    if last==0:
        return None, None

    for i, (X, y) in enumerate(data_loader):
        if i==last: break
        data.append(X)
        label.append(y.long())
    
    data = torch.stack(data, dim=0)
    label = torch.stack(label, dim=0)

    random_ids = torch.randperm(len(data))
    data, label = data[random_ids], label[random_ids]
    data = torch.split(data, min(mean_batch, len(data)))
    label = torch.split(label, min(mean_batch, len(data)))

    for d, l in zip(data, label):
        Xmean.append(torch.mean(d, dim=0))
        ymean.append(torch.mean(F.one_hot(l, num_classes=args.num_classes).to(dtype=torch.float32), dim=0))
    Xmean = torch.stack(Xmean, dim=0)
    ymean = torch.stack(ymean, dim=0)
    return Xmean, ymean

def agg_smash_data(args, server_model, train_loaders):
    Xg, Yg = [], []
    for data_loader in train_loaders:
        Xmean, ymean = calculate_mean_data(args, server_model, data_loader)
        if Xmean is None: continue
        Xg.append(Xmean)
        Yg.append(ymean)
    Xg = torch.cat(Xg, dim=0)
    Yg = torch.cat(Yg, dim=0)
    return Xg, Yg
    