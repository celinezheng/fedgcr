import torch
import torch.nn as nn
from utils.finch import FINCH
from tqdm import tqdm
import numpy as np

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos

def proto_aggregation(local_protos_list, client_num):
    agg_protos_label = dict()
    # for idx in self.online_clients:
    for idx in range(client_num):
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]
    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
            proto_list = np.array(proto_list)

            c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                        ensure_early_exit=False, verbose=True)

            m, n = c.shape
            class_cluster_list = []
            for index in range(m):
                class_cluster_list.append(c[index, -1])

            class_cluster_array = np.array(class_cluster_list)
            uniqure_cluster = np.unique(class_cluster_array).tolist()
            agg_selected_proto = []

            for _, cluster_index in enumerate(uniqure_cluster):
                selected_array = np.where(class_cluster_array == cluster_index)
                selected_proto_list = proto_list[selected_array]
                proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                agg_selected_proto.append(torch.tensor(proto))
            agg_protos_label[label] = agg_selected_proto
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label

def hierarchical_info_loss(f_now, label, all_f, mean_f, all_global_protos_keys):
    device = torch.device('cuda')
    # f_pos = np.array(all_f)[[all_global_protos_keys == label.item()][0]].to(device)
    # f_neg = torch.cat(list(np.array(all_f)[all_global_protos_keys != label.item()])).to(device)
    label_position = 0
    for idx in all_global_protos_keys:
        if idx == label.item():
            label_position = idx
            break
    f_pos = all_f[label_position].to(device)
    f_neg = torch.cat([all_f[idx] for idx in all_global_protos_keys if idx!=label_position]).to(device)
    xi_info_loss = calculate_infonce(f_now, f_pos, f_neg)

    mean_f_pos = mean_f[label_position].to(device)
    mean_f_pos = mean_f_pos.view(1, -1)
    # mean_f_neg = torch.cat(list(np.array(mean_f)[all_global_protos_keys != label.item()]), dim=0).to(self.device)
    # mean_f_neg = mean_f_neg.view(9, -1)

    loss_mse = nn.MSELoss()
    cu_info_loss = loss_mse(f_now, mean_f_pos)

    hierar_info_loss = xi_info_loss + cu_info_loss
    return hierar_info_loss

def calculate_infonce(f_now, f_pos, f_neg):
    device = torch.device('cuda')
    infoNCET = 0.02
    f_proto = torch.cat((f_pos, f_neg), dim=0)
    l = torch.cosine_similarity(f_now, f_proto, dim=1)
    l = l / infoNCET

    exp_l = torch.exp(l)
    exp_l = exp_l.view(1, -1)
    pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
    pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(device)
    pos_mask = pos_mask.view(1, -1)
    # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
    pos_l = exp_l * pos_mask
    sum_pos_l = pos_l.sum(1)
    sum_exp_l = exp_l.sum(1)
    infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
    return infonce_loss

# def loc_update(self, priloader_list, parti_num):
#     total_clients = list(range(args.parti_num))

#     for i in total_clients:
#         self._train_net(i, self.nets_list[i], priloader_list[i])
#     self.global_protos = self.proto_aggregation(self.local_protos)
#     self.aggregate_nets(None)
#     return None

def train_fpl(index, net, train_loader, optimizer, global_protos, local_protos):
    device = torch.device('cuda')
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    if len(global_protos) != 0:
        all_global_protos_keys = np.array(list(global_protos.keys()))
        all_f = []
        mean_f = []
        for protos_key in all_global_protos_keys:
            temp_f = global_protos[protos_key]
            temp_f = torch.cat(temp_f, dim=0).to(device)
            all_f.append(temp_f.cpu())
            mean_f.append(torch.mean(temp_f, dim=0).cpu())
        all_f = [item.detach() for item in all_f]
        mean_f = [item.detach() for item in mean_f]

    num_data = 0
    loss_all = 0
    correct = 0
    agg_protos_label = {}
    train_iter = iter(train_loader)
    for step in tqdm(range(len(train_iter))):
        images, labels = next(train_iter)
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)
        f, outputs =  net.forward(images, return_feature=True)

        lossCE = criterion(outputs, labels)
        loss_all += lossCE.item()
        pred = outputs.data.max(1)[1]
        correct += pred.eq(labels.view(-1)).sum().item()
        num_data += labels.size(0)
        if len(global_protos) == 0:
            loss_InfoNCE = 0 * lossCE
        else:
            i = 0
            loss_InfoNCE = None

            for label in labels:
                if label.item() in global_protos.keys():
                    f_now = f[i].unsqueeze(0)
                    loss_instance = hierarchical_info_loss(f_now, label, all_f, mean_f, all_global_protos_keys)

                    if loss_InfoNCE is None:
                        loss_InfoNCE = loss_instance
                    else:
                        loss_InfoNCE += loss_instance
                i += 1
            loss_InfoNCE = loss_InfoNCE / i
        loss_InfoNCE = loss_InfoNCE

        loss = lossCE + loss_InfoNCE
        loss.backward()
        optimizer.step()

        for i in range(len(labels)):
            if labels[i].item() in agg_protos_label:
                agg_protos_label[labels[i].item()].append(f[i, :])
            else:
                agg_protos_label[labels[i].item()] = [f[i, :]]

    agg_protos = agg_func(agg_protos_label)
    local_protos[index] = agg_protos
    net.to('cpu')
    return local_protos, loss_all/len(train_iter), correct/num_data
