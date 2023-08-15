import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

# x = [5, 2, 1, 1, 1]
# y = [11/2**(i+1) for i in range(len(x))]

def smooth_line(accs):
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    x_new = np.linspace(0.1, 0.9, 50)
    accs = np.array(accs)
    bspline = interpolate.make_interp_spline(x, accs)
    return bspline(x_new)

def plot(client_nums, accs_dict, dataset, expname, colors):
    individual = {}
    domain = {}
    indi_cnt_d = {}
    domain_cnt_d = {}
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    x_new = np.linspace(0.1, 0.9, 50)
    for name, accs in accs_dict.items():
        indi_cnt = [0 for _ in range(9)]
        domain_cnt = [0 for _ in range(9)]
        for site_idx, acc in enumerate(accs):
            acc = int(acc)//10
            indi_cnt[acc-1] += client_nums[site_idx]
            domain_cnt[acc-1] += 1
        individual[name] = smooth_line(indi_cnt)
        domain[name] = smooth_line(domain_cnt)
    #     indi_cnt_d[name] = indi_cnt
    #     domain_cnt_d[name] = domain_cnt
    # print(indi_cnt_d)
    # print(domain_cnt_d)
    plt.clf()
    for name in acc_dict.keys():
        plt.plot(x_new, individual[name], linestyle='dashed', color=colors[name], label=name)
    plt.xlabel("Accuracy")
    plt.ylabel("Number of Clients")
    plt.title("Individual fairness")
    plt.legend()
    plt.savefig(f'./individual_fairness_{dataset}_{expname}.png')
    plt.clf()
    for name in acc_dict.keys():
        plt.plot(x_new, domain[name], linestyle='dashed', color=colors[name], label=name)
    plt.xlabel("Accuracy")
    plt.ylabel("Number of Clients")
    plt.title(f"domain_fairness_{dataset}_{expname}")
    plt.legend()
    plt.savefig(f'./domain_fairness_{dataset}_{expname}.png')

x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
x_new = np.linspace(0.1, 0.9, 50)

# digit-5 imbalanced-2 individual

colors = {
    "fedavg": 'orange',
    "harmo-fl": 'gray',
    "cocoop": 'green',
    "ours": 'red',
    }

client_nums = [2, 2, 2, 2, 2]
acc_dict = {
    # "fedavg": [50.5, 19.65, 61.29, 23.07, 19.29],
    "harmo-fl": [85.14, 27.61, 88.17, 51.01, 34.36],
    "cocoop": [84.36, 32.59, 88.17, 49.36, 37.36],
    "ours": [79.43, 34.86, 87.1, 60.07, 39.5],
    }
dataset = "digit"
expname = "even"
plot(client_nums, acc_dict, dataset, expname, colors)

# client_nums = [4, 3, 1, 1, 1]
# acc_dict = {
#     "fedavg": [50.5, 19.65, 61.29, 23.07, 19.29],
#     "harmo-fl": [89.07, 29.82, 86.02, 49.63, 32.64],
#     "cocoop": [84.21, 32.04, 79.57, 78.04, 36.71],
#     "ours": [79.79, 33.2, 79.03, 51.45, 35.71],
#     }
# dataset = "digit"
# expname = "uneven1"
# plot(client_nums, acc_dict, dataset, expname, colors)

client_nums = [5, 1, 1, 2, 1]
acc_dict = {
    # "fedavg": [40.93, 21.01, 63.44, 38.3, 20.5],
    "harmo-fl": [87.93, 26.9, 75.27, 61.93, 29.14],
    "cocoop": [84.21, 32.04, 77.42, 60.72, 32.29],
    "ours": [83.29, 40.45, 74.73, 70.42, 38.93],
    }
dataset = "digit"
expname = "uneven2"
plot(client_nums, acc_dict, dataset, expname, colors)

client_nums = [5, 1, 3, 2, 2]
acc_dict = {
    # "fedavg": [44.29, 17.38, 68.82, 34.2, 16.79],
    "harmo-fl": [89.14, 25.04, 89.25, 60.95, 35.71],
    "cocoop": [89.71, 45.99, 89.78, 79.14, 59.86],
    "ours": [88.64, 44.99, 88.17, 78.48, 60.71],
    }
dataset = "digit"
expname = "uneven1"
plot(client_nums, acc_dict, dataset, expname, colors)

client_nums = [2, 2, 2, 2, 2, 2]
acc_dict = {
    # "fedavg": [83.46, 43.53, 82.88, 34.4, 91.87, 76.17],
    "harmo-fl": [86.54, 51.14, 88.53, 55.3, 94.91],
    "cocoop": [89.16, 49.47, 88.37, 40.8, 92.85, 83.94],
    "ours": [87.07, 49.47, 88.37, 63.6, 92.52, 86.64],
    }
dataset = "domainnet"
expname = "even"
plot(client_nums, acc_dict, dataset, expname, colors)

client_nums = [5, 1, 4, 1, 2, 2]
acc_dict = {
    # "fedavg": [80.42, 41.86, 81.91, 26.5, 94.17, 78.16],
    "harmo-fl": [90.68, 49.77, 89.01, 52.2, 96.47, 87.36],
    "cocoop": [87.07, 50.99, 89.34, 51.5, 95.15, 84.12],
    "ours": [85.93, 49.32, 90.95, 65.4, 94.66, 85.02],
    }
dataset = "domainnet"
expname = "uneven1"
plot(client_nums, acc_dict, dataset, expname, colors)

# client_nums = [6, 3, 1, 1, 1, 1]
# acc_dict = {
#     "fedavg": [75.86, 40.03, 75.44, 23.1, 86.11, 70.58],
#     "harmo-fl": [88.59, 53.12, 87.88, 46.6, 94, 84.66],
#     "cocoop": [81.56, 47.18, 81.58, 31.9, 89.56, 77.44],
#     "ours": [79.66, 51.29, 83.36, 49.1, 88.17, 77.44],
#     }
# dataset = "domainnet"
# expname = "uneven2"
# plot(client_nums, acc_dict, dataset, expname, colors)

client_nums = [5, 1, 1, 1, 2, 1]
acc_dict = {
    # "fedavg": [80.8, 41.7, 79.97, 23.7, 93.59, 75.99],
    "harmo-fl": [90.11, 49.32, 87.72, 56.1, 96.3, 85.02],
    "cocoop": [88.4, 48.4, 90.15, 48.5, 94.74, 86.64],
    "ours": [86.69, 52.66, 89.34, 64.9, 94.91, 82.13],
    }
dataset = "domainnet"
expname = "uneven2"
plot(client_nums, acc_dict, dataset, expname, colors)