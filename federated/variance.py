import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

# x = [5, 2, 1, 1, 1]
# y = [11/2**(i+1) for i in range(len(x))]

def plot(client_nums, accs_dict, dataset, expname, colors):
    mean = []
    std = []
    names = accs_dict.keys()
    for accs in accs_dict.values():
        accs = np.array(accs)
        mean.append(np.mean(accs))
        std.append(np.std(accs))
    
    fig, ax = plt.subplots()
    x_pos = np.arange(len(names))
    ax.bar(x_pos, mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Performance')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title(f"performance_{dataset}_{expname}")
    ax.yaxis.grid(True)
    plt.xlabel("Algorithm")
    plt.ylabel("Performance")
    plt.savefig(f'./performance_{dataset}_{expname}.png')

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
    "fedavg": [50.5, 19.65, 61.29, 23.07, 19.29],
    "q-ffl": [42.57, 19.7, 60.75, 23.46, 18.14],
    "drfl": [47.21, 19.45, 59.68, 24.02, 18.5], 
    "harmo-fl": [85.14, 27.61, 88.17, 51.01, 34.36],
    "cocoop": [84.36, 32.59, 88.17, 49.36, 37.36],
    "ours": [79.43, 34.86, 87.1, 60.07, 39.5],
    }
dataset = "digit"
expname = "even"
plot(client_nums, acc_dict, dataset, expname, colors)

client_nums = [5, 1, 3, 2, 2]
acc_dict = {
    "fedavg": [44.29, 17.38, 68.82, 34.2, 16.79],
    "q-ffl": [45.64, 17.83, 69.69, 31.23, 17.43],
    "drfl": [42.64, 18.14, 65.05, 35.24, 16.64],
    "harmo-fl": [89.14, 25.04, 89.25, 60.95, 35.71],
    "cocoop": [89.71, 45.99, 89.78, 79.14, 59.86],
    "ours": [88.64, 44.99, 88.17, 78.48, 60.71],
    }
dataset = "digit"
expname = "uneven1"
plot(client_nums, acc_dict, dataset, expname, colors)

client_nums = [5, 1, 1, 2, 1]
acc_dict = {
    "fedavg": [40.93, 21.01, 63.44, 38.3, 20.5],
    "q-ffl": [43.93, 19.85, 58.6, 30.8, 21.21],
    "drfl": [39.79, 21.26, 60.22, 39.68, 19.86],
    "harmo-fl": [87.93, 26.9, 75.27, 61.93, 29.14],
    "cocoop": [84.21, 32.04, 77.42, 60.72, 32.29],
    "ours": [83.29, 40.45, 74.73, 70.42, 38.93],
    }
dataset = "digit"
expname = "uneven2"
plot(client_nums, acc_dict, dataset, expname, colors)


client_nums = [2, 2, 2, 2, 2, 2]
acc_dict = {
    "fedavg": [83.46, 43.53, 82.88, 34.4, 91.87, 76.17],
    "q-ffl": [80.99, 41.55, 81.42, 33, 91.37, 72.92],
    "drfl": [81.94, 41.86, 81.91, 33.5, 91.21, 74.01],
    "harmo-fl": [86.54, 51.14, 88.53, 55.3, 94.91],
    "cocoop": [89.16, 49.47, 88.37, 40.8, 92.85, 83.94],
    "ours": [87.26, 53.27, 89.01, 67, 94.33, 87.55],
    }
dataset = "domainnet"
expname = "even"
plot(client_nums, acc_dict, dataset, expname, colors)

client_nums = [5, 1, 4, 1, 2, 2]
acc_dict = {
    "fedavg": [80.42, 41.86, 81.91, 26.5, 94.17, 78.16],
    "q-ffl": [76.81, 39.27, 80.78, 25.7, 90.22, 73.1],
    "drfl": [79.66, 40.79, 81.26, 27.9, 93.76, 76.9],
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
    "fedavg": [80.8, 41.7, 79.97, 23.7, 93.59, 75.99],
    "q-ffl": [78.9, 40.33, 79.16, 22.4, 91.21, 73.1],
    "drfl": [80.42, 40.79, 79.81, 24.2, 93.26, 75.99],
    "harmo-fl": [90.11, 49.32, 87.72, 56.1, 96.3, 85.02],
    "cocoop": [88.4, 48.4, 90.15, 48.5, 94.74, 86.64],
    "ours": [86.69, 52.66, 89.34, 64.9, 94.91, 82.13],
    }
dataset = "domainnet"
expname = "uneven2"
plot(client_nums, acc_dict, dataset, expname, colors)