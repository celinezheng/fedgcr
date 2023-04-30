import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from math import sqrt
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# Define Data


si = [23, 47, 71, 94, 118]
si = [s/sum(si) for s in si]
mnist_train_loss = [1.59, 1.31, 1.36, 1.35, 1.25]
mean_loss = 0
for i, li in enumerate(mnist_train_loss):
    mean_loss += li * si[i]

mnist_train_loss = [sqrt(ti*mean_loss) for ti in mnist_train_loss]
print(mnist_train_loss)
alphas = [0 for _ in si]
q = 5
for i in range(len(alphas)):
    alphas[i] = si[i] * (mnist_train_loss[i]**(q+1))
sum_alpha = sum(alphas)
alphas = [ai/sum_alpha for ai in alphas]
index = ['M1', 'M2', 'M3', 'M4', 'M5']
df = pd.DataFrame({'fedavg': si,
                   'ours': alphas}, index=index)
ax = df.plot.bar(rot=0, title='intra-domain reweighting: MNIST')
ax.set_xlabel("Client Index")
ax.set_ylabel("Aggregation Weight")
plt.savefig(f'./intra-MNIST.png')



si = [23, 47, 71, 94, 118, 94]
si = [s/sum(si) for s in si]
sum_mnist = sum(si[:5])
mnist_loss = [1.59, 1.31, 1.36, 1.35, 1.25]
usps_loss = 1.35
loss_mnist = 0
for i, li in enumerate(mnist_loss):
    loss_mnist += li * si[i] / sum_mnist
print(loss_mnist)
mnist_loss_c = [sqrt(ti * loss_mnist) for ti in mnist_loss]
print(mnist_loss_c)
print(mnist_loss)
ours = [0 for _ in si]
drfl = [0 for _ in si]
q = 7
for i in range(len(si)):
    if i<5:
        ours[i] = si[i] * (mnist_loss_c[i]**(q+1))
        drfl[i] = si[i] * (mnist_loss[i]**(q+1))
    else:
        ours[i] = si[i] * (usps_loss**(q+1))
        drfl[i] = ours[i]

sum_ours = sum(ours)
sum_drfl = sum(drfl)
ours = [ai/sum_ours for ai in ours]
drfl = [ai/sum_drfl for ai in drfl]

index = ['M1', 'M2', 'M3', 'M4', 'M5', 'U1']
df = pd.DataFrame({'fedavg': si,
                   'drfl': drfl,
                   'ours': ours}, index=index)
ax = df.plot.bar(rot=0, title='inter-domain reweighting: Digit-5')
ax.set_xlabel("Client Index")
ax.set_ylabel("Aggregation Weight")
plt.savefig(f'./inter-MNIST.png')
