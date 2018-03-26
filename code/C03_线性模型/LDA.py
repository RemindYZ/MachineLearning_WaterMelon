# coding=utf-8
# python 3.6
# author: Remind
# C03-LDA_3.5

import numpy as np
from WaterMelon_data import watermelon_dataset as wmd

wmd_x = wmd[0:,0:-1] # sample_n x attr_n
wmd_y = wmd[0:,-1].tolist()
x = wmd_x.T
sample_n, attr_n = wmd_x.shape
omega=np.zeros((attr_n,1)) #(attr_n + 1) x 1
mu_1 = np.zeros((attr_n, 1))
mu_0 = np.zeros((attr_n, 1))
num_1 = 0
num_0 = 0 
for i in range(sample_n):
    if wmd_y[i] == 1:
        num_1 += 1
        mu_1 += x[:,i].reshape(attr_n,1)
    else:
        num_0 += 1
        mu_0 += x[:,i].reshape(attr_n,1)
mu_1 = mu_1 / num_1
mu_0 = mu_0 / num_0
sigma_0 = np.zeros((attr_n, attr_n))
sigma_1 = np.zeros((attr_n, attr_n))
for i in range(sample_n):
    if wmd_y[i] == 1:
        sigma_0 += np.dot(x[:,i] - mu_1, (x[:,i] - mu_1).T)
    else:
        sigma_1 += np.dot(x[:,i] - mu_0, (x[:,i] - mu_0).T)
S = sigma_0 + sigma_1
U,Sigma,VT = np.linalg.svd(S)
omega = np.linalg.multi_dot([VT.T, np.diag(1/Sigma), U.T, mu_0-mu_1])
print(omega)
#omega=[-0.02123147, -0.04806283]
test = np.dot(wmd_x,omega)
y1 = np.dot(mu_1.T,omega)
y2 = np.dot(mu_0.T,omega)
print(y1,y2)
print(test)