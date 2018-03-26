# coding=utf-8
# python 3.6
# author: Remind
# C03-LogisticRegression_3.3

import numpy as np
import math
from WaterMelon_data import watermelon_dataset as wmd

wmd_x = wmd[0:,0:-1] # sample_n x attr_n
wmd_y = wmd[0:,-1:]
sample_n, attr_n = wmd_x.shape
beta=np.zeros((attr_n + 1,1)) #(attr_n + 1) x 1
x=np.column_stack((wmd_x,np.ones((sample_n,1)))).T #attr_n+1 x sample_n
learn_rate = 0.1
number = 0
while True:
    delta_1 = np.zeros((attr_n + 1,1))
    # delta_2 = np.zeros((attr_n + 1,attr_n + 1))
    for i in range(sample_n):
        temp = math.pow(math.e, np.dot(x[:,i],beta))
        e = temp/(1+temp)
        delta_1 = delta_1 - ((wmd_y[i]-e)*x[:,i]).reshape(3,1)
        # delta_2 = delta_2 + np.dot(x[:,i].T,x[:,i])*e*(1-e)
    beta_new = beta - learn_rate*delta_1
    number += 1
    print(str(number)+':'+str(beta_new))
    if np.dot((beta_new-beta).T, beta_new-beta) <= math.pow(0.000000001,2):
        break
    else:
        beta = beta_new
print('finished!')
# beta = [3.1583, 12.5212, -4.4289]
test = np.dot(beta.T, x)
print(test)