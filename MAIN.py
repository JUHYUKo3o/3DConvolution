# 3D Convolution 

import numpy as np
from Convolution import convolution

D = np.random.rand(30,30,1)
K = np.random.rand(3,3,1)
B = np.random.rand(3)

C = convolution(D,K,B,mode='same').conv3()

print(C)
