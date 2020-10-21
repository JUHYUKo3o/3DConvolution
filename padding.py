# 3D Convolution
# Padding Function

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda

# Module
mod = SourceModule(open("3DConvolution\\padding.cu", "r", encoding="utf-8").read())
cu_pad = mod.get_function("cu_pad")

class padding():
    # CUDA Limit size
    cu_lim = 32
    def __init__(self,D,K,mode = 'vaild'):
        # D : Data, K = kernel,
        kw = int(K.shape[0]) # kernel width
        kh = int(K.shape[1]) # kernel height
        kn = int(K.shape[2])
        # size setting (padding)
        if mode == 'vaild':
            aw = int(D.shape[0]-kw+1)
            ah = int(D.shape[1]-kh+1)
            an = int(D.shape[2])
            P = D.astype(np.float64)
        elif mode == 'same':
            D32 = D.T.astype(np.float32)
            aw = int(D.shape[0])
            ah = int(D.shape[1])
            an = int(D.shape[2])

            aw_rem = aw % self.cu_lim
            if (aw_rem == 0):
                aw_n = int(aw/self.cu_lim)
            else : 
                aw_n = int(aw/self.cu_lim +1)
            
            ah_rem = ah % self.cu_lim
            if (ah_rem == 0):
                ah_n = int(ah/self.cu_lim)
            else : 
                ah_n = int(ah/self.cu_lim +1)
            
            # result size
            P32 = np.zeros([aw+kw-1,ah+kh-1,an]).T.astype(np.float32)

            # allocate memory on device
            d_gpu = cuda.mem_alloc(D32.nbytes)
            p_gpu = cuda.mem_alloc(P32.nbytes)

            # memory copy (host to device)
            cuda.memcpy_htod(d_gpu, D32)
            cuda.memcpy_htod(p_gpu, P32)

            # CUDA input data 32-bit
            kw32 = np.int32(kw)
            kh32 = np.int32(kh)
            aw_rem32 = np.int32(aw_rem)
            ah_rem32 = np.int32(ah_rem)

            # padding by CUDA
            cu_pad(d_gpu,
                kw32,kh32,aw_rem32, ah_rem32,
                p_gpu,
                block=(self.cu_lim,self.cu_lim,1),
                grid=(aw_n,ah_n,an))

            # memory copy (device to host)
            cuda.memcpy_dtoh(P32, p_gpu)

            d_gpu.free()
            p_gpu.free()
            P = P32.T.astype(np.float64)
        
        self.P = P # padding data
        self.C = np.zeros([aw,ah,int(an*kn)]).astype(np.float64) # Convolution result ZeroMatrix