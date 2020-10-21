# 3D Convolution
# 

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda

from padding import padding

# Module
mod = SourceModule(open("3DConvolution\\convolution3.cu", "r", encoding="utf-8").read())
cu_conv = mod.get_function("cu_conv")

class convolution():
    # CUDA Limit size
    cu_lim = 32
    def __init__(self,D,K,B,mode='vaild'):
        # D : input data, K : Kernel, B : bias

        self.D = D
        self.K = K
        self.B = B
        self.mode = mode

        self.kw = int(K.shape[0]) # kernel width
        self.kh = int(K.shape[1]) # kernel height
        self.kn = int(K.shape[2])
        
        pad = padding(D,K,mode)

        self.P = pad.P
        self.C = pad.C

    def conv3(self):
        # 32 bit data
        A32 = self.P.astype(np.float32)
        K32 = self.K.astype(np.float32)
        B32 = self.B.astype(np.float32)
        C32 = self.C.astype(np.float32)
        # kernel size
        kw = self.kw
        kh = self.kh
        kn = self.kn
        kw32 = np.int32(kw)
        kh32 = np.int32(kh)
        kn32 = np.int32(kn)
        # Data size
        aw = int(self.P.shape[0])
        ah = int(self.P.shape[1])
        an = int(self.P.shape[2])
        # Result size
        cw = int(self.C.shape[0])
        ch = int(self.C.shape[1])
        cn = int(self.C.shape[2])

        # Reminder
        cw_rem = cw % self.cu_lim
        if (cw_rem == 0):
            cw_n = int(cw/self.cu_lim)
        else : 
            cw_n = int(cw/self.cu_lim +1)
        
        ch_rem = ch % self.cu_lim
        if (ch_rem == 0):
            ch_n = int(ch/self.cu_lim)
        else : 
            ch_n = int(ch/self.cu_lim +1)
        
        cw_rem32 = np.int32(cw_rem)
        ch_rem32 = np.int32(ch_rem)

        # Allocate Memory on Device
        a_gpu = cuda.mem_alloc(A32.nbytes)
        k_gpu = cuda.mem_alloc(K32.nbytes)
        B_gpu = cuda.mem_alloc(B32.nbytes)
        c_gpu = cuda.mem_alloc(C32.nbytes)

        # Memory Copy (Host to Device)
        cuda.memcpy_htod(a_gpu, A32)
        cuda.memcpy_htod(k_gpu, K32)
        cuda.memcpy_htod(B_gpu, B32)
        cuda.memcpy_htod(c_gpu, C32)

        # 3D Convolution by CUDA
        cu_conv(a_gpu, k_gpu, B_gpu,
            kw32,kh32,kn32,
            cw_rem32,ch_rem32,
            c_gpu,
            block = (self.cu_lim,self.cu_lim,1),
            grid = (cw_n,ch_n,int(an*kn*kh*kw)))
        
        # Memory Copy (Device to Host)
        cuda.memcpy_dtoh(C32, c_gpu)

        # free
        a_gpu.free()
        k_gpu.free()
        B_gpu.free()
        c_gpu.free()

        return C32.astype(np.float64)