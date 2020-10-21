// 3D convolution by CUDA

__global__ void cu_conv(const float *A,const float *K,const float *B, int kw, int kh, int kn, int cw_rem, int ch_rem, float *C){
    // A : input data, K : Kernel, B : bias

    int cx = threadIdx.x + blockIdx.x*blockDim.x;
    int cy = threadIdx.y + blockIdx.y*blockDim.y;
    int cz = blockIdx.z/int(kn*kh*kw);

    int n = (blockIdx.z%(kn*kh*kw)) / (kh*kw);
    int j = ((blockIdx.z%(kn*kh*kw)) % (kh*kw)) / kw;
    int i = ((blockIdx.z%(kn*kh*kw)) % (kh*kw)) % kw;

    int cw = blockDim.x*gridDim.x  + cw_rem;
    int ch = blockDim.y*gridDim.y  + ch_rem;

    int aw = cw + (kw-1);
    int ah = ch + (kh-1);

    int cidx = cx + cy*cw + cz*(cw*ch);
    int aidx = (cx+i) + (cy+j)*aw + (cz)*(aw*ah);
    int kidx = i + j*kw + n*(kw*kh);
    int bidx = n;

    if (cx < cw && cy < ch){
        C[cidx] = A[aidx]*K[kidx] + B[bidx]/(kw*kh);
    }
}