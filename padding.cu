// Padding by CUDA

__global__ void cu_pad(const float *A, int kw, int kh, int aw_rem, int ah_rem, float *P){
    // A : input data, P : padding data
    // kw : kernel width, kh : kernel hieght

    // block = (BLOCK_SIZE,BLOCK_SIZE,1)
    // grid = (aw/BLOCK_SIZE, ah/BLOCK_SIZE, an)

    int tx = threadIdx.x + blockIdx.x*blockDim.x;
    int ty = threadIdx.y + blockIdx.y*blockDim.y;
    int tz = blockIdx.z;

    int aw = (gridDim.x*blockDim.x) + aw_rem;
    int ah = (gridDim.y*blockDim.y) + ah_rem;

    int padw = (kw-1)/2;
    int padh = (kh-1)/2;
    int pw = aw + (2*padw);
    int ph = ah + (2*padh);

    int aidx = tx + ty*aw + tz*(aw*ah);
    int pidx = (tx+padw) + (ty+padh)*pw + tz*(pw*ph);

    if ( tx < aw && ty < ah ){
        P[pidx] = A[aidx];
        
        // Left
        if (tx < padw){
            P[tx + (ty+padh)*pw + tz*(pw*ph)] = A[0 + ty*aw + tz*(aw*ah)];
            // up
            if (ty < padh){
                P[tx + ty*pw + tz*(pw*ph)] = A[0 + 0 + tz*(aw*ah)];}
            // down
            else if ((ah-padh-1) < ty){
                P[tx + (ty+2*padh)*pw + tz*(pw*ph)] = A[0 + (ah-1)*aw + tz*(aw*ah)];}
        }
        // Right
        else if ((aw-padw-1) < tx){
            P[(tx+2*padw) + (ty+padh)*pw + tz*(pw*ph)] = A[(aw-1) + ty*aw + tz*(aw*ah)];
            // up
            if (ty < padh){
                P[(tx+2*padw) + ty*pw + tz*(pw*ph)] = A[(aw-1) + 0 + tz*(aw*ah)];}
            // down
            else if ((ah-padh-1) < ty){
                P[(tx+2*padw) + (ty+2*padh)*pw + tz*(pw*ph)] = A[(aw-1) + (ah-1)*aw + tz*(aw*ah)];}
        }

        // Up
        if (ty < padh){
            P[(tx+padw) + ty*pw + tz*(pw*ph)] = A[tx + 0 + tz*(aw*ah)];
        }
        // Down
        else if ((ah-padh-1) < ty){
            P[(tx+padw) + (ty+2*padh)*pw + tz*(pw*ph)] = A[tx + (ah-1)*aw + tz*(aw*ah)];
        }
    }
}