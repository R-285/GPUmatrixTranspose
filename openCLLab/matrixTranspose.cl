__kernel void matTransposeGlobal(__global int* A, __global int* B, int M, int N) {
    int yGl = get_global_id(0);
    int xGl = get_global_id(1);
    if (yGl >= N || xGl >= M)
        return;
    B[yGl * M + xGl] = A[yGl + xGl * N];
}

__kernel void matTransposeLocal(__global int* A, __global int* B, int M, int N) {
    __local int slice[32][32];

    int yLo =  get_local_id(0);
    int xLo =  get_local_id(1);
    int yGl = get_global_id(0);
    int xGl = get_global_id(1);

    if (yGl < N && xGl < M)
        slice[xLo][yLo] = A[(xGl * N) + yGl];    

    int new_yGl = get_group_id(0) * 32 + xLo;
    int new_xGl = get_group_id(1) * 32 + yLo;    

    barrier(CLK_LOCAL_MEM_FENCE);

    if (new_xGl < M && new_yGl < N)
        B[(new_yGl * M + new_xGl)] = slice[yLo][xLo];
}

__kernel void matTransposeLocalWithoutBankConflict(__global int* A, __global int* B, int M, int N) {
    
    __local int slice[32][32+1];

    int yLo = get_local_id(0);
    int xLo = get_local_id(1);
    int yGl = get_global_id(0);
    int xGl = get_global_id(1);

    if (yGl < N && xGl < M) 
        slice[xLo][yLo] = A[(xGl * N) + yGl];
    
    int new_yGl = get_group_id(0) * 32 + xLo;
    int new_xGl = get_group_id(1) * 32 + yLo;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (new_xGl < M && new_yGl < N) 
        B[(new_yGl * M + new_xGl)] = slice[yLo][xLo];
    
}