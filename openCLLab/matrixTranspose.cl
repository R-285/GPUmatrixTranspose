__kernel void matTransposeGlobal(__global int* A, __global int* B, int M, int N) {
   int ty = get_global_id(0);
   int tx = get_global_id(1);
   if (ty >= N || tx >= M) {
    return;
   }
   B[ty * M + tx] = A[ty + tx * N];
}

__kernel void matTransposeLocal(__global int* A, __global int* B, int M, int N) {
    
    __local long tile[32][32];

    const long x = get_local_id(0);
    const long y = get_local_id(1);
    const long x0 = get_group_id(0) * 32 + x;
    const long y0 = get_group_id(1) * 32 + y;

    if (x0 < N && y0 < M) {
        tile[y][x] = A[(y0 * N) + x0];
    }

    const long new_x0 = get_group_id(1) * 32 + x;
    const long new_y0 = get_group_id(0) * 32 + y;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (new_x0 < M && new_y0 < N) {
        B[(new_y0 * M + new_x0)] = tile[x][y];
    }
}

__kernel void matTransposeLocalWithoutBankConflict(__global int* A, __global int* B, int M, int N) {
    __local long tile[32][33];

    const long x = get_local_id(0);
    const long y = get_local_id(1);
    const long x0 = get_group_id(0) * 32 + x;
    const long y0 = get_group_id(1) * 32 + y;

    if (x0 < N && y0 < M) {
        tile[y][x] = A[(y0 * N) + x0];
    }

    const long new_x0 = get_group_id(1) * 32 + x;
    const long new_y0 = get_group_id(0) * 32 + y;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (new_x0 < M && new_y0 < N) {
        B[(new_y0 * M + new_x0)] = tile[x][y];
    }
}