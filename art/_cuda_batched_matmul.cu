// Compact three pointer tables while the matrices stay in their original slots.
extern "C" __global__ void select_matmul_ptrs(
        int K, int count, const int* ids, const unsigned long long* base,
        unsigned long long* ptrs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    for (int row = 0; row < 3; ++row)
        ptrs[row * K + i] = base[row * K + ids[i]];
}
