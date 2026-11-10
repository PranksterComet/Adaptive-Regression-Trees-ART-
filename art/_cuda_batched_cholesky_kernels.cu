// Small checks and layout conversions around the batched library operations.
#ifndef __CUDACC_RTC__
#include <math.h>
#endif

// Validate newly loaded data without allocating a Phi-sized boolean array.
extern "C" __global__ void batch_check_data(
        int k, int n, int p, const double* phi, const double* y, int* invalid) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)k * n * p) return;
    if (!isfinite(phi[t]) || (t % p == 0 && !isfinite(y[t / p])))
        atomicExch(invalid + t / ((long long)n * p), 1);
}

extern "C" __global__ void batch_weights(
        int K, int n, double floor, const double* pi, double* q, int* invalid) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long count = (long long)K * n;
    if (t >= count) return;
    double v = pi[t];
    bool bad = !isfinite(v) || v < 0.0 || v > 1.0;
    if (bad) atomicExch(invalid + t / n, 1);
    q[t] = bad ? 0.0 : fmax(1.0 - v, floor);
    q[count + t] = bad ? 0.0 : fmax(v, floor);
}

// One block per node/side. Reduce the trace, add ridge, and pack each matrix
// into column-major storage required by cuSOLVER (the source Gram is row-major).
__device__ __forceinline__ void regularize_system(
        int K, int p, int j, double ridge, const double* grams, const int* invalid,
        double* a, double* b, double* lambda, int* status) {
    int t = threadIdx.x;
    long long base = (long long)j * p * p;
    __shared__ double trace[256];
    double sum = 0.0;
    for (int i = t; i < p; i += blockDim.x) sum += grams[base + (long long)i * p + i];
    trace[t] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s /= 2) {
        if (t < s) trace[t] += trace[t + s];
        __syncthreads();
    }
    if (t == 0) {
        lambda[j] = ridge == 0.0 ? 0.0 : ridge * (trace[0] / p);
        status[j] = invalid[j % K] ? -1 :
            (!isfinite(trace[0]) || !isfinite(lambda[j]) ? -2 : 0);
    }
    __syncthreads();
    bool bad = status[j] != 0;
    for (long long i = t; i < (long long)p * p; i += blockDim.x) {
        int row = i / p, col = i % p;
        a[base + (long long)col * p + row] = bad ? (double)(row == col) :
            grams[base + i] + (row == col ? lambda[j] : 0.0);
    }
    if (bad) for (int i = t; i < p; i += blockDim.x) b[(long long)j * p + i] = 0.0;
}

// A failed system must not enter potrsBatched with an invalid factor. Substituting
// I/0 isolates it without compacting arrays or synchronizing to read GPU statuses.
__device__ __forceinline__ void protect_system(
        int p, int j, int info, double* a, double* b, int* status) {
    int t = threadIdx.x;
    long long base = (long long)j * p * p;
    __shared__ int bad[256];
    int invalid = 0;
    for (long long i = t; i < (long long)p * p; i += blockDim.x) {
        int row = i % p, col = i / p;
        if (row >= col && (!isfinite(a[base + i]) || (row == col && a[base + i] <= 0.0)))
            invalid = 1;
    }
    for (int i = t; i < p; i += blockDim.x)
        if (!isfinite(b[(long long)j * p + i])) invalid = 1;
    bad[t] = invalid;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s /= 2) {
        if (t < s) bad[t] |= bad[t + s];
        __syncthreads();
    }
    if (t == 0 && status[j] == 0)
        status[j] = info != 0 ? info : (bad[0] ? -2 : 0);
    __syncthreads();
    if (status[j] == 0) return;
    for (long long i = t; i < (long long)p * p; i += blockDim.x)
        a[base + i] = (double)(i % p == i / p);
    for (int i = t; i < p; i += blockDim.x) b[(long long)j * p + i] = 0.0;
}

__device__ __forceinline__ void finish_system(
        int p, int j, double* b, const int* solve_info, int* status) {
    int t = threadIdx.x;
    __shared__ int bad[256];
    int invalid = solve_info[0] != 0;
    for (int i = t; i < p; i += blockDim.x)
        if (!isfinite(b[(long long)j * p + i])) invalid = 1;
    bad[t] = invalid;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s /= 2) {
        if (t < s) bad[t] |= bad[t + s];
        __syncthreads();
    }
    if (t == 0 && status[j] == 0 && bad[0]) status[j] = -3;
    __syncthreads();
    if (status[j] != 0)
        for (int i = t; i < p; i += blockDim.x) b[(long long)j * p + i] = nan("");
}

extern "C" __global__ void batch_regularize(
        int K, int p, double ridge, const double* grams, const int* invalid,
        double* a, double* b, double* lambda, int* status) {
    regularize_system(K, p, blockIdx.x, ridge, grams, invalid, a, b, lambda, status);
}

extern "C" __global__ void batch_protect(
        int p, double* a, double* b, const int* info, int* status) {
    protect_system(p, blockIdx.x, info[blockIdx.x], a, b, status);
}

extern "C" __global__ void batch_finish(
        int p, double* b, const int* solve_info, int* status) {
    finish_system(p, blockIdx.x, b, solve_info, status);
}

// Indexed variants keep all large arrays in their original node slots.
extern "C" __global__ void batch_weights_active(
        int K, int n, int count, double floor, const int* ids,
        const double* pi, double* q, int* invalid) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)count * n) return;
    int node = ids[t / n];
    long long row = (long long)node * n + t % n;
    double v = pi[row];
    bool bad = !isfinite(v) || v < 0.0 || v > 1.0;
    if (bad) atomicExch(invalid + node, 1);
    q[row] = bad ? 0.0 : fmax(1.0 - v, floor);
    q[(long long)K * n + row] = bad ? 0.0 : fmax(v, floor);
}

extern "C" __global__ void batch_weighted_inputs(
        int K, int n, int p, int count, int side, const int* ids,
        const double* phi, const double* y, const double* q, double* wp, double* wy) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)count * n * p) return;
    int node = ids[t / ((long long)n * p)], col = t % p;
    long long row = (long long)node * n + (t / p) % n;
    double w = q[(long long)side * K * n + row];
    wp[row * p + col] = phi[row * p + col] * w;
    if (col == 0) wy[row] = y[row] * w;
}

extern "C" __global__ void batch_system_ptrs(
        int K, int count, const int* ids, const unsigned long long* a,
        const unsigned long long* b, unsigned long long* ap, unsigned long long* bp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * count) return;
    int j = (i / count) * K + ids[i % count];
    ap[i] = a[j]; bp[i] = b[j];
}

extern "C" __global__ void batch_regularize_active(
        int K, int p, int count, double ridge, const int* ids,
        const double* grams, const int* invalid,
        double* a, double* b, double* lambda, int* status) {
    int i = blockIdx.x, j = (i / count) * K + ids[i % count];
    regularize_system(K, p, j, ridge, grams, invalid, a, b, lambda, status);
}

extern "C" __global__ void batch_protect_active(
        int K, int p, int count, const int* ids, double* a, double* b,
        const int* info, int* status) {
    int i = blockIdx.x, j = (i / count) * K + ids[i % count];
    protect_system(p, j, info[i], a, b, status);
}

extern "C" __global__ void batch_finish_active(
        int K, int p, int count, const int* ids, double* b,
        const int* solve_info, int* status) {
    int i = blockIdx.x, j = (i / count) * K + ids[i % count];
    finish_system(p, j, b, solve_info, status);
}
