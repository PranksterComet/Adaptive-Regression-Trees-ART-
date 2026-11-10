// Initialization and bookkeeping for the multi-node optimizer. Blocks have 256 threads.
#ifndef __CUDACC_RTC__
#include <math.h>
#endif

// Normalize into dst; initialization retains the CPU rule for a tiny norm.
__device__ __forceinline__ void unit_vector(
        int d, const double* src, double* dst, bool fallback) {
    int t = threadIdx.x;
    __shared__ double sums[256];
    double s = 0.0;
    for (int k = t; k < d; k += blockDim.x) s += src[k] * src[k];
    sums[t] = s;
    __syncthreads();
    for (int k = 128; k; k >>= 1) {
        if (t < k) sums[t] += sums[t + k];
        __syncthreads();
    }
    double r = sqrt(sums[0]);
    for (int k = t; k < d; k += blockDim.x)
        // Replacing src[0] by 1 gives unit norm to float64 precision here.
        dst[k] = fallback && r <= 1e-12 ? (k == 0 ? 1.0 : src[k]) : src[k] / r;
}

// One block per selected slot. cuRAND draws are packed in pointer-prefix order.
extern "C" __global__ void initial_directions(
        int d, const int* ids, const double* normal, double* theta) {
    int row = blockIdx.x, node = ids[row];
    double* w = theta + (long long)node * (d + 1);
    unit_vector(d, normal + (long long)row * d, w, true);
    if (threadIdx.x == 0) w[d] = 0.0;
}

// Gather scores into disposable scratch; sorting/partitioning must preserve u.
extern "C" __global__ void initial_scores(
        int n, const int* ids, const double* u, double* scratch) {
    int row = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) scratch[(long long)row * n + i] = u[(long long)ids[row] * n + i];
}

extern "C" __global__ void initial_offsets(
        int count, int d, const int* ids, const double* median, double* theta) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < count) theta[(long long)ids[t] * (d + 1) + d] = median[t];
}

extern "C" __global__ void alpha_steps(
        int K, int steps, double rho, const int* searching,
        const double* alpha, double* alphas) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= K || !searching[j]) return;
    double a = alpha[j];
    for (int k = 0; k < steps; ++k) {
        alphas[(long long)j * steps + k] = a;
        a *= rho;
    }
}

// value/step are already the first two rows of out; append the integer state.
extern "C" __global__ void search_state(
        int K, const int* selected, const int* searching, double* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < K) { out[2 * K + j] = selected[j]; out[3 * K + j] = searching[j]; }
}

extern "C" __global__ void save_feasible(
        int K, int n, int d, int required, const int* active,
        const double* theta, const double* packet, double* last) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)K * (d + 1)) return;
    int j = t / (d + 1);
    double right = packet[7LL * j + 3];
    if (active[j] && packet[7LL * j + 6] == 0.0 && right >= required && n - right >= required)
        last[t] = theta[t];
}

extern "C" __global__ void copy_theta(
        int K, int d, const int* mask, const double* src, double* dst) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < (long long)K * (d + 1) && mask[t / (d + 1)]) dst[t] = src[t];
}

extern "C" __global__ void normalize_theta(
        int d, const int* active, double* theta) {
    int j = blockIdx.x;
    if (!active[j]) return;
    double* w = theta + (long long)j * (d + 1);
    unit_vector(d, w, w, false);
}

// Final hard-fit packet: [MSE, left MSE, right MSE, n_right, lambda_L, lambda_R, bad].
extern "C" __global__ void hard_stats(
        int K, int n, const int* active, const double* y, const double* pred,
        const double* pi, const double* ridge, const int* status, double* packet) {
    int j = blockIdx.x, t = threadIdx.x;
    if (!active[j]) return;
    __shared__ double left[256], right[256], count[256];
    double l = 0.0, r = 0.0, c = 0.0;
    for (int i = t; i < n; i += blockDim.x) {
        long long row = (long long)j * n + i;
        bool goes_right = pi[row] >= 0.5;
        double e = y[row] - pred[2 * row + goes_right];
        if (goes_right) { r += e * e; c += 1.0; }
        else l += e * e;
    }
    left[t] = l; right[t] = r; count[t] = c;
    __syncthreads();
    for (int k = 128; k; k >>= 1) {
        if (t < k) { left[t] += left[t + k]; right[t] += right[t + k]; count[t] += count[t + k]; }
        __syncthreads();
    }
    if (t == 0) {
        double* out = packet + 7LL * j;
        out[0] = (left[0] + right[0]) / n;
        out[1] = count[0] < n ? left[0] / (n - count[0]) : 0.0;
        out[2] = count[0] > 0 ? right[0] / count[0] : 0.0;
        out[3] = count[0]; out[4] = ridge[j]; out[5] = ridge[K + j];
        out[6] = status[j] || status[K + j] || !isfinite(out[0]);
    }
}

// Geometry-only completion: count hard memberships without fitting child models.
// One block per occupied node; inactive slots never contribute.
extern "C" __global__ void partition_counts(
        int n, const int* active, const double* pi, double* packet) {
    int j = blockIdx.x, t = threadIdx.x;
    if (!active[j]) return;
    __shared__ int right[256], bad[256];
    int count = 0, invalid = 0;
    for (int i = t; i < n; i += blockDim.x) {
        double v = pi[(long long)j * n + i];
        count += v >= 0.5;
        invalid |= !isfinite(v);
    }
    right[t] = count; bad[t] = invalid;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (t < s) { right[t] += right[t + s]; bad[t] |= bad[t + s]; }
        __syncthreads();
    }
    if (t == 0) {
        packet[7LL * j + 3] = right[0];
        packet[7LL * j + 6] = bad[0];
    }
}

// Keep only the best restart's small parameter/coefficient arrays on the GPU.
extern "C" __global__ void keep_best(
        int K, int d, int p, const int* mask, const double* theta, const double* beta,
        double* best_theta, double* best_beta) {
    int width = d + 1 + 2 * p;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)K * width) return;
    int j = t / width, k = t % width;
    if (!mask[j]) return;
    if (k <= d) best_theta[(long long)j * (d + 1) + k] = theta[(long long)j * (d + 1) + k];
    else {
        k -= d + 1;
        long long i = (long long)((k / p) * K + j) * p + k % p;
        best_beta[i] = beta[i];
    }
}
