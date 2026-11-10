// Float64 kernels shared by the GPU QR and Cholesky optimizers.
// All blocks have 256 threads. Single-node matrices use column-major storage;
// batched layouts and grids are documented beside their entry points below.
// NVRTC supplies device math intrinsics without the host C library headers.
// Keep math.h for the CPU kernel-arithmetic tests and offline compilation.
#ifndef __CUDACC_RTC__
#include <math.h>
#endif

__device__ double positive_infinity() {
#ifdef __CUDACC_RTC__
    return __longlong_as_double(0x7ff0000000000000LL);
#else
    return HUGE_VAL;
#endif
}

__device__ double gate(double score) {
    return 1.0 / (1.0 + exp(-fmin(50.0, fmax(-50.0, score))));
}

extern "C" __global__ void gates(
    int n, int d, double temperature, double floor, double alpha,
    int advance, int hard, const double* theta, const double* v,
    const double* inv_r, double* u, double* pi, double* q, double* sq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = advance ? (u[i] + alpha * v[i]) * inv_r[0] : u[i];
    u[i] = a;
    double t = hard ? (a >= theta[d] ? 1.0 : 0.0)
                    : gate((a - theta[d]) / temperature);
    double left = fmax(1.0 - t, hard ? 0.0 : floor);
    double right = fmax(t, hard ? 0.0 : floor);
    pi[i] = t;
    q[i] = left; q[n + i] = right;
    sq[i] = sqrt(left); sq[n + i] = sqrt(right);
}

extern "C" __global__ void qr_inputs(
    int n, int p, int m, int side, double ridge_scale,
    const double* phi, const double* y, const double* sq,
    const double* traces, double* a, double* b)
{
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (long long)m * p) return;
    int i = k % m, j = k / m;
    a[k] = i < n ? phi[i + (long long)j * n] * sq[i]
                  : (i - n == j ? sqrt(ridge_scale * traces[side]) : 0.0);
    if (j == 0) b[i] = i < n ? y[i] * sq[i] : 0.0;
}

extern "C" __global__ void qr_check(
    int m, int p, int side, double tolerance,
    const double* a, const double* b, const int* info, int* status)
{
    __shared__ double lo[256], hi[256];
    __shared__ int bad[256];
    int t = threadIdx.x;
    double low = positive_infinity(), high = 0.0;
    int invalid = 0;
    for (int j = t; j < p; j += blockDim.x) {
        double value = fabs(a[j + (long long)j * m]);
        low = fmin(low, value); high = fmax(high, value);
        invalid |= !isfinite(value) || !isfinite(b[j]);
    }
    lo[t] = low; hi[t] = high; bad[t] = invalid;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) {
            lo[t] = fmin(lo[t], lo[t + s]);
            hi[t] = fmax(hi[t], hi[t + s]);
            bad[t] |= bad[t + s];
        }
        __syncthreads();
    }
    if (t == 0)
        status[side] = info[0] != 0 || info[1] != 0 || bad[0]
                       || lo[0] <= tolerance * hi[0];
}

extern "C" __global__ void add_ridge(
    int p, int side, double ridge_scale, const double* traces, double* gram)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < p) gram[j + (long long)j * p] += ridge_scale * traces[side];
}

extern "C" __global__ void cholesky_check(
    int p, int side, const double* a, const double* b,
    const int* info, int* status)
{
    __shared__ int bad[256];
    int t = threadIdx.x, invalid = 0;
    for (int j = t; j < p; j += blockDim.x) {
        double diagonal = a[j + (long long)j * p];
        invalid |= !(diagonal > 0.0) || !isfinite(diagonal) || !isfinite(b[j]);
    }
    bad[t] = invalid;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) bad[t] |= bad[t + s];
        __syncthreads();
    }
    if (t == 0) status[side] = bad[0] || info[0] != 0 || info[1] != 0;
}

__device__ __forceinline__ void residual_values(
    int n, double temperature, const double* y, const double* pred,
    int row_stride, int side_stride,
    const double* pi, double* errors, double* coeff, double* partials)
{
    __shared__ double loss[256], gz[256], count[256];
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    double f = 0.0, c = 0.0, right = 0.0;
    if (i < n) {
        long long k = (long long)i * row_stride;
        double l = y[i] - pred[k], r = y[i] - pred[k + side_stride];
        l *= l; r *= r;
        errors[k] = l; errors[k + side_stride] = r;
        c = (r - l) * pi[i] * (1.0 - pi[i]) / (n * temperature);
        coeff[i] = c;
        f = ((1.0 - pi[i]) * l + pi[i] * r) / n;
        right = pi[i] >= 0.5 ? 1.0 : 0.0;
    }
    loss[t] = f; gz[t] = -c; count[t] = right;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) {
            loss[t] += loss[t + s]; gz[t] += gz[t + s];
            count[t] += count[t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        partials[blockIdx.x] = loss[0];
        partials[gridDim.x + blockIdx.x] = gz[0];
        partials[2 * gridDim.x + blockIdx.x] = count[0];
    }
}

extern "C" __global__ void residual_stats(
    int n, double temperature, const double* y, const double* pred,
    const double* pi, double* errors, double* coeff, double* partials)
{
    residual_values(n, temperature, y, pred, 1, n, pi, errors, coeff, partials);
}

// geometry = [||w||^2, w.p_w, ||p_w||^2, p_z].
// packet = [loss, projected norm, slope, right count, lambda_L, lambda_R, bad].
__device__ __forceinline__ void direction_values(
    int d, const double* theta, const double* gw, const double* totals,
    double lambda_left, double lambda_right, int failed,
    double* p, double* geometry, double* packet)
{
    __shared__ double a[256], b[256], c[256];
    int t = threadIdx.x;
    double ww = 0.0, wg = 0.0;
    for (int j = t; j < d; j += blockDim.x) {
        ww += theta[j] * theta[j]; wg += theta[j] * gw[j];
    }
    a[t] = ww; b[t] = wg;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) { a[t] += a[t + s]; b[t] += b[t + s]; }
        __syncthreads();
    }
    double norm_w = a[0], projection = b[0] / a[0];
    // Preserve the shared reduction results before reusing its storage.
    __syncthreads();
    double pp = 0.0, wp = 0.0, gp = 0.0;
    for (int j = t; j < d; j += blockDim.x) {
        double value = -gw[j] + projection * theta[j];
        p[j] = value;
        pp += value * value; wp += theta[j] * value; gp += gw[j] * value;
    }
    a[t] = pp; b[t] = wp; c[t] = gp;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) { a[t] += a[t + s]; b[t] += b[t + s]; c[t] += c[t + s]; }
        __syncthreads();
    }
    if (t == 0) {
        double gz = totals[1], norm = sqrt(a[0] + gz * gz);
        p[d] = -gz;
        geometry[0] = norm_w; geometry[1] = b[0];
        geometry[2] = a[0]; geometry[3] = -gz;
        packet[0] = totals[0]; packet[1] = norm;
        packet[2] = c[0] - gz * gz; packet[3] = totals[2];
        packet[4] = lambda_left;
        packet[5] = lambda_right;
        packet[6] = failed || !(norm_w > 0.0) || !isfinite(norm_w)
                    || !isfinite(norm) || !isfinite(packet[2]) || !isfinite(totals[0]);
    }
}

extern "C" __global__ void direction(
    int d, double ridge_scale, const double* theta, const double* gw,
    const double* totals, const double* traces, const int* status,
    double* p, double* geometry, double* packet)
{
    direction_values(d, theta, gw, totals, ridge_scale * traces[0],
                     ridge_scale * traces[1], status[0] || status[1], p, geometry, packet);
}

__device__ __forceinline__ double trial_value(
    int n, double temperature, double z, double u, double v,
    double left, double right, double alpha, const double* geometry)
{
    double r2 = geometry[0] + 2.0 * alpha * geometry[1]
                + alpha * alpha * geometry[2];
    if (!(r2 > 0.0) || !isfinite(r2) || !(temperature > 0.0) || !isfinite(temperature))
        return positive_infinity();
    double score = ((u + alpha * v) / sqrt(r2) - z - alpha * geometry[3]) / temperature;
    double pi = gate(score);
    double f = ((1.0 - pi) * left + pi * right) / n;
    return isfinite(score) && isfinite(f) ? f : positive_infinity();
}

extern "C" __global__ void trial_losses(
    int n, int d, int offset, double temperature, const double* theta,
    const double* u, const double* v, const double* errors,
    const double* geometry, const double* alphas, double* partials)
{
    __shared__ double sums[256];
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    int k = blockIdx.y;
    double alpha = alphas[offset + k];
    double f = 0.0;
    if (i < n) f = trial_value(n, temperature, theta[d], u[i], v[i],
                               errors[i], errors[n + i], alpha, geometry);
    sums[t] = f;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) sums[t] += sums[t + s];
        __syncthreads();
    }
    if (t == 0) partials[k * gridDim.x + blockIdx.x] = sums[0];
}

extern "C" __global__ void select_trial(
    int count, int offset, double armijo, const double* losses,
    const double* alphas, const double* packet, double* response)
{
    if (threadIdx.x != 0) return;
    int selected = -1;
    for (int k = 0; k < count; ++k)
        if (isfinite(losses[k]) && losses[k] <= packet[0]
                + armijo * alphas[offset + k] * packet[2]) { selected = k; break; }
    int k = selected >= 0 ? selected : count - 1;
    response[0] = selected; response[1] = losses[k];
    response[2] = alphas[offset + k];
}

__device__ __forceinline__ void commit_parameters(
    int d, double alpha, const double* p, const double* geometry,
    double* theta, double* inv_r)
{
    __shared__ double scale;
    if (threadIdx.x == 0) {
        scale = 1.0 / sqrt(geometry[0] + 2.0 * alpha * geometry[1]
                          + alpha * alpha * geometry[2]);
        inv_r[0] = scale;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        theta[j] = (theta[j] + alpha * p[j]) * scale;
    if (threadIdx.x == 0) theta[d] += alpha * p[d];
}

extern "C" __global__ void commit_theta(
    int d, double alpha, const double* p, const double* geometry,
    double* theta, double* inv_r)
{
    commit_parameters(d, alpha, p, geometry, theta, inv_r);
}

// Batched entry points: equal n/d across nodes, C-contiguous arrays unless noted.
// Scalars per node: temperature/alpha/inv_r (K,); masks active/searching are int32.
// theta/p (K,d+1), u/v/pi/coeff/y (K,n), pred/errors (K,n,2), gw (K,d).
// Residual partials (K,3,M), totals (K,3), geometry (K,4), packet (K,7).
// Ridge/status use the Cholesky workspace's SIDE-MAJOR (2,K) storage, including
// when the caller passes its transposed public (K,2) views by raw device pointer.
// Masked nodes preserve their outputs. Every mask check is block-uniform and
// happens before barriers; out-of-range sample threads still join reductions.

// Grid (M,K), M=ceil(n/256). Only pi is needed; fitting owns floored weights.
extern "C" __global__ void batched_gates(
    int n, int d, int advance, int hard, const int* active,
    const double* temperature, const double* theta, const double* alpha,
    const double* v, const double* inv_r, double* u, double* pi)
{
    int node = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!active[node] || i >= n) return;
    long long k = (long long)node * n + i;
    double a = advance ? (u[k] + alpha[node] * v[k]) * inv_r[node] : u[k];
    double z = theta[(long long)node * (d + 1) + d];
    double score = (a - z) / temperature[node];
    u[k] = a;
    pi[k] = hard ? (isfinite(a) && isfinite(z) ? (double)(a >= z) : nan(""))
                  : (temperature[node] > 0.0 && isfinite(temperature[node]) && isfinite(score)
                     ? gate(score) : nan(""));
}

// Grid (M,K). Per-node pointers let both entry points share the same arithmetic.
extern "C" __global__ void batched_residual_stats(
    int n, const int* active, const double* temperature,
    const double* y, const double* pred, const double* pi,
    double* errors, double* coeff, double* partials)
{
    int node = blockIdx.y;
    if (!active[node]) return;
    long long base = (long long)node * n;
    residual_values(n, temperature[node], y + base, pred + 2 * base,
                    2, 1, pi + base, errors + 2 * base, coeff + base,
                    partials + (long long)node * 3 * gridDim.x);
}

// Grid (K,). Ridge already contains the effective, trace-scaled penalties.
extern "C" __global__ void batched_direction(
    int K, int d, const int* active, const double* theta, const double* gw,
    const double* totals, const double* ridge, const int* status,
    double* p, double* geometry, double* packet)
{
    int node = blockIdx.x;
    if (!active[node]) return;
    long long base = (long long)node * (d + 1);
    direction_values(d, theta + base, gw + (long long)node * d, totals + 3LL * node,
                     ridge[node], ridge[K + node], status[node] || status[K + node],
                     p + base, geometry + 4LL * node, packet + 7LL * node);
}

// Backtracking: alphas (K,S), trial partials (K,H,M), losses (K,H).
// A shared offset chooses the next H candidates; padded tail candidates are inf.
// Grid (M,H,K). Already accepted/exhausted nodes return before reading inputs.
extern "C" __global__ void batched_trial_losses(
    int n, int d, int steps, int offset, const int* searching,
    const double* temperature, const double* theta, const double* u,
    const double* v, const double* errors, const double* geometry,
    const double* alphas, double* partials)
{
    int node = blockIdx.z, k = blockIdx.y;
    if (!searching[node]) return;
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    double f = 0.0;
    if (offset + k >= steps) f = positive_infinity();
    else if (i < n) {
        long long row = (long long)node * n + i;
        f = trial_value(n, temperature[node], theta[(long long)node * (d + 1) + d],
                        u[row], v[row], errors[2 * row], errors[2 * row + 1],
                        alphas[(long long)node * steps + offset + k], geometry + 4LL * node);
    }
    __shared__ double sums[256];
    sums[t] = f;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) sums[t] += sums[t + s];
        __syncthreads();
    }
    if (t == 0)
        partials[((long long)node * gridDim.y + k) * gridDim.x + blockIdx.x] = sums[0];
}

// Grid (H,K). Mask the reduction too, so stale inactive buffers are never read.
extern "C" __global__ void batched_trial_reduce(
    int blocks, const int* searching, const double* partials, double* losses)
{
    int node = blockIdx.y, k = blockIdx.x, t = threadIdx.x;
    if (!searching[node]) return;
    long long trial = (long long)node * gridDim.x + k;
    double sum = 0.0;
    for (int b = t; b < blocks; b += blockDim.x) sum += partials[trial * blocks + b];
    __shared__ double sums[256];
    sums[t] = sum;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) sums[t] += sums[t + s];
        __syncthreads();
    }
    if (t == 0) losses[trial] = sums[0];
}

// Grid (K,). Initially searching=1 and selected=-1 for participating nodes.
// Select the earliest passing GLOBAL index, then freeze it across later passes.
// If exhausted, selected=-1 and value/step retain the last attempted candidate;
// the optimizer controller applies its existing line-search failure policy.
extern "C" __global__ void batched_select_trial(
    int steps, int batch_size, int offset, double armijo,
    const double* losses, const double* alphas, const double* packet,
    int* searching, int* selected, double* value, double* step)
{
    int node = blockIdx.x;
    if (threadIdx.x != 0 || !searching[node]) return;
    if (packet[7LL * node + 6] != 0.0) { searching[node] = 0; return; }
    int count = steps - offset;
    if (count > batch_size) count = batch_size;
    for (int k = 0; k < count; ++k) {
        double alpha = alphas[(long long)node * steps + offset + k];
        double loss = losses[(long long)node * batch_size + k];
        value[node] = loss; step[node] = alpha;
        if (isfinite(loss) && loss <= packet[7LL * node]
                + armijo * alpha * packet[7LL * node + 2]) {
            selected[node] = offset + k;
            searching[node] = 0;
            return;
        }
    }
    if (offset + count >= steps) searching[node] = 0;
}

// Grid (K,). Call once after search; failed/unselected nodes retain theta/inv_r.
extern "C" __global__ void batched_commit_theta(
    int d, const int* selected, const double* step, const double* p,
    const double* geometry, double* theta, double* inv_r)
{
    int node = blockIdx.x;
    if (selected[node] < 0) return;
    long long base = (long long)node * (d + 1);
    commit_parameters(d, step[node], p + base, geometry + 4LL * node,
                      theta + base, inv_r + node);
}
