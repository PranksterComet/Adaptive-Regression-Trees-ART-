// X has logical shape (K,n,d), Phi has shape (K,n,p), both row-major.
// Each thread writes one feature of one point; adjacent threads write adjacent
// output entries. Coordinate pairs describe bias/linear/quadratic columns:
// (-1,-1) -> 1; (i,-1) -> x_i; (i,j) -> x_i*x_j (including squares i=j).
extern "C" __global__ void polynomial_features(
    int K, int n, int d, int p, const double* X,
    const int* a, const int* b, double* Phi)
{
    // Use 64-bit offsets even though each individual dimension is int32.
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)K * n * p) return;
    const long long row = t / p;  // Node and sample combined; no data reshaping.
    const int f = t % p;
    const int i = a[f], j = b[f];
    double v = 1.0;
    // -1 is metadata only: neither branch ever reads X at a negative index.
    if (i >= 0) v = X[row * d + i];
    if (j >= 0) v *= X[row * d + j];
    Phi[t] = v;
}
