#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/logsumexp.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Kernel: each warp computes log-sum-exp of its assigned row, writes to output.
__global__ void logsumexp_kernel(const __half* __restrict__ logits,
                                  float* __restrict__ results,
                                  int vocab_size,
                                  int num_rows) {
    // One warp per row (warp_id = blockIdx.x * warps_per_block + warpIdx).
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= num_rows) return;

    float lse = warp_log_sum_exp(logits + (size_t)warp_id * vocab_size, vocab_size);

    // Only lane 0 writes the result.
    if ((threadIdx.x & 31) == 0) {
        results[warp_id] = lse;
    }
}

// CPU double-precision log-sum-exp reference.
static double cpu_lse(const std::vector<float>& logits) {
    double max_val = -1e300;
    for (float x : logits) if (x > max_val) max_val = x;
    double sum = 0.0;
    for (float x : logits) sum += std::exp((double)x - max_val);
    return max_val + std::log(sum);
}

static bool test_vocab(int vocab_size) {
    int num_rows = 4;
    size_t n = (size_t)num_rows * vocab_size;

    std::vector<float> h_logits_f(n);
    srand(42 + vocab_size);
    for (size_t i = 0; i < n; ++i) {
        h_logits_f[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }

    // Convert to half.
    std::vector<__half> h_logits_h(n);
    for (size_t i = 0; i < n; ++i) h_logits_h[i] = __float2half(h_logits_f[i]);

    __half* d_logits;
    float*  d_results;
    CUDA_CHECK(cudaMalloc(&d_logits,  n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_results, num_rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits_h.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

    // 128 threads per block = 4 warps, each handling one row.
    int threads_per_block = 128;
    int rows_per_block = threads_per_block / 32;
    int blocks = (num_rows + rows_per_block - 1) / rows_per_block;
    logsumexp_kernel<<<blocks, threads_per_block>>>(d_logits, d_results, vocab_size, num_rows);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_results(num_rows);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    bool pass = true;
    for (int row = 0; row < num_rows; ++row) {
        std::vector<float> row_logits(h_logits_f.begin() + row * vocab_size,
                                       h_logits_f.begin() + row * vocab_size + vocab_size);
        // Reference uses the float16-rounded values to match what the kernel sees.
        std::vector<float> row_h(vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            row_h[v] = __half2float(h_logits_h[row * vocab_size + v]);
        }
        double ref = cpu_lse(row_h);
        double got = (double)h_results[row];
        double rel_err = std::abs(ref - got) / (std::abs(ref) + 1e-8);
        if (rel_err > 1e-3) {
            printf("  FAIL vocab=%d row=%d: ref=%.6f got=%.6f rel_err=%.2e\n",
                   vocab_size, row, ref, got, rel_err);
            pass = false;
        }
    }
    if (pass) {
        printf("  PASS vocab_size=%d\n", vocab_size);
    }

    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_results));
    return pass;
}

int main() {
    printf("=== test_logsumexp ===\n");
    bool all_pass = true;
    for (int vocab : {32000, 65536, 128256}) {
        all_pass &= test_vocab(vocab);
    }
    printf("%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
