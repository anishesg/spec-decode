#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/types.h"
#include "../src/fused_verify.cuh"
#include "../src/reference.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_err)); \
            exit(1); \
        } \
    } while(0)

// Allocate device DraftResult and host copies for a single verification instance.
struct TestInstance {
    SpecConfig cfg;

    // Host storage.
    std::vector<__half> h_target_logits;
    std::vector<__half> h_draft_logits;
    std::vector<int32_t> h_draft_tokens;
    std::vector<float>   h_random_values;

    // Device storage.
    __half*   d_target_logits = nullptr;
    __half*   d_draft_logits  = nullptr;
    int32_t*  d_draft_tokens  = nullptr;
    float*    d_random_values = nullptr;
    DraftResult* d_input      = nullptr;
    VerifyResult* d_output    = nullptr;

    void alloc() {
        int k = cfg.num_draft_tokens;
        int V = cfg.vocab_size;
        h_target_logits.resize((size_t)(k+1) * V);
        h_draft_logits.resize((size_t)k * V);
        h_draft_tokens.resize(k);
        h_random_values.resize(k+1);

        CUDA_CHECK(cudaMalloc(&d_target_logits, (k+1)*(size_t)V*sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_draft_logits,  k    *(size_t)V*sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_draft_tokens,  k    *sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_random_values,(k+1) *sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_input,  sizeof(DraftResult)));
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(VerifyResult)));
    }

    void upload() {
        int k = cfg.num_draft_tokens;
        int V = cfg.vocab_size;
        CUDA_CHECK(cudaMemcpy(d_target_logits, h_target_logits.data(), (k+1)*(size_t)V*sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_draft_logits,  h_draft_logits.data(),  k    *(size_t)V*sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_draft_tokens,  h_draft_tokens.data(),  k    *sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_random_values, h_random_values.data(),(k+1) *sizeof(float),   cudaMemcpyHostToDevice));

        DraftResult dr;
        dr.target_logits  = d_target_logits;
        dr.draft_logits   = d_draft_logits;
        dr.draft_tokens   = d_draft_tokens;
        dr.random_values  = d_random_values;
        CUDA_CHECK(cudaMemcpy(d_input, &dr, sizeof(DraftResult), cudaMemcpyHostToDevice));
    }

    VerifyResult run_fused() {
        host_fused_verify(cfg, d_input, d_output, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        VerifyResult res;
        CUDA_CHECK(cudaMemcpy(&res, d_output, sizeof(VerifyResult), cudaMemcpyDeviceToHost));
        return res;
    }

    VerifyResult run_reference() {
        return verify_reference(cfg, DraftResult{},
                                h_target_logits.data(),
                                h_draft_logits.data(),
                                h_draft_tokens.data(),
                                h_random_values.data());
    }

    void free_all() {
        cudaFree(d_target_logits);
        cudaFree(d_draft_logits);
        cudaFree(d_draft_tokens);
        cudaFree(d_random_values);
        cudaFree(d_input);
        cudaFree(d_output);
    }
};

// Uniform logit (after softmax all tokens have equal probability ~1/V).
static void fill_uniform_logits(__half* buf, int rows, int V) {
    for (int i = 0; i < rows * V; ++i) buf[i] = __float2half(0.0f);
}

// One-hot logit: token `tok` gets high logit, all others low.
static void fill_onehot(__half* buf, int row, int V, int tok, float high = 10.0f, float low = -10.0f) {
    __half* row_ptr = buf + (size_t)row * V;
    for (int v = 0; v < V; ++v) row_ptr[v] = __float2half(low);
    row_ptr[tok] = __float2half(high);
}

// Test 1: target == draft at all positions, random = 0 => all tokens accepted.
static bool test_deterministic_accept(int V, int k) {
    TestInstance t;
    t.cfg = {V, k, k};
    t.alloc();

    fill_uniform_logits(t.h_target_logits.data(), k+1, V);
    fill_uniform_logits(t.h_draft_logits.data(),  k,   V);
    for (int i = 0; i < k; ++i) t.h_draft_tokens[i] = i % V;
    for (int i = 0; i <= k; ++i) t.h_random_values[i] = 0.0f;

    t.upload();
    VerifyResult res = t.run_fused();
    t.free_all();

    if (res.num_accepted != k) {
        printf("  FAIL test1 (V=%d k=%d): expected %d accepted, got %d\n", V, k, k, res.num_accepted);
        return false;
    }
    printf("  PASS test1 (V=%d k=%d): all %d accepted\n", V, k, k);
    return true;
}

// Test 2: target assigns zero probability to draft_token[0] => immediate rejection.
static bool test_deterministic_reject(int V, int k) {
    TestInstance t;
    t.cfg = {V, k, k};
    t.alloc();

    fill_uniform_logits(t.h_target_logits.data(), k+1, V);
    fill_uniform_logits(t.h_draft_logits.data(),  k,   V);

    // Make draft_token[0] = 0; set target[0][0] = -inf (zero probability).
    t.h_draft_tokens[0] = 0;
    t.h_target_logits[0] = __float2half(-65504.0f);  // largest negative half

    for (int i = 1; i < k; ++i) t.h_draft_tokens[i] = i % V;
    for (int i = 0; i <= k; ++i) t.h_random_values[i] = 0.5f;

    t.upload();
    VerifyResult res = t.run_fused();
    t.free_all();

    if (res.num_accepted != 0) {
        printf("  FAIL test2 (V=%d k=%d): expected 0 accepted, got %d\n", V, k, res.num_accepted);
        return false;
    }
    printf("  PASS test2 (V=%d k=%d): rejected at pos 0 as expected\n", V, k);
    return true;
}

// Test 3: positions 0..2 accept with ratio 1.0, position 3 rejects.
static bool test_partial_accept(int V, int k) {
    if (k < 4) return true;  // need at least 4 positions
    TestInstance t;
    t.cfg = {V, k, k};
    t.alloc();

    // Equal logits => ratio = 1.0 => always accept when rand < 1.0.
    fill_uniform_logits(t.h_target_logits.data(), k+1, V);
    fill_uniform_logits(t.h_draft_logits.data(),  k,   V);
    for (int i = 0; i < k; ++i) t.h_draft_tokens[i] = i % V;
    for (int i = 0; i <= k; ++i) t.h_random_values[i] = 0.0f;

    // Force rejection at position 3: set target[3][draft_tokens[3]] = -inf.
    int tok3 = 3 % V;
    t.h_target_logits[3 * V + tok3] = __float2half(-65504.0f);

    t.upload();
    VerifyResult res = t.run_fused();
    t.free_all();

    if (res.num_accepted != 3) {
        printf("  FAIL test3 (V=%d k=%d): expected 3 accepted, got %d\n", V, k, res.num_accepted);
        return false;
    }
    printf("  PASS test3 (V=%d k=%d): exactly 3 accepted\n", V, k);
    return true;
}

// CPU double-precision log-sum-exp for reference.
static double cpu_lse_h(const __half* row, int V) {
    double mx = -1e300;
    for (int v = 0; v < V; ++v) mx = std::max(mx, (double)__half2float(row[v]));
    double s = 0.0;
    for (int v = 0; v < V; ++v) s += std::exp((double)__half2float(row[v]) - mx);
    return mx + std::log(s);
}

// KL divergence D_KL(p||q) = sum_v p_v log(p_v / q_v).
static double kl_divergence(const std::vector<double>& p, const std::vector<double>& q) {
    double kl = 0.0;
    for (size_t v = 0; v < p.size(); ++v) {
        if (p[v] > 1e-10) kl += p[v] * (std::log(p[v]) - std::log(q[v] + 1e-30));
    }
    return kl;
}

// Test 4: statistical distribution test (output matches target distribution).
// Runs N verifications and checks token histogram KL divergence < threshold.
static bool test_statistical_distribution(int V, int k, int N = 10000) {
    // Use a small vocab subset for tractability.
    const int Vsub = std::min(V, 128);

    TestInstance t;
    t.cfg = {Vsub, k, k};
    t.alloc();

    // Fixed random-ish target logits; draft = uniform.
    srand(12345 + V + k);
    fill_uniform_logits(t.h_draft_logits.data(), k, Vsub);
    for (size_t i = 0; i < (size_t)(k+1)*Vsub; ++i) {
        t.h_target_logits[i] = __float2half(((float)rand()/RAND_MAX) * 6.0f - 3.0f);
    }

    // Compute reference target distribution at position k (bonus position).
    double target_lse = cpu_lse_h(t.h_target_logits.data() + (size_t)k * Vsub, Vsub);
    std::vector<double> p_target(Vsub);
    for (int v = 0; v < Vsub; ++v) {
        p_target[v] = std::exp((double)__half2float(t.h_target_logits[k * Vsub + v]) - target_lse);
    }

    // Force all k draft tokens to accept: target == draft at each position.
    for (int pos = 0; pos < k; ++pos) {
        for (int v = 0; v < Vsub; ++v) {
            t.h_draft_logits[pos * Vsub + v] = t.h_target_logits[pos * Vsub + v];
        }
    }
    for (int i = 0; i < k; ++i) t.h_draft_tokens[i] = 0;

    // Upload fixed logits; vary only the last random value (bonus token).
    // random_values[0..k-1] = 0 (always accept); random_values[k] = varied.
    for (int i = 0; i < k; ++i) t.h_random_values[i] = 0.0f;

    std::vector<int> histogram(Vsub, 0);

    for (int trial = 0; trial < N; ++trial) {
        // Sample a uniform random value for the bonus position.
        t.h_random_values[k] = (float)(trial + 0.5f) / N;
        t.upload();
        VerifyResult res = t.run_fused();
        if (res.num_accepted == k) {
            int bonus = res.output_tokens[k];
            if (bonus >= 0 && bonus < Vsub) histogram[bonus]++;
        }
    }

    // Compute empirical distribution.
    std::vector<double> p_empirical(Vsub);
    double total = std::accumulate(histogram.begin(), histogram.end(), 0.0);
    for (int v = 0; v < Vsub; ++v) p_empirical[v] = histogram[v] / total;

    double kl = kl_divergence(p_target, p_empirical);
    t.free_all();

    bool pass = (kl < 0.05);
    printf("  %s test4 (V=%d k=%d N=%d): KL divergence = %.4f (threshold 0.05)\n",
           pass ? "PASS" : "FAIL", Vsub, k, N, kl);
    return pass;
}

int main() {
    printf("=== test_correctness ===\n");
    bool all_pass = true;

    // Test 1: deterministic accept.
    for (int V : {32000, 128256}) {
        for (int k : {1, 4, 8}) {
            all_pass &= test_deterministic_accept(V, k);
        }
    }

    // Test 2: deterministic reject.
    for (int V : {32000, 128256}) {
        for (int k : {1, 4, 8}) {
            all_pass &= test_deterministic_reject(V, k);
        }
    }

    // Test 3: partial accept.
    for (int V : {32000, 128256}) {
        for (int k : {4, 8}) {
            all_pass &= test_partial_accept(V, k);
        }
    }

    // Test 4: statistical distribution.
    for (int V : {32000, 128256}) {
        for (int k : {1, 4, 8}) {
            all_pass &= test_statistical_distribution(V, k, 10000);
        }
    }

    printf("%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
