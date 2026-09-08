#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/types.h"
#include "../src/fused_verify.cuh"
#include "../src/reference.h"
#include "../src/logsumexp.cuh"
#include "../src/accept_chain.cuh"
#include "../src/residual_sample.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_err)); \
            exit(1); \
        } \
    } while(0)

// Multi-kernel baseline: three separate kernel launches with global memory intermediates.
// Kernel 1: compute log-sum-exp for all positions (one warp per position).
// Kernel 2: compute acceptance chain results (one warp per batch element).
// Kernel 3: perform correction/bonus sampling (one warp per batch element).

__global__ void mk_logsumexp_kernel(const DraftResult* __restrict__ inputs,
                                     float* __restrict__ target_lse,  // [batch * (k+1)]
                                     float* __restrict__ draft_lse,   // [batch * k]
                                     int k, int V) {
    // One warp per (batch, pos, type) triple.
    // type=0: target pos 0..k, type=1: draft pos 0..k-1.
    int total_target = gridDim.x * (k + 1);
    (void)total_target;
    int bid = blockIdx.x;
    int pos = blockIdx.y;
    int type = blockIdx.z;  // 0=target, 1=draft

    if (type == 0 && pos > k) return;
    if (type == 1 && pos >= k) return;

    const __half* logit_row = (type == 0)
        ? inputs[bid].target_logits + (size_t)pos * V
        : inputs[bid].draft_logits  + (size_t)pos * V;

    float lse = warp_log_sum_exp(logit_row, V);

    if ((threadIdx.x & 31) == 0) {
        if (type == 0) target_lse[bid * (k+1) + pos] = lse;
        else           draft_lse[bid * k + pos]       = lse;
    }
}

__global__ void mk_accept_chain_kernel(const DraftResult* __restrict__ inputs,
                                        float* __restrict__ target_lse,
                                        float* __restrict__ draft_lse,
                                        int* __restrict__ num_accepted_out,
                                        int* __restrict__ rejection_pos_out,
                                        int k, int V) {
    int bid = blockIdx.x;
    AcceptChainResult res = run_accept_chain(
            inputs[bid].target_logits,
            inputs[bid].draft_logits,
            inputs[bid].draft_tokens,
            inputs[bid].random_values,
            target_lse + bid * (k+1),
            draft_lse  + bid * k,
            k, V);
    if ((threadIdx.x & 31) == 0) {
        num_accepted_out[bid]  = res.num_accepted;
        rejection_pos_out[bid] = res.rejection_pos;
    }
}

__global__ void mk_sample_kernel(const DraftResult* __restrict__ inputs,
                                  float* __restrict__ target_lse,
                                  float* __restrict__ draft_lse,
                                  const int* __restrict__ num_accepted,
                                  const int* __restrict__ rejection_pos,
                                  VerifyResult* __restrict__ outputs,
                                  int k, int V) {
    int bid = blockIdx.x;
    int lane = threadIdx.x & 31;

    int na  = num_accepted[bid];
    int rp  = rejection_pos[bid];
    int spos = (rp < k) ? rp : k;

    int tok = 0;
    if (rp < k) {
        tok = warp_correction_sample(
                inputs[bid].target_logits + (size_t)spos * V,
                inputs[bid].draft_logits  + (size_t)spos * V,
                target_lse[bid * (k+1) + spos],
                draft_lse[bid * k + spos],
                inputs[bid].random_values[k], V);
    } else {
        tok = warp_bonus_sample(
                inputs[bid].target_logits + (size_t)k * V,
                target_lse[bid * (k+1) + k],
                inputs[bid].random_values[k], V);
    }

    if (lane == 0) {
        VerifyResult& out = outputs[bid];
        out.num_accepted = na;
        for (int i = 0; i < na; ++i) {
            out.output_tokens[i] = inputs[bid].draft_tokens[i];
        }
        out.output_tokens[na] = tok;
        float tlse = target_lse[bid * (k+1) + spos];
        out.log_probs[na] = __half2float(
                inputs[bid].target_logits[spos * V + tok]) - tlse;
    }
}

// Timing result for one configuration.
struct BenchResult {
    int vocab_size, draft_len, batch_size;
    float fused_us;
    float cpu_us;
    float multikernel_us;
};

static float cuda_event_elapsed(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms * 1000.0f;  // microseconds
}

static BenchResult bench_config(int V, int k, int batch) {
    BenchResult br;
    br.vocab_size = V;
    br.draft_len  = k;
    br.batch_size = batch;

    size_t tgt_sz = (size_t)(k+1) * V;
    size_t dft_sz = (size_t)k     * V;

    // Allocate per-element device buffers and host buffers.
    std::vector<__half*>   d_tgt(batch), d_dft(batch);
    std::vector<int32_t*>  d_tok(batch);
    std::vector<float*>    d_rv(batch);
    std::vector<DraftResult> h_drs(batch);

    for (int b = 0; b < batch; ++b) {
        CUDA_CHECK(cudaMalloc(&d_tgt[b], tgt_sz * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_dft[b], dft_sz * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_tok[b], k * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_rv[b],  (k+1) * sizeof(float)));

        // Fill with pseudo-random data.
        std::vector<__half> h_tgt(tgt_sz), h_dft(dft_sz);
        std::vector<int32_t> h_tok(k);
        std::vector<float> h_rv(k+1);
        srand(b + V + k);
        for (auto& x : h_tgt) x = __float2half(((float)rand()/RAND_MAX)*8-4);
        for (auto& x : h_dft) x = __float2half(((float)rand()/RAND_MAX)*8-4);
        for (auto& x : h_tok) x = rand() % V;
        for (auto& x : h_rv)  x = (float)rand()/RAND_MAX;

        CUDA_CHECK(cudaMemcpy(d_tgt[b], h_tgt.data(), tgt_sz*sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dft[b], h_dft.data(), dft_sz*sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tok[b], h_tok.data(), k*sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rv[b],  h_rv.data(),  (k+1)*sizeof(float), cudaMemcpyHostToDevice));

        h_drs[b] = {d_tok[b], d_tgt[b], d_dft[b], d_rv[b]};
    }

    DraftResult* d_inputs;
    VerifyResult* d_outputs;
    CUDA_CHECK(cudaMalloc(&d_inputs,  batch * sizeof(DraftResult)));
    CUDA_CHECK(cudaMalloc(&d_outputs, batch * sizeof(VerifyResult)));
    CUDA_CHECK(cudaMemcpy(d_inputs, h_drs.data(), batch*sizeof(DraftResult), cudaMemcpyHostToDevice));

    // Global memory intermediates for multi-kernel path.
    float *d_tlse, *d_dlse;
    int   *d_na, *d_rp;
    CUDA_CHECK(cudaMalloc(&d_tlse, (size_t)batch*(k+1)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dlse, (size_t)batch*k*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_na,   batch*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rp,   batch*sizeof(int)));

    SpecConfig cfg{V, k, k};
    const int WARMUP = 10, ITERS = 100;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // --- Fused kernel ---
    for (int i = 0; i < WARMUP; ++i)
        host_fused_verify(cfg, d_inputs, d_outputs, batch);
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < ITERS; ++i)
        host_fused_verify(cfg, d_inputs, d_outputs, batch);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    br.fused_us = cuda_event_elapsed(ev_start, ev_stop) / ITERS;

    // --- Multi-kernel baseline ---
    // Three launches with global memory intermediates.
    dim3 lse_grid(batch, k+2, 2);  // over-provisioned, kernels guard themselves
    auto run_mk = [&]() {
        mk_logsumexp_kernel<<<lse_grid, 32>>>(d_inputs, d_tlse, d_dlse, k, V);
        mk_accept_chain_kernel<<<batch, 32>>>(d_inputs, d_tlse, d_dlse, d_na, d_rp, k, V);
        mk_sample_kernel<<<batch, 32>>>(d_inputs, d_tlse, d_dlse, d_na, d_rp, d_outputs, k, V);
    };
    for (int i = 0; i < WARMUP; ++i) { run_mk(); CUDA_CHECK(cudaDeviceSynchronize()); }
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < ITERS; ++i) run_mk();
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    br.multikernel_us = cuda_event_elapsed(ev_start, ev_stop) / ITERS;

    // --- CPU round-trip simulation ---
    // Pinned host buffers for async copies.
    __half* h_tgt_pin; float* h_rv_pin;
    CUDA_CHECK(cudaMallocHost(&h_tgt_pin, (size_t)batch*tgt_sz*sizeof(__half)));
    CUDA_CHECK(cudaMallocHost(&h_rv_pin,  (size_t)batch*(k+1)*sizeof(float)));

    // Stage: D2H for target logits + draft logits, CPU verify, H2D for result tokens.
    // We time only the round-trip (memcpy + sync), not the CPU computation itself,
    // since that's what adds to decode latency.
    size_t transfer_bytes = (size_t)batch * (tgt_sz + dft_sz) * sizeof(__half);
    __half* d_tgt_flat;
    CUDA_CHECK(cudaMalloc(&d_tgt_flat, transfer_bytes));

    for (int i = 0; i < WARMUP; ++i) {
        CUDA_CHECK(cudaMemcpy(h_tgt_pin, d_tgt_flat, transfer_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_tgt_flat, h_tgt_pin, transfer_bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemcpy(h_tgt_pin, d_tgt_flat, transfer_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_tgt_flat, h_tgt_pin, batch*(k+1)*sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    br.cpu_us = cuda_event_elapsed(ev_start, ev_stop) / ITERS;

    // Cleanup.
    CUDA_CHECK(cudaFreeHost(h_tgt_pin));
    CUDA_CHECK(cudaFreeHost(h_rv_pin));
    CUDA_CHECK(cudaFree(d_tgt_flat));
    CUDA_CHECK(cudaFree(d_tlse)); CUDA_CHECK(cudaFree(d_dlse));
    CUDA_CHECK(cudaFree(d_na));   CUDA_CHECK(cudaFree(d_rp));
    CUDA_CHECK(cudaFree(d_inputs)); CUDA_CHECK(cudaFree(d_outputs));
    for (int b = 0; b < batch; ++b) {
        CUDA_CHECK(cudaFree(d_tgt[b])); CUDA_CHECK(cudaFree(d_dft[b]));
        CUDA_CHECK(cudaFree(d_tok[b])); CUDA_CHECK(cudaFree(d_rv[b]));
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return br;
}

int main() {
    const float BASE_DECODE_US = 300.0f;

    printf("%-10s %-10s %-10s  %10s %10s %10s  %8s %8s  %s\n",
           "vocab", "draft_len", "batch",
           "fused(us)", "cpu(us)", "mkern(us)",
           "spdup_cpu", "spdup_mk",
           "decode_overhead_fused");
    printf("%s\n", std::string(100, '-').c_str());

    for (int V    : {32000, 65536, 128256}) {
        for (int k : {1, 2, 4, 8, 16}) {
            for (int bs : {1, 8, 32}) {
                BenchResult r = bench_config(V, k, bs);
                float spdup_cpu = r.cpu_us       / r.fused_us;
                float spdup_mk  = r.multikernel_us / r.fused_us;
                float overhead  = r.fused_us / BASE_DECODE_US * 100.0f;
                printf("%-10d %-10d %-10d  %10.2f %10.2f %10.2f  %8.1fx %8.1fx  %.2f%%\n",
                       r.vocab_size, r.draft_len, r.batch_size,
                       r.fused_us, r.cpu_us, r.multikernel_us,
                       spdup_cpu, spdup_mk, overhead);
            }
        }
    }

    return 0;
}
