#include "fused_verify.cuh"
#include "logsumexp.cuh"
#include "accept_chain.cuh"
#include "residual_sample.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_err)); \
            abort(); \
        } \
    } while(0)

// Shared memory layout offsets (in floats / ints).
struct SmemLayout {
    int target_lse_off;   // float [k+1]
    int draft_lse_off;    // float [k]
    int draft_tok_off;    // int32 [k]
    int rand_val_off;     // float [k+1]
    int total_floats;     // total smem size in float units
};

static SmemLayout make_layout(int k) {
    SmemLayout l;
    l.target_lse_off = 0;
    l.draft_lse_off  = l.target_lse_off + (k + 1);
    l.draft_tok_off  = l.draft_lse_off  + k;
    l.rand_val_off   = l.draft_tok_off  + k;    // int32 same size as float
    l.total_floats   = l.rand_val_off   + (k + 1);
    return l;
}

int fused_verify_smem_bytes(const SpecConfig& cfg) {
    return make_layout(cfg.num_draft_tokens).total_floats * sizeof(float);
}

// Fused kernel: one block per verification instance.
// Block size = 64 threads (2 warps).
// Warp 0 = threads 0..31, Warp 1 = threads 32..63.
__global__ void fused_verify_kernel(
        const DraftResult* __restrict__ inputs,
        VerifyResult*      __restrict__ outputs,
        int k,
        int vocab_size,
        SmemLayout layout)
{
    extern __shared__ float smem[];

    const int bid   = blockIdx.x;
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;

    const DraftResult& inp = inputs[bid];

    float* target_lse = smem + layout.target_lse_off;
    float* draft_lse  = smem + layout.draft_lse_off;
    int*   draft_toks = reinterpret_cast<int*>(smem + layout.draft_tok_off);
    float* rand_vals  = smem + layout.rand_val_off;

    // Load draft tokens and random values into shared memory (warp 0, lane by lane).
    if (warp == 0) {
        for (int i = lane; i < k; i += 32) {
            draft_toks[i] = inp.draft_tokens[i];
        }
        for (int i = lane; i <= k; i += 32) {
            rand_vals[i] = inp.random_values[i];
        }
    }

    // Warp 0: compute target log-sum-exp for positions 0..k.
    // Warp 1: compute draft  log-sum-exp for positions 0..k-1.
    // Warps operate independently on different positions (no sync needed here).
    if (warp == 0) {
        for (int pos = lane / 32; pos <= k; pos += 1) {
            // Each warp computes one position at a time sequentially.
            // Use all 32 lanes of warp 0 for the vocab reduction.
            // Reassign: every thread in warp 0 participates in each position.
            // We loop positions one-by-one; all 32 lanes of warp 0 cooperate.
            break;  // handled below
        }
        for (int pos = 0; pos <= k; ++pos) {
            float lse = warp_log_sum_exp(
                    inp.target_logits + (size_t)pos * vocab_size, vocab_size);
            if (lane == 0) target_lse[pos] = lse;
        }
    } else {
        // warp 1
        for (int pos = 0; pos < k; ++pos) {
            float lse = warp_log_sum_exp(
                    inp.draft_logits + (size_t)pos * vocab_size, vocab_size);
            if (lane == 0) draft_lse[pos] = lse;
        }
    }

    __syncthreads();

    // Phase 2: warp 0 runs the acceptance chain.
    AcceptChainResult chain{};
    if (warp == 0) {
        chain = run_accept_chain(
                inp.target_logits,
                inp.draft_logits,
                draft_toks,
                rand_vals,
                target_lse,
                draft_lse,
                k, vocab_size);
    }

    // Broadcast chain result from warp 0 lane 0 to all threads.
    int num_accepted_broad  = __shfl_sync(FULL_MASK, (warp == 0) ? chain.num_accepted  : 0, 0);
    int rejection_pos_broad = __shfl_sync(FULL_MASK, (warp == 0) ? chain.rejection_pos : 0, 0);
    // Cross-warp broadcast via shared memory.
    __shared__ int s_num_accepted, s_rejection_pos;
    if (threadIdx.x == 0) {
        s_num_accepted  = chain.num_accepted;
        s_rejection_pos = chain.rejection_pos;
    }
    __syncthreads();
    int num_accepted  = s_num_accepted;
    int rejection_pos = s_rejection_pos;
    (void)num_accepted_broad;
    (void)rejection_pos_broad;

    // Phase 3: correction or bonus sampling (all warps cooperate via warp 0).
    int sampled_tok = 0;
    float sampled_lp = 0.0f;

    if (warp == 0) {
        if (rejection_pos < k) {
            // Correction sampling from residual at rejection_pos.
            float rv = rand_vals[k];  // use last random value for correction
            sampled_tok = warp_correction_sample(
                    inp.target_logits + (size_t)rejection_pos * vocab_size,
                    inp.draft_logits  + (size_t)rejection_pos * vocab_size,
                    target_lse[rejection_pos],
                    draft_lse[rejection_pos],
                    rv, vocab_size);
        } else {
            // Bonus token from target[k].
            float rv = rand_vals[k];
            sampled_tok = warp_bonus_sample(
                    inp.target_logits + (size_t)k * vocab_size,
                    target_lse[k],
                    rv, vocab_size);
        }
        // Compute log prob of sampled token.
        if (lane == 0) {
            int spos = (rejection_pos < k) ? rejection_pos : k;
            sampled_lp = __half2float(inp.target_logits[spos * vocab_size + sampled_tok])
                         - target_lse[spos];
        }
    }

    // Write output from thread 0 only.
    if (threadIdx.x == 0) {
        VerifyResult& out = outputs[bid];
        out.num_accepted = num_accepted;

        // Copy accepted tokens from draft.
        for (int i = 0; i < num_accepted; ++i) {
            int tok = draft_toks[i];
            out.output_tokens[i] = tok;
            out.log_probs[i]     = __half2float(inp.target_logits[i * vocab_size + tok])
                                    - target_lse[i];
        }
        out.output_tokens[num_accepted] = sampled_tok;
        out.log_probs[num_accepted]     = sampled_lp;
    }
}

void host_fused_verify(const SpecConfig&  cfg,
                        const DraftResult* d_inputs,
                        VerifyResult*      d_outputs,
                        int                batch_size,
                        cudaStream_t       stream)
{
    const int k           = cfg.num_draft_tokens;
    const int vocab_size  = cfg.vocab_size;
    SmemLayout layout     = make_layout(k);
    int smem_bytes        = layout.total_floats * (int)sizeof(float)
                            + 2 * (int)sizeof(int);  // s_num_accepted, s_rejection_pos

    // 64 threads per block (2 warps).
    const int threads_per_block = 64;
    fused_verify_kernel<<<batch_size, threads_per_block, smem_bytes, stream>>>(
            d_inputs, d_outputs, k, vocab_size, layout);
}
