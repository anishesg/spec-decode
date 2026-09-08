#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "logsumexp.cuh"

// Result of the sequential acceptance chain evaluation.
struct AcceptChainResult {
    int num_accepted;   // number of draft tokens accepted (0..k)
    int rejection_pos;  // position of first rejection (== num_accepted; k if all accepted)
};

// Sequential acceptance chain executed by a single warp.
//
// Lane 0 iterates through positions 0..k-1. For each position:
//   - Loads target_logit[pos][draft_token[pos]] and draft_logit[pos][draft_token[pos]].
//   - Computes log acceptance ratio = min(0, log_p_target - log_p_draft).
//   - Tests exp(log_ratio) >= random[pos]; decides accept or reject.
//   - Broadcasts decision to all lanes via __shfl_sync.
//   - On first rejection, breaks out of the loop.
//
// All lanes receive the final AcceptChainResult via shuffle at the end.
//
// Preconditions:
//   - Called with all 32 lanes of a warp active.
//   - target_lse[pos] = log-sum-exp of target_logits row pos (precomputed).
//   - draft_lse[pos]  = log-sum-exp of draft_logits row pos (precomputed).
//   - All arrays are in global or shared memory, accessible by all lanes.
__device__ __forceinline__ AcceptChainResult run_accept_chain(
        const __half* __restrict__ target_logits,  // [(k+1) x vocab_size]
        const __half* __restrict__ draft_logits,   // [k x vocab_size]
        const int32_t* __restrict__ draft_tokens,  // [k]
        const float* __restrict__ random_values,   // [k+1]
        const float* __restrict__ target_lse,      // [k+1] precomputed log-sum-exp
        const float* __restrict__ draft_lse,       // [k] precomputed log-sum-exp
        int k,
        int vocab_size)
{
    const int lane = threadIdx.x & 31;

    int num_accepted = 0;
    int rejection_pos = k;  // default: all accepted

    // Lane 0 drives the sequential chain; all lanes receive decisions via broadcast.
    for (int pos = 0; pos < k; ++pos) {
        int accept = 0;

        if (lane == 0) {
            int tok = draft_tokens[pos];
            float tgt_logit  = __half2float(target_logits[pos * vocab_size + tok]);
            float dft_logit  = __half2float(draft_logits[pos * vocab_size + tok]);
            float log_p_tgt  = tgt_logit - target_lse[pos];
            float log_p_dft  = dft_logit - draft_lse[pos];

            // Auto-reject if draft assigns zero (or near-zero) probability.
            if (log_p_dft < -87.0f) {  // exp(-87) ~ FLT_MIN
                accept = 0;
            } else {
                // Auto-reject if target assigns zero probability.
                if (log_p_tgt < -87.0f) {
                    accept = 0;
                } else {
                    float log_ratio   = fminf(0.0f, log_p_tgt - log_p_dft);
                    float accept_prob = __expf(log_ratio);
                    accept = (random_values[pos] < accept_prob) ? 1 : 0;
                }
            }
        }

        // Broadcast lane-0 decision to all lanes.
        accept = __shfl_sync(FULL_MASK, accept, 0);

        if (accept) {
            ++num_accepted;
        } else {
            rejection_pos = pos;
            break;
        }
    }

    AcceptChainResult res;
    res.num_accepted  = num_accepted;
    res.rejection_pos = rejection_pos;
    return res;
}
