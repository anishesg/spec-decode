#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "logsumexp.cuh"

// Warp-cooperative correction sampling from residual distribution:
//   q(v) = max(0, p_target(v) - p_draft(v)), normalized.
//
// Pass 1: all lanes stream through vocab in stride-32, accumulate per-lane partial
//         residual sums, then butterfly-reduce to get total residual mass.
// Pass 2: all lanes stream through vocab again in stride-32, accumulating a running
//         cumulative sum of residuals. Each lane checks if its cumulative segment
//         crosses (rand_val * total_residual). __ballot_sync finds the first crossing
//         lane; __ffs extracts its index; that lane's last-seen token is the sample.
//
// Also handles the bonus token case when all drafts are accepted: sampling from
// p_target directly using the same two-pass approach.
//
// Returns the sampled token ID, broadcast to all 32 lanes.
__device__ __forceinline__ int warp_correction_sample(
        const __half* __restrict__ target_logits,  // row ptr: [vocab_size] for rejected pos
        const __half* __restrict__ draft_logits,   // row ptr: [vocab_size] for rejected pos
        float target_lse,
        float draft_lse,
        float rand_val,
        int vocab_size)
{
    const int lane = threadIdx.x & 31;

    // Pass 1: accumulate total residual mass.
    float lane_residual = 0.0f;
    for (int v = lane; v < vocab_size; v += 32) {
        float p_t = __expf(__half2float(target_logits[v]) - target_lse);
        float p_d = __expf(__half2float(draft_logits[v])  - draft_lse);
        float r   = fmaxf(0.0f, p_t - p_d);
        lane_residual += r;
    }
    float total_residual = warp_reduce_sum(lane_residual);

    float threshold = rand_val * total_residual;

    // Pass 2: warp-stride cumulative scan to find the crossing entry.
    // Each lane accumulates its own partial cumsum. We use a cross-lane prefix
    // to know the cumsum at the start of each lane's segment.
    //
    // Strategy: iterate in rounds of 32 vocabulary entries per round.
    // After each round, compute lane prefix sums so every lane knows its global
    // offset into the cumulative sum, then check for crossing.

    float global_offset = 0.0f;  // cumsum before current round
    int sampled_tok     = vocab_size - 1;  // fallback

    for (int v_base = 0; v_base < vocab_size; v_base += 32) {
        int v = v_base + lane;
        float r = 0.0f;
        if (v < vocab_size) {
            float p_t = __expf(__half2float(target_logits[v]) - target_lse);
            float p_d = __expf(__half2float(draft_logits[v])  - draft_lse);
            r = fmaxf(0.0f, p_t - p_d);
        }

        // Exclusive prefix sum across the warp for this round.
        // Use inclusive scan first, then shift.
        float inc = r;
        inc += __shfl_up_sync(FULL_MASK, inc, 1);
        inc += __shfl_up_sync(FULL_MASK, inc, 2);
        inc += __shfl_up_sync(FULL_MASK, inc, 4);
        inc += __shfl_up_sync(FULL_MASK, inc, 8);
        inc += __shfl_up_sync(FULL_MASK, inc, 16);
        // Exclusive: subtract own contribution.
        float excl = inc - r;
        float cumsum_here = global_offset + excl + r;

        // Does this lane's entry cross the threshold?
        int crossed = (v < vocab_size) && (cumsum_here >= threshold) ? 1 : 0;
        unsigned ballot = __ballot_sync(FULL_MASK, crossed);
        if (ballot) {
            // First crossing lane.
            int first_lane = __ffs((int)ballot) - 1;  // __ffs returns 1-indexed
            if (lane == first_lane && v < vocab_size) {
                sampled_tok = v;
            }
            // Broadcast to all lanes.
            sampled_tok = __shfl_sync(FULL_MASK, sampled_tok, first_lane);
            return sampled_tok;
        }

        // Advance global offset by sum of this round.
        float round_sum = __shfl_sync(FULL_MASK, inc, 31);
        global_offset += round_sum;
    }

    // Fallback: return last vocab entry (handles rounding edge cases).
    return sampled_tok;
}

// Bonus token sampling from target distribution directly (all-accepted case).
// Same two-pass CDF approach without the draft subtraction.
__device__ __forceinline__ int warp_bonus_sample(
        const __half* __restrict__ target_logits,  // row ptr for position k
        float target_lse,
        float rand_val,
        int vocab_size)
{
    const int lane = threadIdx.x & 31;

    float global_offset = 0.0f;
    int sampled_tok     = vocab_size - 1;

    for (int v_base = 0; v_base < vocab_size; v_base += 32) {
        int v = v_base + lane;
        float p = 0.0f;
        if (v < vocab_size) {
            p = __expf(__half2float(target_logits[v]) - target_lse);
        }

        float inc = p;
        inc += __shfl_up_sync(FULL_MASK, inc, 1);
        inc += __shfl_up_sync(FULL_MASK, inc, 2);
        inc += __shfl_up_sync(FULL_MASK, inc, 4);
        inc += __shfl_up_sync(FULL_MASK, inc, 8);
        inc += __shfl_up_sync(FULL_MASK, inc, 16);
        float excl = inc - p;
        float cumsum_here = global_offset + excl + p;

        int crossed = (v < vocab_size) && (cumsum_here >= rand_val) ? 1 : 0;
        unsigned ballot = __ballot_sync(FULL_MASK, crossed);
        if (ballot) {
            int first_lane = __ffs((int)ballot) - 1;
            if (lane == first_lane && v < vocab_size) {
                sampled_tok = v;
            }
            sampled_tok = __shfl_sync(FULL_MASK, sampled_tok, first_lane);
            return sampled_tok;
        }

        float round_sum = __shfl_sync(FULL_MASK, inc, 31);
        global_offset += round_sum;
    }

    return sampled_tok;
}
