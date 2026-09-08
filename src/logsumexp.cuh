#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

// Full warp mask.
static constexpr unsigned FULL_MASK = 0xffffffffu;

// Warp-level max reduction via butterfly shuffle.
__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, 16));
    val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val,  8));
    val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val,  4));
    val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val,  2));
    val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val,  1));
    return val;
}

// Warp-level sum reduction via butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(FULL_MASK, val, 16);
    val += __shfl_xor_sync(FULL_MASK, val,  8);
    val += __shfl_xor_sync(FULL_MASK, val,  4);
    val += __shfl_xor_sync(FULL_MASK, val,  2);
    val += __shfl_xor_sync(FULL_MASK, val,  1);
    return val;
}

// Warp-cooperative log-sum-exp over a half-precision logit row of length vocab_size.
//
// Each warp lane streams through [row_start, row_start + vocab_size) in stride-32 order.
// Phase 1: find global max via two-pass running max + butterfly reduction.
// Phase 2: sum exp(x - max) via lane accumulation + butterfly sum reduction.
//
// Returns: log(sum_v exp(logit[v])) broadcast to all 32 lanes.
// Also writes per-lane partial results if out_max/out_sum are non-null (used by fused kernel).
//
// Handles vocab_size up to 256K (8192 iterations per lane at 32 lanes).
__device__ __forceinline__ float warp_log_sum_exp(
        const __half* __restrict__ logits,
        int vocab_size,
        float* __restrict__ out_max = nullptr,
        float* __restrict__ out_sum = nullptr)
{
    const int lane = threadIdx.x & 31;

    // Phase 1: find max over all vocab entries.
    float lane_max = -FLT_MAX;
    for (int v = lane; v < vocab_size; v += 32) {
        float x = __half2float(logits[v]);
        lane_max = fmaxf(lane_max, x);
    }
    float global_max = warp_reduce_max(lane_max);

    // Phase 2: sum exp(x - global_max) over all vocab entries.
    float lane_sum = 0.0f;
    for (int v = lane; v < vocab_size; v += 32) {
        float x = __half2float(logits[v]);
        lane_sum += __expf(x - global_max);
    }
    float global_sum = warp_reduce_sum(lane_sum);

    if (out_max) *out_max = global_max;
    if (out_sum) *out_sum = global_sum;

    return global_max + __logf(global_sum);
}

// Variant for float32 logits (used in benchmarks).
__device__ __forceinline__ float warp_log_sum_exp_f32(
        const float* __restrict__ logits,
        int vocab_size)
{
    const int lane = threadIdx.x & 31;

    float lane_max = -FLT_MAX;
    for (int v = lane; v < vocab_size; v += 32) {
        lane_max = fmaxf(lane_max, logits[v]);
    }
    float global_max = warp_reduce_max(lane_max);

    float lane_sum = 0.0f;
    for (int v = lane; v < vocab_size; v += 32) {
        lane_sum += __expf(logits[v] - global_max);
    }
    float global_sum = warp_reduce_sum(lane_sum);

    return global_max + __logf(global_sum);
}
