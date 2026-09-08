#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Configuration for a single speculative decoding verification instance.
struct SpecConfig {
    int vocab_size;        // V: vocabulary size
    int max_draft_len;     // maximum k supported by the allocations
    int num_draft_tokens;  // k: actual draft tokens in this instance (<= max_draft_len)
};

// Input to the verification kernel.
// All device pointers, row-major layout.
struct DraftResult {
    int32_t*  draft_tokens;    // [k] token IDs proposed by draft model
    __half*   target_logits;   // [(k+1) x vocab_size] target model logits, row-major
    __half*   draft_logits;    // [k x vocab_size] draft model logits, row-major
    float*    random_values;   // [k+1] uniform samples in [0, 1) for acceptance + bonus
};

// Output from the verification kernel.
struct VerifyResult {
    int32_t  num_accepted;      // number of draft tokens accepted (0..k)
    int32_t  output_tokens[17]; // accepted tokens + 1 correction/bonus token (max k+1 = 17)
    float    log_probs[17];     // log probability under target for each output token
};

// Inline device helper: row-major index into a [rows x cols] matrix.
__device__ __forceinline__ int logit_idx(int row, int col, int cols) {
    return row * cols + col;
}

// Bounds-checked logit access (only active in debug builds).
__device__ __forceinline__ float load_logit_f32(const __half* __restrict__ logits,
                                                 int row, int col, int cols) {
#ifdef DEBUG_BOUNDS
    assert(col >= 0 && col < cols);
#endif
    return __half2float(logits[row * cols + col]);
}
