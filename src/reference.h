#pragma once
#include "types.h"
#include <cuda_fp16.h>

// CPU reference implementation of speculative decoding verification.
// Input pointers must be host-side (accessible from CPU).
VerifyResult verify_reference(const SpecConfig& cfg,
                               const DraftResult& d,
                               const __half*  h_target_logits,
                               const __half*  h_draft_logits,
                               const int32_t* h_draft_tokens,
                               const float*   h_random_values);
