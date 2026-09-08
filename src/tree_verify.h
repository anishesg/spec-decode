#pragma once
#include "tree_types.h"
#include <cuda_runtime.h>

// Launch the fused tree verification kernel.
void host_tree_verify(const TreeDraftConfig& cfg,
                       const TreeDraft*       d_inputs,
                       TreeVerifyResult*      d_outputs,
                       int                    batch_size,
                       const int*             d_path_nodes,
                       const int*             d_path_lengths,
                       int                    num_paths,
                       cudaStream_t           stream = 0);
