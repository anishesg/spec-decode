#pragma once
#include "types.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Maximum tree dimensions.
static constexpr int MAX_TREE_NODES = 1024;  // depth 16 x width 64
static constexpr int MAX_TREE_DEPTH = 16;
static constexpr int MAX_TREE_WIDTH = 64;    // max leaves / max paths

// Extends SpecConfig with tree topology constraints.
struct TreeDraftConfig {
    SpecConfig base;    // vocab_size, max_draft_len, num_draft_tokens (= node_count)
    int max_tree_width; // maximum number of parallel paths (leaves)
    int max_tree_depth; // maximum path length (tree depth)
};

// A tree of draft tokens where each node may have multiple children.
// Logit and token arrays are indexed by node ID (0-indexed).
// Node 0 is the root; root's parent is -1.
struct TreeDraft {
    int      node_count;                       // total nodes in tree
    int32_t* parent_ids;                       // [node_count] parent of each node (-1 for root)
    int32_t* depth;                            // [node_count] depth of each node (root = 0)
    int32_t* token_ids;                        // [node_count] draft token at each node
    __half*  target_logits;                    // [node_count x vocab_size] target logits per node
    __half*  draft_logits;                     // [node_count x vocab_size] draft logits per node
    float*   random_values;                    // [node_count] random value per node
};

// Result of tree verification: the longest accepted root-to-leaf path.
struct TreeVerifyResult {
    int     path_length;                       // number of accepted tokens
    int32_t path_tokens[MAX_TREE_DEPTH + 1];   // accepted tokens + correction token
    float   path_log_probs[MAX_TREE_DEPTH + 1];
};

// Utility: enumerate all root-to-leaf paths in a tree given parent_ids.
// Fills path_nodes[path_idx][depth] with node IDs.
// Returns number of paths found (number of leaves).
// path_nodes must be pre-allocated: [MAX_TREE_WIDTH][MAX_TREE_DEPTH].
int enumerate_paths(const int32_t* parent_ids,
                    const int32_t* depth_arr,
                    int            node_count,
                    int            path_nodes[MAX_TREE_WIDTH][MAX_TREE_DEPTH],
                    int            path_lengths[MAX_TREE_WIDTH]);
