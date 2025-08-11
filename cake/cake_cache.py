import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from transformers.cache_utils import DynamicCache, Cache, HybridCache
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import json

from cake.utils import adjust_budgets, compute_head_budgets, compute_head_budgets_dynamic
from datetime import datetime


class CakeCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.pref_scores = []
        self.evict_scores = []
        self.layer_budget = []
        self.head_budgets = None  # Will store pre-computed head budgets
        self.turn_off_eviction = True  # Default to True, will be set by prefill logic
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `CakeCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_score(
        self,
        pref_score: torch.Tensor,
        evict_score: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ):
        self.pref_scores.append(pref_score)
        self.evict_scores.append(evict_score)

    def initialize_budgets(self, head_budgets: Dict[int, List[int]]):
        """Initialize pre-computed budgets for static allocation"""
        self.head_budgets = head_budgets
        self.layer_budget = []
        for layer_idx in sorted(head_budgets.keys()):
            layer_budget = sum(head_budgets[layer_idx])
            self.layer_budget.append(layer_budget)
        # print(f"[CAKE] Initialized budgets for {len(self.layer_budget)} layers: {self.layer_budget}")
        self.turn_off_eviction = False

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. CakeCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `CakeCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "CakeCache":
        """Converts a cache in the legacy cache format into an equivalent `CakeCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
    @classmethod
    def from_dynamic_cache(cls, past_key_values: Optional[DynamicCache] = None) -> "CakeCache":
        cache = cls()
        if past_key_values is not None:
            cache.key_cache = past_key_values.key_cache
            cache.value_cache = past_key_values.value_cache

        return cache
    @classmethod
    def from_hybrid_cache(cls, past_key_values: Optional[HybridCache] = None) -> "CakeCache":
        cache = cls()
        if past_key_values is not None:
            cache.key_cache = past_key_values.key_cache
            cache.value_cache = past_key_values.value_cache

        return cache
    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""

        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["CakeCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = CakeCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            current_split.pref_scores = self.pref_scores
            current_split.evict_scores = self.evict_scores
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["CakeCache"]) -> "CakeCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            layer_keys = torch.cat([current.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.value_cache[idx] for current in splits], dim=0)
            cache.update(layer_keys, layer_values, idx)
            cache.pref_scores = splits[0].pref_scores
            cache.evict_scores = splits[0].evict_scores
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]



class CakeprefillKVCache:
    def __init__(
        self,
        cache_size=512,
        window_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        num_heads = 32, 
        num_layers = 32,
        use_cascading = False,
        config=None,
        model_layers=None,
        precomputed_head_budgets=None  # NEW: Pre-computed budgets
    ):

        self.window_size = window_size
        self.cache_size = cache_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_cascading = use_cascading
        self.config = config
        self.model_layers = model_layers
        # Store pre-computed budgets instead of calculating them every time
        self.precomputed_head_budgets = precomputed_head_budgets
        # print(f"[CAKE] CakeprefillKVCache initialized with pre-computed budgets: {precomputed_head_budgets is not None}")

    def __call__(self, past_key_values, seq_len):
        if seq_len <= self.cache_size + self.window_size:
            past_key_values.turn_off_eviction = True
            return past_key_values

        # Use pre-computed budgets instead of calculating them
        if self.precomputed_head_budgets is None:
            raise ValueError("No pre-computed head budgets available! Make sure to call precompute_static_head_budgets during model initialization.")
        
        # Initialize budgets in the cache if not already done
        if past_key_values.head_budgets is None:
            past_key_values.initialize_budgets(self.precomputed_head_budgets)
            
        # print(f"[CAKE] Using pre-computed budgets, total layers: {len(self.precomputed_head_budgets)}")
        return past_key_values

# class CakeDecodingKVCache_LayerWise:
#     def __init__(
#         self,
#         hh_size=128,
#         window_size=32,
#         k_seq_dim=2,
#         v_seq_dim=2,

#     ):
#         # print(f"CakeDecodingKVCache_LayerWise: {hh_size}, {window_size}")
#         self.hh_size = hh_size
#         self.window_size = window_size
#         self.cache_size = hh_size + window_size
#         self.k_seq_dim = k_seq_dim
#         self.v_seq_dim = v_seq_dim
#         self.hh_score = None


#     def __call__(self, past_key_values, attn_score_cache, layer_idx, head_budgets):
#         # print("[CAKE] total budget for layer", layer_idx, ":", self.hh_size)



#         num_heads = attn_score_cache.shape[1]  # query heads, 32
#         bsz, num_kv_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape
#         device = past_key_values.key_cache[layer_idx].device
#         num_groups = num_heads // num_kv_heads  # typically 4

#         if seq_len <= self.cache_size:
#             print(f"[CAKE] Layer {layer_idx} seq_len ({seq_len}) <= cache_size ({self.cache_size}), skipping eviction.")
#             return past_key_values

#         # Step 1: Reduce scores to per-head values (mean over query)
#         attn_cache = attn_score_cache[:, :, :, :-self.window_size].mean(dim=-2)  # [B, 32, S-window]

#         # Step 2: Smooth scores with avg pooling
#         attn_cache = F.avg_pool1d(attn_cache, kernel_size=5, padding=2, stride=1)

#         # Step 3: Reshape to KV head grouping
#         attn_cache = attn_cache.reshape(bsz, num_kv_heads, num_groups, -1)  # [B, 8, 4, S]
#         attn_cache = attn_cache.mean(dim=2)  # [B, 8, S] — 1 score per KV head

#         # Step 4: Group budgets from 32 heads → 8 KV heads
#         if len(head_budgets) == 32 and num_kv_heads == 8:
#             head_budgets_grouped = [sum(head_budgets[i*4:(i+1)*4]) for i in range(8)]
#         else:
#             head_budgets_grouped = head_budgets

#         # head_budgets_grouped = [sum(head_budgets[i*4:(i+1)*4]) for i in range(num_kv_heads)]  # [8]

#         # Step 5: Get indices per KV head
#         max_k = max([max(k, self.window_size) for k in head_budgets_grouped])
#         print(f"[CAKE] For layer {layer_idx}, max_k (for padding): {max_k}")
        
#         new_key_cache = []
#         new_value_cache = []

#         # Past cache excluding window
#         past_kv_len = seq_len - self.window_size
#         key_past = past_key_values.key_cache[layer_idx][:, :, :past_kv_len, :]  # [B, H, S-w, D]
#         value_past = past_key_values.value_cache[layer_idx][:, :, :past_kv_len, :]

#         total_padded_tokens = 0

#         for h in range(num_kv_heads):
#             k = max(head_budgets_grouped[h], self.window_size)

#             # Top-k indices for head h
#             scores = attn_cache[:, h, :]  # [B, S-window]
#             topk_indices = scores.topk(k, dim=-1).indices  # [B, k]
#             topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # [B, k, D]

#             # Gather K/V tokens for this head
#             key_sel = key_past[:, h].gather(dim=1, index=topk_indices)  # [B, k, D]
#             value_sel = value_past[:, h].gather(dim=1, index=topk_indices)

#             # print the shapes of selected keys and values
#             print(f"[CAKE] Layer {layer_idx}, Head {h}: Selected key shape: {key_sel.shape}, Selected value shape: {value_sel.shape}")

#             # Pad if needed
#             pad_len = max_k - k
#             if pad_len > 0:
#                 total_padded_tokens += pad_len
#                 pad_shape = (bsz, pad_len, head_dim)
#                 eps = 1e-6
#                 key_pad = torch.full(pad_shape, eps, device=device, dtype=key_sel.dtype)
#                 value_pad = torch.full(pad_shape, eps, device=device, dtype=value_sel.dtype)

#                 key_sel = torch.cat([key_pad, key_sel], dim=1)  # [B, max_k, D]
#                 value_sel = torch.cat([value_pad, value_sel], dim=1)

#             print(f"[CAKE] Layer {layer_idx}, Head {h}: After padding, key shape: {key_sel.shape}, value shape: {value_sel.shape}")
#             new_key_cache.append(key_sel)
#             new_value_cache.append(value_sel)

#         # Stack across heads: [B, H, max_k, D]
#         key_compressed = torch.stack(new_key_cache, dim=1)
#         value_compressed = torch.stack(new_value_cache, dim=1)

#         print(f"[CAKE] Layer {layer_idx}: Total padded positions this eviction: {total_padded_tokens}")

#         # Keep the current window (last W tokens)
#         key_window = past_key_values.key_cache[layer_idx][:, :, -self.window_size:, :]
#         value_window = past_key_values.value_cache[layer_idx][:, :, -self.window_size:, :]

#         # Concatenate compressed + window → final [B, H, max_k + W, D]
#         key_final = torch.cat([key_compressed, key_window], dim=2)
#         value_final = torch.cat([value_compressed, value_window], dim=2)

#         # Update cache
#         past_key_values.key_cache[layer_idx] = key_final
#         past_key_values.value_cache[layer_idx] = value_final

#         print("[CAKE] After eviction, key cache shape:", key_final.shape)
#         print("[CAKE] After eviction, value cache shape:", value_final.shape)

#         return past_key_values

class CakeDecodingKVCache_LayerWise:
    def __init__(
        self,
        hh_size=128,
        window_size=32,
        k_seq_dim=2,
        v_seq_dim=2,

    ):
        # # Fix: ensure hh_size is an integer
        # if isinstance(hh_size, (list, tuple)):
        #     self.hh_size = hh_size[0] if len(hh_size) > 0 else 128
        # else:
        #     self.hh_size = hh_size

        self.hh_size = hh_size
        self.window_size = window_size
        self.cache_size = hh_size + window_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache, layer_idx, head_budgets, query_states):
        eviction_start_time = time.time()
        eviction_timing = {}
        
        # print("[CAKE] total budget for layer", layer_idx, ":", self.hh_size)

        num_heads = attn_score_cache.shape[1]  # query heads, 32
        bsz, num_kv_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape
        device = past_key_values.key_cache[layer_idx].device
        num_groups = num_heads // num_kv_heads  # typically 4

        # if seq_len <= self.cache_size:
        #     # print("Skipping eviction")
        #     # print(f"[CAKE] Layer {layer_idx} seq_len ({seq_len}) <= cache_size ({self.cache_size}), skipping eviction.")
        #     B, H, Q, D = query_states.shape
        #     device = past_key_values.key_cache[layer_idx].device
        #     # Query for varlen (typically Q=1 for decoding)
        #     q_varlen = query_states.transpose(1, 2).reshape(B * Q, H, D)  # (B*Q, H, D)
        #     # All K,V tokens (no selection needed)
        #     k_varlen = past_key_values.key_cache[layer_idx].transpose(1, 2).reshape(-1, past_key_values.key_cache[layer_idx].shape[1], D)  # (B*S, H_kv, D)
        #     v_varlen = past_key_values.value_cache[layer_idx].transpose(1, 2).reshape(-1, past_key_values.value_cache[layer_idx].shape[1], D)  # (B*S, H_kv, D)
            
        #     # Cumulative sequence lengths
        #     cu_seqlens_q = torch.arange(0, (B + 1) * Q, Q, dtype=torch.int32, device=device)
        #     cu_seqlens_k = torch.arange(0, (B + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
            
        #     max_seqlen_q = Q
        #     max_seqlen_k = seq_len
            
        #     return past_key_values, q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k

        if seq_len <= self.cache_size:

            # print(f"\n=== DEBUG NO EVICTION LAYER {layer_idx} ===")
            # print(f"seq_len ({seq_len}) <= cache_size ({self.cache_size}), skipping eviction")
            
            B, H, Q, D = query_states.shape
            device = past_key_values.key_cache[layer_idx].device
            
            # print(f"Input shapes - B:{B}, H:{H}, Q:{Q}, D:{D}")
            # print(f"KV cache shape: {past_key_values.key_cache[layer_idx].shape}")
            # print(f"seq_len: {seq_len}, window_size: {self.window_size}")
            
            # Query for varlen (typically Q=1 for decoding)
            q_varlen = query_states.transpose(1, 2).reshape(B * Q, H, D)  # (B*Q, H, D)
            # print(f"q_varlen shape: {q_varlen.shape}")
            
            # All K,V tokens (no selection needed)
            k_varlen = past_key_values.key_cache[layer_idx].transpose(1, 2).reshape(-1, past_key_values.key_cache[layer_idx].shape[1], D)  # (B*S, H_kv, D)
            v_varlen = past_key_values.value_cache[layer_idx].transpose(1, 2).reshape(-1, past_key_values.value_cache[layer_idx].shape[1], D)  # (B*S, H_kv, D)
            
            # print(f"k_varlen shape: {k_varlen.shape}")
            # print(f"v_varlen shape: {v_varlen.shape}")
            
            # Check for any NaN or inf values
            if torch.isnan(k_varlen).any():
                print("WARNING: NaN values found in k_varlen!")
            if torch.isinf(k_varlen).any():
                print("WARNING: Inf values found in k_varlen!")
            
            # Cumulative sequence lengths
            cu_seqlens_q = torch.arange(0, (B + 1) * Q, Q, dtype=torch.int32, device=device)
            cu_seqlens_k = torch.arange(0, (B + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
            
            max_seqlen_q = Q
            max_seqlen_k = seq_len
            
            # print(f"Metadata:")
            # print(f"  cu_seqlens_q: {cu_seqlens_q}")
            # print(f"  cu_seqlens_k: {cu_seqlens_k}")
            # print(f"  max_seqlen_q: {max_seqlen_q}")
            # print(f"  max_seqlen_k: {max_seqlen_k}")
            # print(f"=== END NO EVICTION DEBUG ===\n")
            
            return past_key_values, q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        # Timing: Score processing
        score_start = time.time()
        # Step 1: Reduce scores to per-head values (mean over query)
        attn_cache = attn_score_cache[:, :, :, :-self.window_size].mean(dim=-2)  # [B, 32, S-window]

        # Step 2: Smooth scores with avg pooling
        attn_cache = F.avg_pool1d(attn_cache, kernel_size=5, padding=2, stride=1)

        # Step 3: Reshape to KV head grouping
        # attn_cache = attn_cache.reshape(bsz, num_kv_heads, num_groups, -1)  # [B, 8, 4, S]
        # attn_cache = attn_cache.mean(dim=2)  # [B, 8, S] — 1 score per KV head

        attn_cache = attn_cache.view(bsz, num_kv_heads, num_groups, -1).mean(dim=2)
        eviction_timing['score_processing'] = time.time() - score_start

        # Timing: Budget grouping
        budget_start = time.time()
        # Step 4: Group budgets from 32 heads → 8 KV heads
        if len(head_budgets) == 32 and num_kv_heads == 8:
            head_budgets_grouped = [sum(head_budgets[i*4:(i+1)*4]) for i in range(8)]
        else:
            head_budgets_grouped = head_budgets
        eviction_timing['budget_processing'] = time.time() - budget_start
        

        # NEW: Prepare varlen flash attention inputs instead of padding
        return self._prepare_varlen_inputs(
            past_key_values, attn_cache, head_budgets_grouped, 
            layer_idx, query_states, eviction_timing, eviction_start_time
        )
    
    # def _prepare_varlen_inputs(self, past_key_values, attn_cache, head_budgets_grouped, 
    #                          layer_idx, query_states, eviction_timing, eviction_start_time):
    #     """
    #     Prepare inputs for flash_attn_varlen_func without padding.
    #     Returns: (past_key_values, varlen_data)
    #     """
    #     bsz, num_kv_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape
    #     device = past_key_values.key_cache[layer_idx].device
    #     B, H, Q, D = query_states.shape
        
    #     # Timing: Varlen preparation
    #     varlen_start = time.time()
        
    #     # Past cache excluding window
    #     past_kv_len = seq_len - self.window_size
    #     key_past = past_key_values.key_cache[layer_idx][:, :, :past_kv_len, :]  # [B, H_kv, S-w, D]
    #     value_past = past_key_values.value_cache[layer_idx][:, :, :past_kv_len, :]
    #     key_window = past_key_values.key_cache[layer_idx][:, :, -self.window_size:, :]  # [B, H_kv, w, D]
    #     value_window = past_key_values.value_cache[layer_idx][:, :, -self.window_size:, :]
        
    #     # Prepare query for varlen (typically Q=1 for decoding)
    #     q_varlen = query_states.transpose(1, 2).reshape(B * Q, H, D)  # (B*Q, H, D)
        
    #     # Select K,V tokens per head without padding
    #     selected_k_list = []
    #     selected_v_list = []
    #     cu_seqlens_k = [0]  # Cumulative sequence lengths for K/V
        
    #     for b in range(B):  # Usually B=1
    #         total_selected = 0
    #         for h_kv in range(num_kv_heads):
    #             budget = max(head_budgets_grouped[h_kv], 0)
                
    #             if budget > 0:
    #                 # Select top-k tokens from past (excluding window)
    #                 scores = attn_cache[b, h_kv, :]  # [S-window]
    #                 topk_indices = scores.topk(budget, dim=-1).indices  # [budget]
                    
    #                 # Gather selected tokens
    #                 selected_k_past = key_past[b, h_kv, topk_indices, :]  # [budget, D]
    #                 selected_v_past = value_past[b, h_kv, topk_indices, :]  # [budget, D]
    #             else:
    #                 # No tokens selected from past
    #                 selected_k_past = torch.empty(0, head_dim, device=device, dtype=key_past.dtype)
    #                 selected_v_past = torch.empty(0, head_dim, device=device, dtype=value_past.dtype)
                
    #             # Add window tokens (always keep these)
    #             k_head_window = key_window[b, h_kv, :, :]  # [window, D]
    #             v_head_window = value_window[b, h_kv, :, :]  # [window, D]
                
    #             # Concatenate selected + window for this head
    #             k_head_total = torch.cat([selected_k_past, k_head_window], dim=0)  # [budget+window, D]
    #             v_head_total = torch.cat([selected_v_past, v_head_window], dim=0)  # [budget+window, D]
                
    #             selected_k_list.append(k_head_total)
    #             selected_v_list.append(v_head_total)
    #             total_selected += len(k_head_total)
            
    #         cu_seqlens_k.append(cu_seqlens_k[-1] + total_selected)
        
    #     # Concatenate all selected K,V tokens across heads
    #     k_varlen = torch.cat(selected_k_list, dim=0)  # (total_selected_k, D)
    #     v_varlen = torch.cat(selected_v_list, dim=0)  # (total_selected_k, D)
        
    #     # Add head dimension back for compatibility with flash attention
    #     k_varlen = k_varlen.unsqueeze(1).expand(-1, num_kv_heads, -1)  # (total_selected_k, H_kv, D)
    #     v_varlen = v_varlen.unsqueeze(1).expand(-1, num_kv_heads, -1)  # (total_selected_k, H_kv, D)
        
    #     # Query cumulative lengths (simple for decoding)
    #     cu_seqlens_q = torch.arange(0, (B + 1) * Q, Q, dtype=torch.int32, device=device)
    #     cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
        
    #     max_seqlen_q = Q
    #     max_seqlen_k = max(cu_seqlens_k[i+1] - cu_seqlens_k[i] for i in range(B)) if B > 0 else 0
        
    #     eviction_timing['varlen_preparation'] = time.time() - varlen_start
    #     eviction_timing['total_eviction'] = time.time() - eviction_start_time
    #     eviction_timing['total_selected_tokens'] = int(len(k_varlen))
    #     eviction_timing['max_seqlen_k'] = int(max_seqlen_k)
        
    #     # Write eviction timing to file (convert any tensor values to Python types)
    #     serializable_timing = {}
    #     for key, value in eviction_timing.items():
    #         if hasattr(value, 'item'):  # PyTorch tensor
    #             serializable_timing[key] = value.item()
    #         elif isinstance(value, (int, float, str, bool)):
    #             serializable_timing[key] = value
    #         else:
    #             serializable_timing[key] = str(value)  # fallback to string
        
    #     with open(f"eviction_timing_layer_{layer_idx}.json", "a") as f:
    #         f.write(json.dumps(serializable_timing) + '\n')
        
    #     # Return the 8 values that modify_llama.py expects
    #     return past_key_values, q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
    def _prepare_varlen_inputs(self, past_key_values, attn_cache, head_budgets_grouped, 
                         layer_idx, query_states, eviction_timing, eviction_start_time):
        """
        Prepare inputs for flash_attn_varlen_func without padding.
        Returns: (past_key_values, varlen_data)
        """
        bsz, num_kv_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape
        device = past_key_values.key_cache[layer_idx].device
        B, H, Q, D = query_states.shape
        
        # print(f"\n=== DEBUG EVICTION LAYER {layer_idx} ===")
        # print(f"Input shapes - B:{B}, H:{H}, Q:{Q}, D:{D}")
        # print(f"KV cache shape: {past_key_values.key_cache[layer_idx].shape}")
        # print(f"seq_len: {seq_len}, window_size: {self.window_size}")
        # print(f"head_budgets_grouped: {head_budgets_grouped}")
        # print(f"attn_cache shape: {attn_cache.shape}")
        
        # Timing: Varlen preparation
        varlen_start = time.time()
        
        # Past cache excluding window
        past_kv_len = seq_len - self.window_size
        key_past = past_key_values.key_cache[layer_idx][:, :, :past_kv_len, :]  # [B, H_kv, S-w, D]
        value_past = past_key_values.value_cache[layer_idx][:, :, :past_kv_len, :]
        key_window = past_key_values.key_cache[layer_idx][:, :, -self.window_size:, :]  # [B, H_kv, w, D]
        value_window = past_key_values.value_cache[layer_idx][:, :, -self.window_size:, :]
        
        # print(f"past_kv_len: {past_kv_len}")
        # print(f"key_past shape: {key_past.shape}")
        # print(f"key_window shape: {key_window.shape}")
        
        # Prepare query for varlen - one query per head sequence
        num_q_groups = H // num_kv_heads  # typically 32 // 8 = 4
        q_varlen = query_states.transpose(1, 2).reshape(B * Q, num_kv_heads, num_q_groups, D)  # [1, 8, 4, 128]
        q_varlen = q_varlen.reshape(B * Q * num_kv_heads, num_q_groups, D)  # [8, 4, 128]

        # print(f"q_varlen shape: {q_varlen.shape}")
        
        # Create separate sequences for each head (this is the key change!)
        # Pre-calculate total size and allocate once
        # Each head gets exactly its budget (which includes the window)
        total_tokens = sum(head_budgets_grouped[h] for h in range(num_kv_heads))
        k_varlen = torch.empty(total_tokens, 1, head_dim, device=device, dtype=key_past.dtype)
        v_varlen = torch.empty(total_tokens, 1, head_dim, device=device, dtype=value_past.dtype)

        # Track cumulative sequence lengths
        cu_seqlens_k = [0]
        start_idx = 0

        for b in range(B):  # Usually B=1
            for h_kv in range(num_kv_heads):
                budget = max(head_budgets_grouped[h_kv] - self.window_size, 0)  # e.g., 1024 - 32 = 992
                
                if budget > 0:
                    # Select top-k tokens from past (excluding window)
                    scores = attn_cache[b, h_kv, :]  # [S-window]
                    
                    if budget > len(scores):
                        budget = len(scores)
                    
                    _, topk_indices = scores.topk(budget, dim=-1)  # Optimized: don't store values
                    
                    # Gather selected tokens
                    selected_k_past = key_past[b, h_kv, topk_indices, :]  # [budget, D]
                    selected_v_past = value_past[b, h_kv, topk_indices, :]  # [budget, D]
                else:
                    # No tokens selected from past
                    selected_k_past = torch.empty(0, head_dim, device=device, dtype=key_past.dtype)
                    selected_v_past = torch.empty(0, head_dim, device=device, dtype=value_past.dtype)
                
                # Add window tokens (always keep these)
                k_head_window = key_window[b, h_kv, :, :]  # [window, D]
                v_head_window = value_window[b, h_kv, :, :]  # [window, D]
                
                # Concatenate selected + window for this head
                k_head_total = torch.cat([selected_k_past, k_head_window], dim=0)  # [budget+window, D]
                v_head_total = torch.cat([selected_v_past, v_head_window], dim=0)  # [budget+window, D]
                
                # Fill directly into pre-allocated tensor
                end_idx = start_idx + len(k_head_total)
                k_varlen[start_idx:end_idx, 0, :] = k_head_total
                v_varlen[start_idx:end_idx, 0, :] = v_head_total
                
                # Update cumulative sequence lengths
                cu_seqlens_k.append(cu_seqlens_k[-1] + len(k_head_total))
                start_idx = end_idx
        
        # print(f"Final k_varlen shape: {k_varlen.shape}")
        # print(f"Final v_varlen shape: {v_varlen.shape}")
        
        
        
        # Now cu_seqlens_k represents: [0, head0_len, head0_len+head1_len, ...]
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
        
        # Query metadata needs to match the number of head sequences
        cu_seqlens_q = torch.arange(0, B * Q * num_kv_heads + 1, dtype=torch.int32, device=device)  # [0, 1, 2, ..., 8]
        
        max_seqlen_q = 1
        # max_seqlen_k = max(len(seq) for seq in k_varlen_list) if k_varlen_list else 0
        # Calculate max_seqlen_k from cumulative sequence lengths
        max_seqlen_k = max(cu_seqlens_k[i+1] - cu_seqlens_k[i] for i in range(len(cu_seqlens_k)-1)) if len(cu_seqlens_k) > 1 else 0     
        # print(f"Metadata:")
        # print(f"  cu_seqlens_q: {cu_seqlens_q}")
        # print(f"  cu_seqlens_k: {cu_seqlens_k}")
        # print(f"  max_seqlen_q: {max_seqlen_q}")
        # print(f"  max_seqlen_k: {max_seqlen_k}")
        # print(f"  Number of head sequences: {len(k_varlen_list)}")
        # print(f"=== END DEBUG ===\n")
        
        eviction_timing['varlen_preparation'] = time.time() - varlen_start
        eviction_timing['total_eviction'] = time.time() - eviction_start_time
        eviction_timing['total_selected_tokens'] = int(len(k_varlen))
        eviction_timing['max_seqlen_k'] = int(max_seqlen_k)
        
        # Write eviction timing to file
        serializable_timing = {}
        for key, value in eviction_timing.items():
            if hasattr(value, 'item'):
                serializable_timing[key] = value.item()
            elif isinstance(value, (int, float, str, bool)):
                serializable_timing[key] = value
            else:
                serializable_timing[key] = str(value)
        
        with open(f"eviction_timing_layer_{layer_idx}.json", "a") as f:
            f.write(json.dumps(serializable_timing) + '\n')
        
        return past_key_values, q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
    #     # shape of key value cache
    #     print("[CAKE] Key cache shape:", past_key_values.key_cache[layer_idx].shape)
    #     print("[CAKE] Value cache shape:", past_key_values.value_cache[layer_idx].shape)
    #     bsz, num_key_value_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape
    #     num_key_value_groups = num_heads // num_key_value_heads

    #     seq_len = past_key_values.key_cache[layer_idx].size(self.k_seq_dim)
    #     if seq_len <= self.hh_size:
    #         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    #         print("No eviction needed, seq_len:", seq_len, "hh_size:", self.hh_size, "window_size:", self.window_size)
    #         return past_key_values

    #     attn_cache = attn_score_cache[:, :, :, :-self.window_size].mean(dim = -2)

    #     attn_cache = F.avg_pool1d(attn_cache, kernel_size = 5, padding=5//2, stride=1)
    #     attn_cache = attn_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)

    #     attn_cache = attn_cache.mean(dim=-2)

    #     indices = attn_cache.topk(self.hh_size - self.window_size, dim=-1).indices
    #     # indices = indices.sort().values
    #     indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

    #     k_past_compress = past_key_values.key_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
    #     v_past_compress = past_key_values.value_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
    #     k_cur = past_key_values.key_cache[layer_idx][:, :, -self.window_size:, :]
    #     v_cur = past_key_values.value_cache[layer_idx][:, :, -self.window_size:, :]
    #     key_states = torch.cat([k_past_compress, k_cur], dim=2)
    #     value_states = torch.cat([v_past_compress, v_cur], dim=2)
    #     print("[CAKE] After eviction, key cache shape:", key_states.shape)
    #     print("[CAKE] After eviction, value cache shape:", value_states.shape)

    #     past_key_values.key_cache[layer_idx] = key_states
    #     past_key_values.value_cache[layer_idx] = value_states


    #     return past_key_values

    def _update_hh_score(self, attn_score_cache, num_key_value_heads):

        bsz,num_heads, num_new_tokens,_ = attn_score_cache.shape
        num_key_value_groups = num_heads //  num_key_value_heads
        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(2) #bsz, total num_heads, seq_len
            self.hh_score = self.hh_score.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)
            self.hh_score = self.hh_score.mean(dim=-2)
        
        else:
            # print(self.hh_score.shape)
            attn_score_cache = attn_score_cache.sum(2)
            attn_score_cache = attn_score_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)
            attn_score_cache = attn_score_cache.mean(dim=-2)
            attn_score_cache[:, :, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None