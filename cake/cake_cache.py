import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from transformers.cache_utils import DynamicCache, Cache, HybridCache
from typing import Any, Dict, List, Optional, Tuple, Union

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
            cache.pref_scores = self.pref_scores
            cache.evict_scores = self.evict_scores
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
        model_layers=None
    ):

        self.window_size = window_size
        # self.total_size = (cache_size-window_size) * num_layers * num_heads # might have to change
        self.total_size = (cache_size-window_size) * num_layers # might have to change

        self.cache_size = cache_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_cascading = use_cascading  # If true, ensure high attention precision
        # Although the cascading came with CAKE, I have not used it here. 
        self.config = config
        self.model_layers = model_layers
        # print(f"CakeprefillKVCache: {self.total_size}, {self.window_size}")

    def __call__(self, past_key_values, seq_len):
        if seq_len<=self.cache_size+self.window_size:
            past_key_values.turn_off_eviction = True
            return past_key_values

        past_key_values.turn_off_eviction = False
        pref_scores = past_key_values.pref_scores
        # print(f"[CAKE] Pref Scores: {pref_scores}")
        head_budgets = compute_head_budgets_dynamic(
            pref_scores,
            self.total_size,
            # allocation_strategy="entropy_based",
            allocation_strategy="static",
            max_seq_len=self.cache_size
        )
        print(head_budgets)
        # Store head budgets in the CakeCache object
        past_key_values.head_budgets = head_budgets
        # print(f"[CAKE] Head Budgets: {head_budgets}")
        for layer_idx in head_budgets:
            # print(f"[CAKE] Layer {layer_idx} Head Budget: {head_budgets[layer_idx]}")
            layer_budget = sum(head_budgets[layer_idx])
            past_key_values.layer_budget[layer_idx] = layer_budget
            

        
        # print(f"[CAKE] Head Budgets: {past_key_values.head_budgets}")
        # print(f"[CAKE] Stored head_budgets in past_key_values: {head_budgets}")
        # for layer_idx in head_budgets:
        #     past_key_values = self.evict_kvcache_headwise(
        #         past_key_values,
        #         layer_idx,
        #         head_budgets[layer_idx],
        #         self.window_size
        #     )
        #     past_key_values.layer_budget[layer_idx] = sum(head_budgets[layer_idx])

        # head_budgets = compute_head_budgets(evict_scores, total_budget=self.total_size, window_size=self.window_size)

  
        # layer_budgets = [pref_score/sum(pref_scores)*self.total_size for pref_score in pref_scores]
    
        # layer_budgets = adjust_budgets(layer_budgets, self.total_size, seq_len-self.window_size,  self.num_layers)

        # if self.use_cascading:
        #     layer_idx = 0
        #     print(layer_budgets)
        #     for budget in layer_budgets:
        #         if budget>= seq_len-self.window_size:
        #             budget = seq_len-self.window_size
        #         past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget)
        #         past_key_values.layer_budget[layer_idx]=budget
        #         layer_idx +=1
        # else:
        #     layer_idx = 0
        #     if len(layer_budgets) ==self.num_layers:
        #         for budget in layer_budgets:
        #             if budget>= seq_len-self.window_size:
        #                 budget = seq_len-self.window_size
        #             past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget)
        #             past_key_values.layer_budget[layer_idx]=budget
        #             layer_idx +=1

        return past_key_values

class CakeDecodingKVCache_LayerWise:
    def __init__(
        self,
        hh_size=128,
        window_size=32,
        k_seq_dim=2,
        v_seq_dim=2,

    ):
        # print(f"CakeDecodingKVCache_LayerWise: {hh_size}, {window_size}")
        self.hh_size = hh_size
        self.window_size = window_size
        self.cache_size = hh_size + window_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None


    def __call__(self, past_key_values, attn_score_cache, layer_idx, head_budgets):
        # print("[CAKE] total budget for layer", layer_idx, ":", self.hh_size)



        num_heads = attn_score_cache.shape[1]  # query heads, 32
        bsz, num_kv_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape
        device = past_key_values.key_cache[layer_idx].device
        num_groups = num_heads // num_kv_heads  # typically 4

        if seq_len <= self.cache_size:
            print(f"[CAKE] Layer {layer_idx} seq_len ({seq_len}) <= cache_size ({self.cache_size}), skipping eviction.")
            return past_key_values

        # Step 1: Reduce scores to per-head values (mean over query)
        attn_cache = attn_score_cache[:, :, :, :-self.window_size].mean(dim=-2)  # [B, 32, S-window]

        # Step 2: Smooth scores with avg pooling
        attn_cache = F.avg_pool1d(attn_cache, kernel_size=5, padding=2, stride=1)

        # Step 3: Reshape to KV head grouping
        attn_cache = attn_cache.reshape(bsz, num_kv_heads, num_groups, -1)  # [B, 8, 4, S]
        attn_cache = attn_cache.mean(dim=2)  # [B, 8, S] — 1 score per KV head

        # Step 4: Group budgets from 32 heads → 8 KV heads
        if len(head_budgets) == 32 and num_kv_heads == 8:
            head_budgets_grouped = [sum(head_budgets[i*4:(i+1)*4]) for i in range(8)]
        else:
            head_budgets_grouped = head_budgets

        # head_budgets_grouped = [sum(head_budgets[i*4:(i+1)*4]) for i in range(num_kv_heads)]  # [8]

        # Step 5: Get indices per KV head
        max_k = max([max(k, self.window_size) for k in head_budgets_grouped])
        # print(f"[CAKE] For layer {layer_idx}, max_k (for padding): {max_k}")
        
        new_key_cache = []
        new_value_cache = []

        # Past cache excluding window
        past_kv_len = seq_len - self.window_size
        key_past = past_key_values.key_cache[layer_idx][:, :, :past_kv_len, :]  # [B, H, S-w, D]
        value_past = past_key_values.value_cache[layer_idx][:, :, :past_kv_len, :]

        total_padded_tokens = 0

        for h in range(num_kv_heads):
            k = max(head_budgets_grouped[h], self.window_size)

            # Top-k indices for head h
            scores = attn_cache[:, h, :]  # [B, S-window]
            topk_indices = scores.topk(k, dim=-1).indices  # [B, k]
            topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # [B, k, D]

            # Gather K/V tokens for this head
            key_sel = key_past[:, h].gather(dim=1, index=topk_indices)  # [B, k, D]
            value_sel = value_past[:, h].gather(dim=1, index=topk_indices)

            # print the shapes of selected keys and values
            # print(f"[CAKE] Layer {layer_idx}, Head {h}: Selected key shape: {key_sel.shape}, Selected value shape: {value_sel.shape}")

            # Pad if needed
            pad_len = max_k - k
            if pad_len > 0:
                total_padded_tokens += pad_len
                pad_shape = (bsz, pad_len, head_dim)
                eps = 1e-6
                key_pad = torch.full(pad_shape, eps, device=device, dtype=key_sel.dtype)
                value_pad = torch.full(pad_shape, eps, device=device, dtype=value_sel.dtype)

                key_sel = torch.cat([key_pad, key_sel], dim=1)  # [B, max_k, D]
                value_sel = torch.cat([value_pad, value_sel], dim=1)

            # print(f"[CAKE] Layer {layer_idx}, Head {h}: After padding, key shape: {key_sel.shape}, value shape: {value_sel.shape}")
            new_key_cache.append(key_sel)
            new_value_cache.append(value_sel)

        # Stack across heads: [B, H, max_k, D]
        key_compressed = torch.stack(new_key_cache, dim=1)
        value_compressed = torch.stack(new_value_cache, dim=1)

        # print(f"[CAKE] Layer {layer_idx}: Total padded positions this eviction: {total_padded_tokens}")

        # Keep the current window (last W tokens)
        key_window = past_key_values.key_cache[layer_idx][:, :, -self.window_size:, :]
        value_window = past_key_values.value_cache[layer_idx][:, :, -self.window_size:, :]

        # Concatenate compressed + window → final [B, H, max_k + W, D]
        key_final = torch.cat([key_compressed, key_window], dim=2)
        value_final = torch.cat([value_compressed, value_window], dim=2)

        # Update cache
        past_key_values.key_cache[layer_idx] = key_final
        past_key_values.value_cache[layer_idx] = value_final

        # print("[CAKE] After eviction, key cache shape:", key_final.shape)
        # print("[CAKE] After eviction, value cache shape:", value_final.shape)

        return past_key_values

    # def __call__(self, past_key_values, attn_score_cache, layer_idx):

    #     print("[CAKE] total budget for layer", layer_idx, ":", self.hh_size)
    #     num_heads = attn_score_cache.shape[1]

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