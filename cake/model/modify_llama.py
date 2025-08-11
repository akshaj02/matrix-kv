import math
from typing import Optional, Tuple
import time
import threading

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F
import transformers

from transformers.models.llama.modeling_llama import *
from transformers.modeling_flash_attention_utils import _flash_attention_forward

try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_VARLEN = True
except ImportError:
    print("Warning: flash_attn_varlen_func not available, falling back to regular flash attention")
    HAS_FLASH_VARLEN = False

from transformers.cache_utils import Cache, DynamicCache, StaticCache

from transformers.models.llama.configuration_llama import LlamaConfig


from ..cake_cache import CakeCache, CakeDecodingKVCache_LayerWise

from ..utils import calculate_entropy

import json

# modify_llama.py

layer_logs = {}  # global dict to store scores for debugging
prefill_head_norms = {}

# Global timing accumulator for detailed timing
timing_lock = threading.Lock()
global_timing_stats = {
    'prefill_setup': 0.0,
    'attention_computation': 0.0,
    'masking_time': 0.0,
    'softmax_time': 0.0,
    'eviction_time': 0.0,
    'varlen_selection_time': 0.0,
    'varlen_preparation_time': 0.0,
    'varlen_flash_attention_time': 0.0,
    'flash_attention_time': 0.0,
    'output_processing': 0.0,
    'total_forward_calls': 0
}

# prepare_varlen_attention function removed - functionality moved to CakeDecodingKVCache_LayerWise

def llama_attn_forward_cake(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    forward_start_time = time.time()
    local_timing = {}

    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    if isinstance(past_key_value, DynamicCache):
        past_key_value = CakeCache.from_dynamic_cache(past_key_value)
    
    # Timing: Initial setup and projections
    setup_start = time.time()
    if (self.config.decoding_evict[self.layer_idx] is None and 
        hasattr(past_key_value, 'layer_budget') and 
        len(past_key_value.layer_budget) > self.layer_idx):
        self.config.decoding_evict[self.layer_idx] = CakeDecodingKVCache_LayerWise(
            hh_size=past_key_value.layer_budget[self.layer_idx],
            window_size=self.config.window_size[self.layer_idx],
            k_seq_dim=2,
            v_seq_dim=2
        )
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    # # Flash attention requires the input to have the shape
    # # batch_size x seq_length x head_dim x hidden_dim
    # # therefore we just need to keep the original shape
    # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout


    local_timing['setup_projections'] = time.time() - setup_start

    is_prefill = q_len != 1


    # Timing: Prefill phase
    prefill_start = time.time()
    # if self.config.prefill[self.layer_idx]:
    if is_prefill:
        # Initialize budgets on first prefill layer if not done yet
        if past_key_value.head_budgets is None and hasattr(self.config, 'head_budgets'):
            past_key_value.initialize_budgets(self.config.head_budgets)
            
            # # Check sequence length to determine if eviction is needed
            # total_seq_len = past_key_value.get_seq_length() + q_len
            # if total_seq_len <= self.config.cache_size + self.config.window_size[self.layer_idx]:
            #     past_key_value.turn_off_eviction = True
            #     # print(f"[CAKE] Sequence length {total_seq_len} <= cache limit, eviction disabled")
            # else:
            #     past_key_value.turn_off_eviction = False
            #     # print(f"[CAKE] Sequence length {total_seq_len} > cache limit, eviction enabled")

        # Timing: Tensor reshaping for flash attention
        reshape_start = time.time()
        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        local_timing['tensor_reshape'] = time.time() - reshape_start

        # Timing: Flash attention call
        flash_attn_start = time.time()
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        local_timing['flash_attention_time'] = time.time() - flash_attn_start
        self.config.prefill[self.layer_idx] = False
    local_timing['prefill_setup'] = time.time() - prefill_start
    
    # Timing: Decoding eviction phase
    decoding_start = time.time()
    # if self.config.decoding_evict[self.layer_idx] is not None:
    if not is_prefill:
        # print the KV cache shapes after prefill and before eviction
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(f"WE ARE IN LAYER {self.layer_idx}")

        attn_comp_start = time.time()
        tmp_attn_weights = torch.matmul(query_states[..., -self.config.window_size[self.layer_idx]:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        local_timing['attention_computation'] = time.time() - attn_comp_start
        # Timing: Attention computation for eviction

        # NEW: Varlen Flash Attention Path (replaces masking + eviction + padding)
        if hasattr(past_key_value, 'head_budgets') and past_key_value.head_budgets and HAS_FLASH_VARLEN:
            head_budgets = past_key_value.head_budgets.get(self.layer_idx)
            if head_budgets is not None:
                # Timing: Varlen eviction and preparation
                varlen_selection_start = time.time()

                # Softmax for attention scores (needed for token selection)
                tmp_attn_weights_softmax = nn.functional.softmax(tmp_attn_weights, dim=-1, dtype=torch.float32)

                # Call eviction with varlen preparation - this replaces masking + eviction + padding
                result = self.config.decoding_evict[self.layer_idx](
                    past_key_value, 
                    tmp_attn_weights_softmax, 
                    self.layer_idx, 
                    head_budgets,
                    query_states=query_states
                )
                past_key_value, q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = result

                        
                local_timing['varlen_selection_time'] = time.time() - varlen_selection_start
                        
                        # Timing: Varlen flash attention call
                varlen_flash_start = time.time()
                attn_output = flash_attn_varlen_func(
                    q_varlen,
                    k_varlen, 
                    v_varlen,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    causal=True
                )  # Returns (total_q, nheads, headdim)
                local_timing['varlen_flash_attention_time'] = time.time() - varlen_flash_start
                        
                # Reshape back to original format
                # B, Q = query_states.shape[0], query_states.shape[2] 
                # attn_output = attn_output.reshape(B, self.num_heads, Q, self.head_dim)
                # attn_output = attn_output.transpose(1, 2).reshape(B, Q, self.hidden_size)
                

                ##### THIS VERSION WORKS ######
                # B, Q = query_states.shape[0], query_states.shape[2]
                # num_kv_heads = self.num_key_value_heads  # Usually 8 for Llama
                # num_q_groups = self.num_heads // num_kv_heads  # Usually 32 // 8 = 4

                # # attn_output from flash_attn_varlen_func has shape [8, 4, 128] 
                # # We need to reshape it back to [B, Q, num_heads, head_dim] = [1, 1, 32, 128]
                # attn_output = attn_output.reshape(B * Q * num_kv_heads, num_q_groups, self.head_dim)  # [8, 4, 128]
                # attn_output = attn_output.reshape(B, Q, num_kv_heads * num_q_groups, self.head_dim)  # [1, 1, 32, 128]
                # attn_output = attn_output.reshape(B, Q, self.hidden_size)  # [1, 1, 4096]
                ##############################

                #### EFFICIENT

                # Replace the entire reshape section with this single line:
                B, Q = query_states.shape[0], query_states.shape[2]
                # attn_output from flash_attn_varlen_func has shape [8, 4, 128]
                # Direct reshape to final format: [B, Q, hidden_size]
                attn_output = attn_output.reshape(B, Q, self.hidden_size)  # [1, 1, 4096]
    
    local_timing['decoding_total'] = time.time() - decoding_start

    # we will find the norms for each layer in this if statement and then add them to the prefill_head_norms dict
    # if is_prefill:
    #     head_norms = torch.norm(attn_output, p=2, dim=-1)  # shape: (batch_size, query_length, num_heads)
    #     # print(f"[CAKE] Layer {self.layer_idx} | head norms shape = {head_norms.shape}")
    #     # i only want a scalar for each head norm 
    #     head_norms = head_norms.mean(dim=1)  # shape: (batch_size, num_heads)
    #     # print(f"[CAKE] Layer {self.layer_idx} | head norms mean shape = {head_norms.shape}")
    #     # the batch size is always 1 so we only need 32 scalar values per layer
    #     head_norms = head_norms.squeeze(0)  # shape: (num_heads,)
    #     # print(f"[CAKE] Layer {self.layer_idx} | head norms squeezed shape = {head_norms.shape}")
    #     # store the head norms in the prefill_head_norms dict, append from last entry
    #     # if self.layer_idx not in prefill_head_norms:
    #     #     prefill_head_norms[self.layer_idx] = []
    #     # remove the head_norms from tensor and append just the values
    #     head_norms = head_norms.detach().cpu().numpy().tolist()  # convert to numpy

    #     prefill_head_norms.update({self.layer_idx: head_norms})  # update the dict with the head norms for this layer
    #     if self.layer_idx == self.config.num_hidden_layers - 1:
    #     # Save to disk (write prefill norms as the base of your sample JSON)
    #         with open(f"sample_xyz_prefill_norms.tmp", "w") as f:
    #             json.dump({"prefill": prefill_head_norms}, f, indent=2)

    # Timing: Final output processing
    output_start = time.time()
    # print(f"[CAKE] Layer {self.layer_idx} | attn_output shape = {attn_output.shape}")
    # attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    # print(f"[CAKE] Layer {self.layer_idx} | attn_output reshaped shape = {attn_output.shape}") # should be (batch_size, query_length, num_heads * head_dim)
    attn_output = self.o_proj(attn_output)
    local_timing['output_processing'] = time.time() - output_start
    
    # Accumulate timing statistics globally
    local_timing['total_forward'] = time.time() - forward_start_time
    with timing_lock:
        global_timing_stats['total_forward_calls'] += 1
        for key, value in local_timing.items():
            if key not in global_timing_stats:
                global_timing_stats[key] = 0.0
            global_timing_stats[key] += value
        
        # Write detailed per-layer timing every 10 calls for debugging
        if global_timing_stats['total_forward_calls'] % 10 == 0:
            avg_stats = {k: v/global_timing_stats['total_forward_calls'] if k != 'total_forward_calls' else v 
                        for k, v in global_timing_stats.items()}
            timing_file = f"layer_timing_stats_layer_{self.layer_idx}.json"
            with open(timing_file, 'w') as f:
                json.dump({
                    "layer_idx": self.layer_idx,
                    "current_timing": local_timing,
                    "cumulative_avg": avg_stats
                }, f, indent=2)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def llama_model_forward_cake(
    
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if (
        use_cache and not isinstance(past_key_values, Cache) and not self.training
    ):  # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            past_key_values = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )