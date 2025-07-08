import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F
import transformers

from transformers.models.llama.modeling_llama import *
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from transformers.cache_utils import Cache, DynamicCache, StaticCache

from transformers.models.llama.configuration_llama import LlamaConfig


from ..cake_cache import CakeCache, CakeDecodingKVCache_LayerWise, MatrixDecodingKVCache

from ..utils import calculate_entropy

# modify_llama.py

layer_logs = {}  # global dict to store scores for debugging


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

    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    if isinstance(past_key_value, DynamicCache):
        past_key_value = CakeCache.from_dynamic_cache(past_key_value)
    if self.config.decoding_evict[self.layer_idx] is None and len(past_key_value.layer_budget) == self.config.prefill_cake_evict[self.layer_idx].num_layers:
        # self.config.decoding_evict[self.layer_idx] =CakeDecodingKVCache_LayerWise(
        #         hh_size =past_key_value.layer_budget[self.layer_idx],
        #         window_size=self.config.window_size[self.layer_idx],
        #         k_seq_dim=2,
        #         v_seq_dim=2
        #         )
        # here for decoration not being used rn 
        self.config.decoding_evict[self.layer_idx] = MatrixDecodingKVCache(
            head_budgets=past_key_value.head_budgets[self.layer_idx],
            window_size=self.config.window_size[self.layer_idx],
            k_seq_dim=2,
            v_seq_dim=2
        )
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    # # Flash attention requires the input to have the shape
    # # batch_size x seq_length x head_dim x hidden_dim
    # # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

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


    # query_states = query_states.transpose(1, 2)
    # key_states = key_states.transpose(1, 2)
    # value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0


    is_prefill = q_len != 1

    # prefill phase, no eviction, just regular attention
    # if is_prefill:
        


    if self.config.prefill[self.layer_idx]:

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        tmp_attn_weights = torch.matmul(query_states[..., -self.config.window_size[self.layer_idx]:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # # Retrieve the per-head budgets for this layer
        # if hasattr(past_key_value, 'head_budgets') and past_key_value.head_budgets:
        #     head_budgets = past_key_value.head_budgets.get(self.layer_idx)
        # print(f"[CAKE] Layer {self.layer_idx} | head_budgets = {head_budgets}")
        # B, H, Q, S = tmp_attn_weights.shape
        # device = tmp_attn_weights.device

        # # Build the mask: True means "mask this token"
        # mask = torch.ones(H, S, dtype=torch.bool, device=device)
        # for h in range(H):
        #     k = head_budgets[h]
        #     if k > 0:
        #         keep_indices = torch.arange(S - k, S, device=device)
        #         mask[h, keep_indices] = False  # False = not masked (keep)

        # # Expand mask for broadcasting: [1, H, 1, S]
        # attn_mask = mask.unsqueeze(0).unsqueeze(2)  # [1, H, 1, S]
        # tmp_attn_weights = tmp_attn_weights.masked_fill(attn_mask, float('-inf'))

        if q_len !=1:

            mask = torch.full((self.config.window_size[self.layer_idx], self.config.window_size[self.layer_idx]), torch.finfo(tmp_attn_weights.dtype).min, device=tmp_attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=tmp_attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(tmp_attn_weights.device)
            tmp_attention_mask = mask[None, None, :, :]

            tmp_attn_weights[:, :, -self.config.window_size[self.layer_idx]:, -self.config.window_size[self.layer_idx]:] += tmp_attention_mask

        # tmp_attn_weights = nn.functional.softmax(tmp_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        tmp_attn_weights = nn.functional.softmax(tmp_attn_weights, dim=-1, dtype=torch.float64)
    
        # print(f"[CAKE] Layer {self.layer_idx} | tmp_attn_weights shape = {tmp_attn_weights.shape}")


        # disp = calculate_entropy(tmp_attn_weights[:,:,-self.config.window_size[self.layer_idx]:,:-self.config.window_size[self.layer_idx]])
        # var = torch.var(tmp_attn_weights[:,:,-self.config.window_size[self.layer_idx]:,:-self.config.window_size[self.layer_idx]],dim=-2).sum(0).sum(0).sum(0)
        attn_focus = tmp_attn_weights[:, :, -self.config.window_size[self.layer_idx]:, :-self.config.window_size[self.layer_idx]]  # [B, H, W, L]

        entropy = -torch.sum(attn_focus * torch.log(attn_focus + 1e-10), dim=-1)  # [B, H, W]
        entropy_per_head = entropy.mean(dim=(0, 2))  # [H]

        variance_per_head = attn_focus.var(dim=-1).mean(dim=(0, 2))  # [H]
        entropy_per_head = torch.nan_to_num(entropy_per_head, nan=0.0, posinf=0.0, neginf=0.0)
        variance_per_head = torch.nan_to_num(variance_per_head, nan=0.0, posinf=0.0, neginf=0.0)

        pref_score_per_head = (entropy_per_head ** (1 / self.config.tau1)) * (variance_per_head ** (1 / self.config.tau2))
        pref_score_per_head = torch.nan_to_num(pref_score_per_head, nan=0.0, posinf=0.0, neginf=0.0)
        # print("Preference score calculated.")

        # print(f"[CAKE] Layer {self.layer_idx} | pref_score_per_head shape = {pref_score_per_head.shape}")
        # print(f"[CAKE] Layer {self.layer_idx} | pref_score_per_head sample = {pref_score_per_head[:5].detach().cpu().numpy()}")

        #compute preference score and hh score
        attention_score = tmp_attn_weights[:, :, -self.config.window_size[self.layer_idx]:, :] 

        attn_mean = attention_score.mean(dim = -2)
        attn_var = attention_score.var(dim = -2)
        attn_cache = attn_mean + self.config.gamma * attn_var
        attn_cache = attn_cache[:, :, :-self.config.window_size[self.layer_idx]]
        attn_cache = F.avg_pool1d(attn_cache, kernel_size=5, padding=5//2, stride=1)

        # attn_cache = attn_cache.reshape(bsz, self.num_key_value_heads, self.num_key_value_groups, -1)
        # hh_score = attn_cache.mean(dim=-2)
        hh_score = attn_cache
        hh_score = torch.nan_to_num(hh_score, nan=0.0, posinf=0.0, neginf=0.0)

        past_key_value.update_score(pref_score_per_head, hh_score)
        # past_key_value.update_score(pref_score_per_head, hh_score)
        past_key_value.layer_budget.append(self.config.key_size[self.layer_idx])
        self.config.prefill[self.layer_idx] =False #here prefill is complete
        past_key_value = self.config.prefill_cake_evict[self.layer_idx](past_key_value, q_len)

        # I am putting the regular flash attention inside of the prefill condition because during decoding we will use the AdaKV style flash attention varlen
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0
        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.

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

        # cache contents
        print(f"[MATRIX] Layer {self.layer_idx} | Key Cache contents: {past_key_value.key_cache[self.layer_idx]}")
        print(f"[MATRIX] Layer {self.layer_idx} | Value Cache contents: {past_key_value.value_cache[self.layer_idx]}")

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
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None


    # we only want to evict once after prefill is done
    # and not during decoding, so we check if eviction is done
    if not hasattr(self.config, "_eviction_done"):
        self.config._eviction_done = False


    # Check if prefill is done for all layers and eviction not yet done
    if all(flag is False for flag in self.config.prefill) and not self.config._eviction_done:
        print("[CAKE] Running one-time KV cache eviction after prefill (AdaKV style)")
        for layer_idx in range(len(self.config.prefill)):
            head_budgets = past_key_value.head_budgets[layer_idx]
            # Run one-time eviction
            past_key_value = MatrixDecodingKVCache(
                head_budgets=head_budgets,
                window_size=self.config.window_size[layer_idx],
                k_seq_dim=2,
                v_seq_dim=2
            )(head_budgets, past_key_value, layer_idx)
        # Disable further eviction during decoding
        # for i in range(len(self.config.prefill)):
        #     self.config.decoding_evict[i] = None
        self.config._eviction_done = True

        print(f"[MATRIX] Layer {self.layer_idx} | Query States shape: {query_states.shape}")
        print(f"[MATRIX] Layer {self.layer_idx} | Key States shape: {key_states.shape}")
        print(f"[MATRIX] Layer {self.layer_idx} | Value States shape: {value_states.shape}")
        print("----------------------")


        # for h, k in enumerate(past_key_value.key_cache[0]):
        #     print(f"[MATRIX] Layer: {0} Key Cache Head {h} shape after eviction: {k.shape}")
        # for h, v in enumerate(past_key_value.value_cache[0]):
        #     print(f"[MATRIX] Layer: {0} Value Cache Head {h} shape after eviction: {v.shape}")
    print("POST EVICTION KV CACHE CONTENTS")
    print(f"[MATRIX] Layer {self.layer_idx} | Key Cache shape: {past_key_value.key_cache[self.layer_idx].shape}")
    print(f"[MATRIX] Layer {self.layer_idx} | Value Cache shape: {past_key_value.value_cache[self.layer_idx].shape}")
    print(f"[MATRIX] Layer {self.layer_idx} | Key Cache contents: {past_key_value.key_cache[self.layer_idx]}")
    print(f"[MATRIX] Layer {self.layer_idx} | Value Cache contents: {past_key_value.value_cache[self.layer_idx]}")


    if self.config.decoding_evict[self.layer_idx] is not None:

        # print the shapes of KQV tensors
        print(f"[MATRIX] Layer {self.layer_idx} | Query States shape: {query_states.shape}")
        print(f"[MATRIX] Layer {self.layer_idx} | Key States shape: {key_states.shape}")
        print(f"[MATRIX] Layer {self.layer_idx} | Value States shape: {value_states.shape}")

        B, H, S, D = key_states.shape
        device = key_states.device
        head_budgets = past_key_value.head_budgets[self.layer_idx]
        if len(head_budgets) == 32 and H == 8:
            kv_head_budgets = [sum(head_budgets[i*4:(i+1)*4]) for i in range(8)]
        else:
            kv_head_budgets = head_budgets
        # convert kv_head_budgets to a tuple of ints
        kv_head_budgets = tuple(kv_head_budgets)

        # outputs of attention per head
        outputs = []    
        for h in range(H):
            # make the attn_mask for each head
            attn_mask = torch.ones(B, 1, 1, S, device=device, dtype=torch.bool)
            # print("[MATRIX] Head Number", h, "| kv_head_budgets[h] =", kv_head_budgets[h])
            if kv_head_budgets[h] > 0:
                keep_indices = torch.arange(S - kv_head_budgets[h], S, device=device)
                attn_mask[:, :, :, keep_indices] = False
            
            # print(f"[MATRIX] Head Number {h} | attn_mask shape: {attn_mask.shape}")


            # # attn_mask = past_key_value.attn_mask[self.layer_idx]
            # if attn_mask.dim() == 3:
            #     attn_mask = attn_mask.unsqueeze(2).expand(-1, -1, query_states.size(2), -1)

            # float_mask = torch.zeros_like(attn_mask, dtype=query_states.dtype)
            # float_mask = float_mask.masked_fill(attn_mask, float('-inf'))
            # print(f"[MATRIX] Head Number {h} | float_mask shape: {float_mask.shape}")
            print(f"[MATRIX] made it to decoding for head {h} in layer {self.layer_idx}")

            q_len = query_states.size(2)  # Query length (typically 1 for decoding)
            k_len = attn_mask.size(-1)    # Key length (should match cache length after eviction/padding)

            causal_mask = torch.tril(torch.ones((q_len, k_len), dtype=torch.bool, device=attn_mask.device))  # [Q, S]
            causal_mask = ~causal_mask  # Invert: True means masked (future tokens)
            
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, S]

            print(f"[MATRIX] head {h} in Layer {self.layer_idx} | causal_mask shape: {causal_mask.shape}")

            # Both masks are [B, H, Q, S] (broadcast causal_mask if needed)
            combined_mask = attn_mask | causal_mask  # Logical OR: masked if either is masked

            float_mask = torch.zeros_like(combined_mask, dtype=query_states.dtype)
            float_mask = float_mask.masked_fill(combined_mask, float('-inf'))
            # float_mask = float_mask.repeat_interleave(self.num_key_value_groups, dim=1)
            # key_states = repeat_kv(key_states, self.num_key_value_groups)
            # value_states = repeat_kv(value_states, self.num_key_value_groups)

            

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

            # query_states = query_states.transpose(1, 2)
            # key_states = key_states.transpose(1, 2)
            # value_states = value_states.transpose(1, 2)

            dropout_rate = self.attention_dropout if self.training else 0.0

            
            print(f"[MATRIX] Layer {self.layer_idx} | Float Mask shape: {float_mask.shape}")

            for g in range(self.num_key_value_groups):
                qh = h * self.num_key_value_groups + g
                q = query_states[:, qh:qh+1, :, :]       # [B, 1, q_len, D]
                k = key_states[:, h:h+1, :, :]           # [B, 1, S, D]
                v = value_states[:, h:h+1, :, :]         # [B, 1, S, D]
                m = float_mask                  # [B, 1, q_len, S]

                #print the shapes of q, k, v, m
                print(f"[MATRIX] GQA {g} in Layer {self.layer_idx} | q shape: {q.shape}")
                print(f"[MATRIX] GQA {g} in Layer {self.layer_idx} | k shape: {k.shape}")
                print(f"[MATRIX] GQA {g} in Layer {self.layer_idx} | v shape: {v.shape}")
                print(f"[MATRIX] GQA {g} in Layer {self.layer_idx} | m shape: {m.shape}")

                # Run SDPA for this head-group
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=m, dropout_p=0.0, is_causal=False
                )  # [B, 1, q_len, D]
                outputs.append(out)

        # Concatenate all outputs along the head dimension
        attn_output = torch.cat(outputs, dim=1)  # [B, H * num_key_value_groups, q_len, D]
        print("ATTENTION HAS BEEN COMPUTED FOR ALL HEADS")
        print(f"[MATRIX] Layer {self.layer_idx} | attn_output shape: {attn_output.shape}")

            # attn_output = F.scaled_dot_product_attention(
            #     query_states,
            #     key_states,
            #     value_states,
            #     attn_mask=float_mask,
            #     dropout_p=0.0,
            #     is_causal=False
            # )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # head_budgets = past_key_value.head_budgets[self.layer_idx]
        # print("[DECODING EVICT] Layer:", self.layer_idx, "Head Budgets:", head_budgets)

        # past_key_value = self.config.decoding_evict[self.layer_idx](head_budgets, past_key_value, self.layer_idx)

        # tmp_attn_weights = torch.matmul(query_states[..., -self.config.window_size[self.layer_idx]:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # tmp_attn_weights = nn.functional.softmax(tmp_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # past_key_value = self.config.decoding_evict[self.layer_idx](past_key_value, tmp_attn_weights, self.layer_idx)



    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    # query_states = query_states.transpose(1, 2)
    # key_states = key_states.transpose(1, 2)
    # value_states = value_states.transpose(1, 2)

    # dropout_rate = self.attention_dropout if self.training else 0.0

    # # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # # therefore the input hidden states gets silently casted in float32. Hence, we need
    # # cast them back in the correct dtype just to be sure everything works as expected.
    # # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # # in fp32. (LlamaRMSNorm handles it correctly)

    # input_dtype = query_states.dtype
    # if input_dtype == torch.float32:
    #     if torch.is_autocast_enabled():
    #         target_dtype = torch.get_autocast_gpu_dtype()
    #     # Handle the case where the model is quantized
    #     elif hasattr(self.config, "_pre_quantization_dtype"):
    #         target_dtype = self.config._pre_quantization_dtype
    #     else:
    #         target_dtype = self.q_proj.weight.dtype

    #     logger.warning_once(
    #         f"The input hidden states seems to be silently casted in float32, this might be related to"
    #         f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
    #         f" {target_dtype}."
    #     )

    #     query_states = query_states.to(target_dtype)
    #     key_states = key_states.to(target_dtype)
    #     value_states = value_states.to(target_dtype)

    # attn_output = _flash_attention_forward(
    #     query_states,
    #     key_states,
    #     value_states,
    #     attention_mask,
    #     q_len,
    #     dropout=dropout_rate,
    #     sliding_window=getattr(self, "sliding_window", None),
    #     use_top_left_mask=self._flash_attn_uses_top_left_mask,
    #     is_causal=self.is_causal,
    # )

    # attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    

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

