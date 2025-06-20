from typing import Dict, List
import torch
import numpy as np

class CompressConfig:
    def __init__(self, compress=False, cascading=False, cache_size=1024, window_size=32, hyper=None, 
                 allocation_strategy="entropy_based"):
        self.compress = compress
        self.cascading = cascading
        self.cache_size = cache_size
        self.window_size = window_size
        self.hyper = hyper
        self.allocation_strategy = allocation_strategy  # NEW
        self.head_budgets = None 
    
    def __str__(self):
        return f"Config(cache_size={self.cache_size}, window_size={self.window_size}, " \
               f"allocation={self.allocation_strategy}, hyper={self.hyper})"


def calculate_entropy(attention_scores):
    attention_scores = attention_scores.to(torch.float32)
    entropy = -torch.sum(attention_scores * torch.log(attention_scores + 1e-10))  
    entropy= entropy.to(dtype=torch.float32)
    return entropy

def adjust_budgets(budget_list, total_budget, seq_len, layer_nums):

    budget_list = np.array(budget_list, dtype=int)
    # Limit the budget of all layers to not exceed seq_len
    excess = np.maximum(budget_list - seq_len, 0)
    budget_list = np.minimum(budget_list, seq_len)

    # Adjust excess budget
    total_excess = np.sum(excess)

    if total_excess > 0:

        valid_indices = budget_list < seq_len
        num_valid = np.sum(valid_indices)

        if num_valid > 0:
            
            distribute_per_layer = total_excess // num_valid
            remainder = total_excess % num_valid

            budget_list[valid_indices] += distribute_per_layer
            budget_list[np.where(valid_indices)[0][:remainder]] += 1

    # Ensure total budget equals total_budget
    current_total_budget = np.sum(budget_list)
    budget_diff = total_budget - current_total_budget

    if budget_diff != 0:
        if budget_diff > 0:
            valid_indices = budget_list < seq_len  
        else:
            valid_indices = budget_list > 1  

        num_valid = np.sum(valid_indices)

        if num_valid > 0:
            adjust_per_layer = abs(budget_diff) // num_valid
            remainder = abs(budget_diff) % num_valid

            if budget_diff > 0:
                budget_list[valid_indices] += adjust_per_layer
                budget_list[np.where(valid_indices)[0][:remainder]] += 1
            else:
                budget_list[valid_indices] -= adjust_per_layer
                budget_list[np.where(valid_indices)[0][:remainder]] -= 1

    return budget_list.tolist()

def compute_head_budgets(pref_scores: List[torch.Tensor], total_budget: int) -> Dict[int, List[int]]:
    """
    Compute per-head budgets across all layers.

    Args:
        pref_scores: List of Tensor[H] (one per layer)
        total_budget: total number of tokens to keep across all heads

    Returns:
        Dict[layer_idx] = List[head_budgets]
    """

    # for i, score in enumerate(pref_scores):
    #     if not isinstance(score, torch.Tensor):
    #         print(f"[ERROR] pref_score[{i}] is not a tensor: {type(score)}")
    #     elif torch.isnan(score).any():
    #         print(f"[ERROR] pref_score[{i}] contains NaNs")
    #     elif score.device != pref_scores[0].device:
    #         print(f"[ERROR] pref_score[{i}] is on device {score.device}, expected {pref_scores[0].device}")
    #     elif score.dim() != 1:
    #         print(f"[ERROR] pref_score[{i}] has wrong shape: {score.shape}")

    flat_scores = torch.cat(pref_scores)  # shape: [L * H]
    normed = flat_scores / flat_scores.sum()
    raw_budgets = (normed * total_budget).long()

    out = {}
    offset = 0
    for layer_idx, layer_score in enumerate(pref_scores):
        H = layer_score.shape[0]
        out[layer_idx] = raw_budgets[offset:offset+H].tolist()
        offset += H

    return out

# def compute_head_budgets_dynamic(pref_scores: List[torch.Tensor], total_budget: int, allocation_strategy: str = "entropy_based"):
#     """
#     Dynamic budget allocation with Multi-GPU Support
#     """
#     if not pref_scores:
#         return {}
    
#     # Multi-GPU device alignment
#     device_aligned_scores = []
#     target_device = pref_scores[0].device
    
#     for scores in pref_scores:
#         if scores.device != target_device:
#             scores = scores.to(target_device)
#         device_aligned_scores.append(scores)
    
#     if allocation_strategy == "entropy_based":
#         # Use device-aligned scores for computation
#         flat_scores = torch.cat(device_aligned_scores)  # shape: [L * H]
        
#         # Normalize scores to get allocation weights
#         allocation_weights = flat_scores / flat_scores.sum()
#         raw_budgets = (allocation_weights * total_budget).long()
        
#         # Ensure minimum budget per head
#         min_budget_per_head = 1
#         total_heads = len(flat_scores)
#         min_total = min_budget_per_head * total_heads
        
#         if total_budget < min_total:
#             raw_budgets = torch.full_like(raw_budgets, min_budget_per_head)
#         else:
#             raw_budgets = torch.clamp(raw_budgets, min=min_budget_per_head)
#             # Adjust to meet total budget constraint
#             current_total = raw_budgets.sum()
#             if current_total != total_budget:
#                 diff = total_budget - current_total
#                 if diff > 0:
#                     _, top_indices = torch.topk(flat_scores, min(abs(diff), len(flat_scores)))
#                     raw_budgets[top_indices[:diff]] += 1
#                 else:
#                     _, bottom_indices = torch.topk(flat_scores, min(abs(diff), len(flat_scores)), largest=False)
#                     for i in range(abs(diff)):
#                         if raw_budgets[bottom_indices[i]] > min_budget_per_head:
#                             raw_budgets[bottom_indices[i]] -= 1
    
#     # Convert back to per-layer format
#     out = {}
#     offset = 0
#     for layer_idx, layer_score in enumerate(device_aligned_scores):
#         H = layer_score.shape[0]
#         layer_budgets = raw_budgets[offset:offset+H].cpu().tolist()  # Move to CPU for storage
#         out[layer_idx] = layer_budgets
#         offset += H
    
   
#     return out

def compute_head_budgets_dynamic(pref_scores: List[torch.Tensor], total_budget: int, allocation_strategy: str = "entropy_based", max_seq_len: int = None):
    """
    Dynamic budget allocation with Multi-GPU Support
    """
    if not pref_scores:
        return {}
    
    # Multi-GPU device alignment
    device_aligned_scores = []
    target_device = pref_scores[0].device
    
    for scores in pref_scores:
        if scores.device != target_device:
            scores = scores.to(target_device)
        device_aligned_scores.append(scores)
    
    if allocation_strategy == "entropy_based":
        # Use device-aligned scores for computation
        flat_scores = torch.cat(device_aligned_scores)  # shape: [L * H]
        
        # Normalize scores to get allocation weights
        allocation_weights = flat_scores / flat_scores.sum()
        raw_budgets = (allocation_weights * total_budget).long()
        
        # Ensure minimum budget per head
        min_budget_per_head = 1
        total_heads = len(flat_scores)
        min_total = min_budget_per_head * total_heads
        
        if total_budget < min_total:
            raw_budgets = torch.full_like(raw_budgets, min_budget_per_head)
        else:
            raw_budgets = torch.clamp(raw_budgets, min=min_budget_per_head)
            
            # SEQUENCE LENGTH AWARENESS: Clamp budgets to max_seq_len if provided
            if max_seq_len is not None:
                raw_budgets = torch.clamp(raw_budgets, max=max_seq_len)
            
            # Adjust to meet total budget constraint after clamping
            current_total = raw_budgets.sum()
            if current_total != total_budget:
                diff = total_budget - current_total
                if diff > 0:
                    # Only add to heads that aren't at max_seq_len limit
                    available_mask = raw_budgets < (max_seq_len if max_seq_len is not None else float('inf'))
                    if available_mask.any():
                        available_indices = torch.where(available_mask)[0]
                        _, top_indices = torch.topk(flat_scores[available_indices], min(abs(diff), len(available_indices)))
                        actual_indices = available_indices[top_indices[:diff]]
                        raw_budgets[actual_indices] += 1
                else:
                    _, bottom_indices = torch.topk(flat_scores, min(abs(diff), len(flat_scores)), largest=False)
                    for i in range(abs(diff)):
                        if raw_budgets[bottom_indices[i]] > min_budget_per_head:
                            raw_budgets[bottom_indices[i]] -= 1
    
    # Convert back to per-layer format
    out = {}
    offset = 0
    for layer_idx, layer_score in enumerate(device_aligned_scores):
        H = layer_score.shape[0]
        layer_budgets = raw_budgets[offset:offset+H].cpu().tolist()  # Move to CPU for storage
        out[layer_idx] = layer_budgets
        offset += H
    
    return out
