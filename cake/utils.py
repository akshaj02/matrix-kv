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
        self.allocation_strategy = allocation_strategy
        self.head_budgets = None  # Will store pre-computed budgets
        self.layer_budgets = None  # Will store layer-level budgets 
    
    def __str__(self):
        return f"Config(cache_size={self.cache_size}, window_size={self.window_size}, " \
               f"allocation={self.allocation_strategy}, hyper={self.hyper})"


def precompute_static_head_budgets(
    cache_size: int, 
    window_size: int, 
    num_layers: int, 
    num_heads: int,
    importance_file_path: str = "./importance_scores/meta_llama_Meta_Llama_3.1_8B_Instruct_importance.npy"
) -> Dict[int, List[int]]:
    """
    Pre-compute head budgets for all layers based on static importance scores.
    This function should be called once during model initialization.
    
    Args:
        cache_size: Total cache size (e.g., 1024)
        window_size: Window size to keep recent tokens (e.g., 32)  
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        importance_file_path: Path to the .npy file with importance scores
        
    Returns:
        Dict[layer_idx] = List[head_budgets] for each layer
    """
    # print(f"[CAKE] Pre-computing static head budgets from {importance_file_path}")
    
    # Load importance scores once
    importance_scores = np.load(importance_file_path)
    importance_scores = torch.from_numpy(importance_scores)
    
    # Validate dimensions
    expected_shape = (num_layers, num_heads)
    if importance_scores.shape != expected_shape:
        raise ValueError(f"Importance scores shape {importance_scores.shape} doesn't match expected {expected_shape}")
    
    # Calculate total budget
    total_budget = (cache_size - window_size) * num_layers * num_heads
    max_seq_len = cache_size
    
    # Flatten scores and compute allocation weights
    flat_importance = importance_scores.flatten()  # shape: [L * H]
    allocation_weights = flat_importance / flat_importance.sum()
    
    # Compute raw budgets
    raw_budgets = (allocation_weights * total_budget).long()
    
    # Ensure minimum budget per head
    min_importance = flat_importance.min()
    min_weight = min_importance / flat_importance.sum()
    min_budget_per_head = max(1, int(min_weight * total_budget))
    total_heads = len(flat_importance)
    min_total = min_budget_per_head * total_heads
    
    if total_budget < min_total:
        # If total budget is too small, scale down proportionally
        scale_factor = total_budget / min_total
        raw_budgets = (flat_importance / flat_importance.sum() * total_budget * scale_factor).long()
        raw_budgets = torch.maximum(raw_budgets, torch.tensor(1))
    else:
        raw_budgets = torch.maximum(raw_budgets, torch.tensor(min_budget_per_head))
        
        # Clamp budgets to max_seq_len
        raw_budgets = torch.minimum(raw_budgets, torch.tensor(max_seq_len))
        
        # Adjust to meet total budget constraint after clamping
        current_total = raw_budgets.sum()
        budget_diff = total_budget - current_total
        
        if budget_diff != 0:
            importance_order = torch.argsort(flat_importance)
            
            if budget_diff > 0:
                # Add extra budget to most important heads that aren't at max limit
                available_mask = raw_budgets < max_seq_len
                available_indices = torch.where(available_mask)[0]
                
                if len(available_indices) > 0:
                    available_importance = flat_importance[available_indices]
                    sorted_indices = torch.argsort(available_importance, descending=True)
                    available_by_importance = available_indices[sorted_indices]
                    
                    for i in range(min(budget_diff, len(available_by_importance))):
                        raw_budgets[available_by_importance[i]] += 1
                        
            else:  # budget_diff < 0
                # Remove budget from least important heads
                for i in range(min(abs(budget_diff), total_heads)):
                    head_idx = importance_order[i]
                    if raw_budgets[head_idx] > min_budget_per_head:
                        raw_budgets[head_idx] -= 1
    
    # Convert back to per-layer format
    head_budgets = {}
    for layer_idx in range(num_layers):
        start_idx = layer_idx * num_heads
        end_idx = start_idx + num_heads
        layer_budgets = raw_budgets[start_idx:end_idx].tolist()
        head_budgets[layer_idx] = layer_budgets
    
    # print(f"[CAKE] Pre-computed budgets for {num_layers} layers, {num_heads} heads each")
    # print(f"[CAKE] Total budget: {total_budget}, Actual allocated: {sum(sum(budgets) for budgets in head_budgets.values())}")
    
    return head_budgets


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

    if allocation_strategy == "static":
        # open the file ./importance_scores/meta_llama_Meta_Llama_3.1_8B_Instruct_importance.npy and read the data
        importance_scores = np.load("./importance_scores/meta_llama_Meta_Llama_3.1_8B_Instruct_importance.npy")

        # convert to torch tensor
        importance_scores = torch.from_numpy(importance_scores)
        # make all the importance scores the same value to essentially remove any ratios 
        # importance_scores = np.full(importance_scores.shape, 1.0)  # Set all scores to 1.0
    
        # Get dimensions
        num_layers, num_heads = importance_scores.shape
        # flatten the scores
        flat_importance = importance_scores.flatten()  # shape: [L * H]

        min_importance = flat_importance.min()
        # print(f"[CAKE] Min Importance: {min_importance}")
        # max_importance = flat_importance.max()
        # print(f"[CAKE] Max Importance: {max_importance}")
        total_heads = len(flat_importance)
        
        allocation_weights = flat_importance / flat_importance.sum()

        # print(f"[CAKE] Allocation Weights: {allocation_weights}")
        min_weight = min_importance / flat_importance.sum()
        # print(f"[CAKE] Min Weight: {min_weight}")
        # max_weight = max_importance / flat_importance.sum()
        # print(f"[CAKE] Max Weight: {max_weight}")
        # almost always it will be more than 1
        min_budget_per_head = max(1, int(min_weight * total_budget))
        # print(f"[CAKE] Min Budget Per Head: {min_budget_per_head}")

        # max_budget_per_head = max(1, int(max_weight * total_budget))
        # print(f"[CAKE] Max Budget Per Head: {max_budget_per_head}")

        raw_budgets = (allocation_weights * total_budget).long()  # shape: [L * H]
        min_total = min_budget_per_head * total_heads
        # print(f"[CAKE] Min Total: {min_total}")

        if total_budget < min_total:
            # If total budget is too small, scale down proportionally
            scale_factor = total_budget / min_total
            raw_budgets = (flat_importance / flat_importance.sum() * total_budget * scale_factor).long()
            raw_budgets = torch.maximum(raw_budgets, torch.tensor(1, device=raw_budgets.device))
        else:
            raw_budgets = torch.maximum(raw_budgets, torch.tensor(min_budget_per_head, device=raw_budgets.device))
            
            # SEQUENCE LENGTH AWARENESS: Clamp budgets to max_seq_len if provided
            if max_seq_len is not None:
                raw_budgets = torch.minimum(raw_budgets, torch.tensor(max_seq_len, device=raw_budgets.device))
            
            # Adjust to meet total budget constraint after clamping
            current_total = raw_budgets.sum()
            budget_diff = total_budget - current_total
            
            if budget_diff != 0:
                # Sort by importance for redistribution
                importance_order = torch.argsort(flat_importance)
                
                if budget_diff > 0:
                    # Add extra budget to most important heads that aren't at max limit
                    max_limit = max_seq_len if max_seq_len is not None else float('inf')
                    available_mask = raw_budgets < max_limit
                    available_indices = torch.where(available_mask)[0]
                    
                    if len(available_indices) > 0:
                        # Sort available indices by importance (descending)
                        # Sort available indices by importance (descending)
                        available_importance = flat_importance[available_indices]
                        sorted_indices = torch.argsort(available_importance, descending=True)
                        available_by_importance = available_indices[sorted_indices]
                                    
                        # Distribute extra budget
                        for i in range(min(budget_diff, len(available_by_importance))):
                            raw_budgets[available_by_importance[i]] += 1
                            
                else:  # budget_diff < 0
                    # Remove budget from least important heads, but respect the dynamic minimum
                    for i in range(min(abs(budget_diff), total_heads)):
                        head_idx = importance_order[i]
                        if raw_budgets[head_idx] > min_budget_per_head:
                            raw_budgets[head_idx] -= 1

    # Convert back to per-layer format
    out = {}
    offset = 0
    for layer_idx, layer_score in enumerate(device_aligned_scores):
        H = layer_score.shape[0]
        layer_budgets = raw_budgets[offset:offset+H].cpu().tolist()  # Move to CPU for storage
        out[layer_idx] = layer_budgets
        offset += H

    # out = {}
    # offset = 0
    # for layer_idx in range(num_layers):
    #     layer_budgets = raw_budgets[offset : offset + num_heads]  # no need for .cpu().tolist() if already on CPU
    #     out[layer_idx] = layer_budgets.tolist() if hasattr(layer_budgets, 'tolist') else list(layer_budgets)
    #     offset += num_heads
    
    return out

# run co




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