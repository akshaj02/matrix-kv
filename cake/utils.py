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

def compute_head_budgets_dynamic(pref_scores: List[torch.Tensor], total_budget: int, allocation_strategy: str = "entropy_based") -> Dict[int, List[int]]:
    """
    Dynamic budget allocation based on attention dispersion patterns.
    
    Args:
        pref_scores: List of Tensor[H] (one per layer) - higher score = more dispersed attention
        total_budget: total number of tokens to keep across all heads
        allocation_strategy: "entropy_based", "proportional", or "adaptive"
    
    Returns:
        Dict[layer_idx] = List[head_budgets]
    """
    
    if allocation_strategy == "entropy_based":
        # print the allocation strategy being used
        # print(f"[CAKE] Using dynamic budget allocation strategy: {allocation_strategy}")
        # More dispersed heads (higher entropy/preference) get proportionally more budget
        flat_scores = torch.cat(pref_scores)  # shape: [L * H]
        
        # Normalize scores to get allocation weights
        allocation_weights = flat_scores / flat_scores.sum()
        raw_budgets = (allocation_weights * total_budget).long()
        
        # Ensure minimum budget per head
        min_budget_per_head = 1
        total_heads = len(flat_scores)
        min_total = min_budget_per_head * total_heads
        
        if total_budget < min_total:
            print(f"[WARNING] Total budget {total_budget} < minimum required {min_total}")
            raw_budgets = torch.full_like(raw_budgets, min_budget_per_head)
        else:
            raw_budgets = torch.clamp(raw_budgets, min=min_budget_per_head)
            # Adjust to meet total budget constraint
            current_total = raw_budgets.sum()
            if current_total != total_budget:
                diff = total_budget - current_total
                # Distribute difference proportionally
                if diff > 0:
                    # Add extra budget to highest scoring heads
                    _, top_indices = torch.topk(flat_scores, min(abs(diff), len(flat_scores)))
                    raw_budgets[top_indices[:diff]] += 1
                else:
                    # Remove budget from lowest scoring heads (but keep minimum)
                    _, bottom_indices = torch.topk(flat_scores, min(abs(diff), len(flat_scores)), largest=False)
                    for i in range(abs(diff)):
                        if raw_budgets[bottom_indices[i]] > min_budget_per_head:
                            raw_budgets[bottom_indices[i]] -= 1
    
    # Convert back to per-layer format
    out = {}
    offset = 0
    for layer_idx, layer_score in enumerate(pref_scores):
        H = layer_score.shape[0]
        layer_budgets = raw_budgets[offset:offset+H].tolist()
        out[layer_idx] = layer_budgets
        
        # Log dynamic allocation for this layer
        total_layer_budget = sum(layer_budgets)
        avg_budget = total_layer_budget / H
        max_budget = max(layer_budgets)
        min_budget = min(layer_budgets)
        
        # print(f"[CAKE] Layer {layer_idx} Dynamic Allocation:")
        # print(f"  Total: {total_layer_budget}, Avg: {avg_budget:.1f}, Range: [{min_budget}-{max_budget}]")
        # print(f"  Budgets: {layer_budgets}")
        # print(f"  Pref Scores: {layer_score.tolist()}")
        
        offset += H
    
    return out

def compute_head_budgets_vanilla_cake(pref_scores: List[torch.Tensor], total_budget: int) -> Dict[int, List[int]]:
    """
    Replicate vanilla CAKE allocation from per-head preference scores
    """
    print(f"[CAKE] Using vanilla CAKE allocation strategy (layer-wise uniform)")
    
    # Step 1: Aggregate per-head scores to layer-level scores
    layer_pref_scores = []
    for layer_head_scores in pref_scores:
        # Average across heads to get layer-level preference
        layer_avg_score = layer_head_scores.mean()
        layer_pref_scores.append(layer_avg_score)
    
    layer_pref_tensor = torch.stack(layer_pref_scores)
    
    # Step 2: Allocate budget to layers based on layer-level scores
    layer_weights = layer_pref_tensor / layer_pref_tensor.sum()
    layer_budgets = (layer_weights * total_budget).long()
    
    # Step 3: Distribute each layer's budget uniformly across its heads
    out = {}
    for layer_idx, layer_budget in enumerate(layer_budgets):
        H = pref_scores[layer_idx].shape[0]  # Number of heads (8 for Llama 3.1-8B)
        uniform_head_budget = layer_budget.item() // H
        remainder = layer_budget.item() % H
        
        # All heads get same budget within layer
        head_budgets = [uniform_head_budget] * H
        for i in range(remainder):
            head_budgets[i] += 1
            
        out[layer_idx] = head_budgets
        
        print(f"[CAKE] Layer {layer_idx}: Layer score={layer_pref_scores[layer_idx]:.3f}, "
              f"Total budget={layer_budget.item()}, Per head={uniform_head_budget}")
    
    return out



def analyze_budget_distribution(head_budgets: Dict[int, List[int]], pref_scores: List[torch.Tensor]):
    """
    Analyze and log budget distribution statistics
    """
    print("\n[CAKE] Budget Distribution Analysis:")
    print("=" * 60)
    
    total_budget = 0
    total_heads = 0
    
    for layer_idx, budgets in head_budgets.items():
        layer_total = sum(budgets)
        layer_scores = pref_scores[layer_idx]
        
        # Calculate statistics
        budget_std = np.std(budgets)
        score_std = layer_scores.std().item()
        
        # Calculate correlation between scores and budgets
        correlation = np.corrcoef(layer_scores.cpu().numpy(), budgets)[0, 1]
        
        print(f"Layer {layer_idx:2d}: Budget={layer_total:4d}, Heads={len(budgets):2d}, "
              f"Std={budget_std:5.2f}, Score-Budget Corr={correlation:5.3f}")
        
        total_budget += layer_total
        total_heads += len(budgets)
    
    avg_budget_per_head = total_budget / total_heads
    print(f"\nTotal Budget: {total_budget}, Avg per Head: {avg_budget_per_head:.1f}")
    print("=" * 60)

