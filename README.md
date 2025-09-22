# Matrix-KV: Head-wise Budget Allocation for KV Cache Optimization

## Overview

Matrix-KV extends KV cache management with **head-level entropy-based budget allocation**, moving beyond uniform layer-wise allocation to optimize cache distribution at individual attention head granularity.

## What We're Achieving

**Head-wise Cache Optimization**: Instead of allocating uniform budgets to all heads within a layer, Matrix-KV computes individual preference scores for each attention head and allocates cache budgets proportionally based on attention entropy patterns.

**Entropy-driven Resource Distribution**: High-entropy heads (dispersed attention) receive more cache budget, while low-entropy heads (concentrated attention) get minimal allocation, eliminating systematic over-allocation.

## Current Implementation Status

### ✅ Completed

- **Head-level preference calculation**: Individual entropy scoring for each attention head across all layers
- **Variable-length FlashAttention integration**: Required for actual KV cache eviction with different head budgets
- **Token eviction implementation**: Currently only allocating budgets, not performing actual cache reduction

### Get Started
To use Matrix-KV, you will first have to download the LongBench dataset. Please run the file `download_dataset.py` to download the dataset. After that, you can run this program just like CAKE. 

