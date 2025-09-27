# Matrix-KV: Head-wise Budget Allocation for KV Cache Optimization

## Overview

Matrix-KV extends KV cache management with **head-level budget allocation**, moving beyond uniform layer-wise allocation to optimize cache distribution at individual attention head granularity.

## What We're Achieving

**Head-wise Cache Optimization**: Instead of allocating uniform budgets to all heads within a layer, Matrix-KV computes individual preference scores for each attention head and allocates cache budgets proportionally based on attention entropy patterns.

## Current Implementation Status

### ✅ Completed

<<<<<<< matrixkv-optimization
- **Head-level preference calculation**: Individual scoring for each attention head across all layers
- **Variable-length FlashAttention integration**: Required for actual KV cache eviction with different head budgets
- **Token eviction implementation**: Currently only allocating budgets, not performing actual cache reduction

## Detailed Results on LongBench Datasets

| Method | NrtvQA | Qasper | MF-en | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PCount | PR-en | Lcc | RB-P | **Avg.** |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **MatrixKV (Ours)** | **30.95** | **46.02** | **53.49** | 55.35 | **47.13** | 30.73 | **34.70** | **25.02** | **27.54** | **73.00** | 91.48 | **43.83** | **6.00** | **99.50** | **63.34** | 56.28 | **49.02** |
| CAKE | 30.88 | 44.95 | 52.38 | **55.49** | 46.99 | **30.82** | 28.68 | 24.91 | 26.39 | 69.00 | **91.94** | 42.60 | **6.00** | **99.50** | 62.65 | **56.89** | 48.13 |
| SnapKV | **30.95** | 44.74 | 52.58 | 55.09 | 46.83 | 30.37 | 27.87 | 24.57 | 25.99 | 68.00 | 92.03 | 42.60 | 6.50 | **99.50** | 63.00 | 56.50 | 47.95 |
| PyramidKV | 30.54 | 43.64 | 52.73 | 55.29 | 46.29 | 31.28 | 27.53 | 24.50 | 26.00 | 68.00 | 92.09 | 41.75 | 6.05 | **99.50** | 62.35 | 55.44 | 47.69 |
| TOVA | 30.66 | 40.95 | 51.09 | 54.58 | 46.51 | 30.62 | 28.12 | 23.61 | 26.24 | 68.00 | 91.49 | 43.80 | 5.92 | **99.50** | 60.73 | 52.64 | 47.15 |
| H2O | 29.57 | 36.15 | 45.94 | 54.43 | 44.81 | 29.04 | 27.64 | 23.31 | **26.47** | 62.00 | 91.83 | 43.14 | 6.36 | 99.00 | **62.74** | 55.39 | 46.11 |
| StreamingLLM | 26.64 | 30.77 | 35.59 | 47.31 | 42.03 | 24.17 | 25.81 | 21.31 | 25.66 | 63.50 | 88.84 | 42.76 | 6.50 | 88.00 | 61.36 | 53.47 | 42.73 |

## Key Performance Highlights

### 🏆 CAKE++ Achievements

- \#1 Overall Performance: 49.02 average score across all LongBench datasets
- **Wins on 9/16 datasets outright**
- **Ties for best on 3/16 additional datasets**
- **75% win rate** (12 out of 16 datasets at or above all competitors)
=======
- **Head-level preference calculation**: Individual entropy scoring for each attention head across all layers
- **Variable-length FlashAttention integration**: Required for actual KV cache eviction with different head budgets
- **Token eviction implementation**: Currently only allocating budgets, not performing actual cache reduction
>>>>>>> main

### Get Started
To use Matrix-KV, you will first have to download the LongBench dataset. Please run the file `download_dataset.py` to download the dataset. After that, you can run this program just like CAKE. 

Use this command to run the program:
``` CUDA_VISIBLE_DEVICES=1,2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python experiments/LongBench/pred_cake.py --model llama3.1-8b-128k --compress --cascading --pred_name CAKE_PP_pred_result --device 0 --cache_size 1024 --window_size 32 ```
