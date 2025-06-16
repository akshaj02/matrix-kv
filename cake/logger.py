import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch

class LongBenchBudgetLogger:
    def __init__(self, save_dir: str = "longbench_budget_logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-dataset storage
        self.dataset_stats = {}
        self.current_dataset = None
        self.current_sample = None
        
        # Global aggregation across all datasets
        self.global_stats = {
            'datasets_processed': [],
            'total_samples': 0,
            'allocation_strategies': set(),
            'head_level_stats': defaultdict(list)
        }
    
    def start_dataset(self, dataset_name: str):
        """Initialize logging for a new dataset"""
        self.current_dataset = dataset_name
        self.dataset_stats[dataset_name] = {
            'samples': [],
            'dataset_summary': {},
            'allocation_patterns': defaultdict(list)
        }
        print(f"[CAKE Logger] Started logging for dataset: {dataset_name}")
    
    def start_sample(self, sample_data: dict):
        """Start logging for a new sample"""
        sample_id = sample_data.get('_id', f"sample_{len(self.dataset_stats[self.current_dataset]['samples'])}")
        
        self.current_sample = {
            'sample_id': sample_id,
            'dataset': self.current_dataset,
            'input_length': len(sample_data.get('input', '')),
            'context_length': len(sample_data.get('context', '')),
            'layers': {},
            'sample_summary': {},
            'metadata': {
                'language': sample_data.get('language', 'unknown'),
                'length': sample_data.get('length', 0),
                'all_classes': sample_data.get('all_classes', [])
            }
        }
    
    def log_layer_allocation(self, layer_idx: int, head_budgets: List[int], 
                           pref_scores: torch.Tensor, allocation_strategy: str,
                           available_tokens: int, window_size: int = 32):
        """Log detailed head-wise allocation for a layer"""
        
        if self.current_sample is None:
            return
        
        # Convert tensors to numpy for calculations
        budgets = np.array(head_budgets)
        scores = pref_scores.cpu().numpy() if torch.is_tensor(pref_scores) else np.array(pref_scores)
        
        # Calculate comprehensive statistics
        layer_stats = {
            'layer_idx': layer_idx,
            'allocation_strategy': allocation_strategy,
            'num_heads': len(head_budgets),
            'available_tokens': available_tokens,
            'window_size': window_size,
            
            # Budget statistics
            'head_budgets': head_budgets,
            'total_budget': int(budgets.sum()),
            'budget_mean': float(budgets.mean()),
            'budget_std': float(budgets.std()),
            'budget_min': int(budgets.min()),
            'budget_max': int(budgets.max()),
            'budget_variance': float(np.var(budgets)),
            'budget_range': int(budgets.max() - budgets.min()),
            
            # Preference score statistics
            'preference_scores': scores.tolist(),
            'score_mean': float(scores.mean()),
            'score_std': float(scores.std()),
            'score_min': float(scores.min()),
            'score_max': float(scores.max()),
            
            # Correlation and efficiency metrics
            'score_budget_correlation': float(np.corrcoef(scores, budgets)[0, 1]) if len(scores) > 1 else 1.0,
            'allocation_efficiency': float(budgets.std() / budgets.mean()) if budgets.mean() > 0 else 0.0,
            
            # Head-level details
            'head_details': []
        }
        
        # Calculate per-head metrics
        uniform_budget = layer_stats['total_budget'] // len(head_budgets)
        
        for head_idx, (budget, score) in enumerate(zip(head_budgets, scores)):
            utilization = (budget / available_tokens) * 100 if available_tokens > 0 else 0
            efficiency = budget / uniform_budget if uniform_budget > 0 else 1.0
            
            head_detail = {
                'head_idx': head_idx,
                'budget': budget,
                'preference_score': float(score),
                'utilization_percent': float(utilization),
                'efficiency_vs_uniform': float(efficiency),
                'budget_rank': int(np.argsort(np.argsort(-budgets))[head_idx]) + 1,  # Rank by budget (1=highest)
                'score_rank': int(np.argsort(np.argsort(-scores))[head_idx]) + 1     # Rank by score (1=highest)
            }
            layer_stats['head_details'].append(head_detail)
        
        # Store layer statistics
        self.current_sample['layers'][layer_idx] = layer_stats
        
        # Update dataset-level patterns
        self.dataset_stats[self.current_dataset]['allocation_patterns']['correlations'].append(
            layer_stats['score_budget_correlation']
        )
        self.dataset_stats[self.current_dataset]['allocation_patterns']['efficiencies'].append(
            layer_stats['allocation_efficiency']
        )
    
    def end_sample(self):
        """Finalize current sample and compute summary statistics"""
        if self.current_sample is None:
            return
        
        # Calculate sample-level summary
        all_budgets = []
        all_scores = []
        all_correlations = []
        total_budget = 0
        
        for layer_data in self.current_sample['layers'].values():
            all_budgets.extend(layer_data['head_budgets'])
            all_scores.extend(layer_data['preference_scores'])
            all_correlations.append(layer_data['score_budget_correlation'])
            total_budget += layer_data['total_budget']
        
        if all_budgets:
            all_budgets = np.array(all_budgets)
            all_scores = np.array(all_scores)
            
            self.current_sample['sample_summary'] = {
                'total_budget_all_layers': total_budget,
                'num_layers': len(self.current_sample['layers']),
                'total_heads': len(all_budgets),
                'avg_budget_per_head': float(all_budgets.mean()),
                'global_budget_std': float(all_budgets.std()),
                'global_score_budget_correlation': float(np.corrcoef(all_scores, all_budgets)[0, 1]),
                'avg_layer_correlation': float(np.mean(all_correlations)),
                'budget_concentration_index': float(np.sum((all_budgets / all_budgets.sum()) ** 2)),  # Herfindahl index
                'allocation_diversity': float(1 - (all_budgets.std() / all_budgets.mean())) if all_budgets.mean() > 0 else 0
            }
        
        # Store sample in dataset
        self.dataset_stats[self.current_dataset]['samples'].append(self.current_sample)
        self.global_stats['total_samples'] += 1
        
        # Reset current sample
        self.current_sample = None
    
    def end_dataset(self):
        """Finalize current dataset and compute dataset-level statistics"""
        if self.current_dataset is None:
            return
        
        dataset_samples = self.dataset_stats[self.current_dataset]['samples']
        
        if not dataset_samples:
            return
        
        # Calculate dataset-level summary
        dataset_budgets = []
        dataset_correlations = []
        sample_lengths = []
        
        for sample in dataset_samples:
            if 'sample_summary' in sample:
                dataset_budgets.append(sample['sample_summary']['total_budget_all_layers'])
                dataset_correlations.append(sample['sample_summary']['global_score_budget_correlation'])
                sample_lengths.append(sample['input_length'] + sample['context_length'])
        
        self.dataset_stats[self.current_dataset]['dataset_summary'] = {
            'num_samples': len(dataset_samples),
            'avg_total_budget': float(np.mean(dataset_budgets)) if dataset_budgets else 0,
            'budget_std_across_samples': float(np.std(dataset_budgets)) if dataset_budgets else 0,
            'avg_correlation_across_samples': float(np.mean(dataset_correlations)) if dataset_correlations else 0,
            'correlation_std': float(np.std(dataset_correlations)) if dataset_correlations else 0,
            'avg_sample_length': float(np.mean(sample_lengths)) if sample_lengths else 0,
            'length_budget_correlation': float(np.corrcoef(sample_lengths, dataset_budgets)[0, 1]) if len(sample_lengths) > 1 else 0
        }
        
        # Save dataset-specific file
        dataset_file = self.save_dir / f"{self.current_dataset}_budget_stats.json"
        with open(dataset_file, 'w') as f:
            json.dump(self.dataset_stats[self.current_dataset], f, indent=2)
        
        print(f"[CAKE Logger] Saved {self.current_dataset} statistics to {dataset_file}")
        
        # Update global stats
        self.global_stats['datasets_processed'].append(self.current_dataset)
        self.current_dataset = None
    
    def save_comprehensive_report(self):
        """Save comprehensive report across all datasets"""
        
        # Global summary across all datasets
        global_summary = {
            'evaluation_overview': {
                'datasets_processed': list(self.global_stats['datasets_processed']),
                'total_samples': self.global_stats['total_samples'],
                'total_datasets': len(self.global_stats['datasets_processed'])
            },
            'cross_dataset_analysis': self._compute_cross_dataset_analysis(),
            'dataset_summaries': {name: stats['dataset_summary'] 
                                for name, stats in self.dataset_stats.items()}
        }
        
        # Save global report
        global_file = self.save_dir / "longbench_global_budget_report.json"
        with open(global_file, 'w') as f:
            json.dump(global_summary, f, indent=2)
        
        # Create CSV exports for analysis
        self._export_analysis_csvs()
        
        print(f"[CAKE Logger] Comprehensive report saved to {self.save_dir}")
        return global_summary
    
    def _compute_cross_dataset_analysis(self):
        """Compute analysis across all datasets"""
        dataset_metrics = {}
        
        for dataset_name, dataset_data in self.dataset_stats.items():
            if 'dataset_summary' in dataset_data:
                dataset_metrics[dataset_name] = dataset_data['dataset_summary']
        
        if not dataset_metrics:
            return {}
        
        # Cross-dataset comparisons
        budgets = [m['avg_total_budget'] for m in dataset_metrics.values()]
        correlations = [m['avg_correlation_across_samples'] for m in dataset_metrics.values()]
        sample_lengths = [m['avg_sample_length'] for m in dataset_metrics.values()]
        
        return {
            'budget_variation_across_datasets': {
                'mean': float(np.mean(budgets)),
                'std': float(np.std(budgets)),
                'min_dataset': min(dataset_metrics.keys(), key=lambda k: dataset_metrics[k]['avg_total_budget']),
                'max_dataset': max(dataset_metrics.keys(), key=lambda k: dataset_metrics[k]['avg_total_budget'])
            },
            'correlation_patterns': {
                'mean_correlation': float(np.mean(correlations)),
                'correlation_consistency': float(np.std(correlations)),
                'best_correlation_dataset': max(dataset_metrics.keys(), key=lambda k: dataset_metrics[k]['avg_correlation_across_samples'])
            },
            'length_budget_relationship': {
                'length_budget_correlation': float(np.corrcoef(sample_lengths, budgets)[0, 1]) if len(budgets) > 1 else 0,
                'datasets_by_efficiency': sorted(dataset_metrics.keys(), 
                                               key=lambda k: dataset_metrics[k]['avg_total_budget'] / dataset_metrics[k]['avg_sample_length'])
            }
        }
    
    def _export_analysis_csvs(self):
        """Export detailed data for external analysis"""
        
        # Head-level analysis across all datasets
        head_rows = []
        sample_rows = []
        
        for dataset_name, dataset_data in self.dataset_stats.items():
            for sample in dataset_data['samples']:
                # Sample-level row
                sample_row = {
                    'dataset': dataset_name,
                    'sample_id': sample['sample_id'],
                    'input_length': sample['input_length'],
                    'context_length': sample['context_length'],
                    'total_length': sample['input_length'] + sample['context_length'],
                    **sample.get('sample_summary', {})
                }
                sample_rows.append(sample_row)
                
                # Head-level rows
                for layer_idx, layer_data in sample['layers'].items():
                    for head_detail in layer_data['head_details']:
                        head_row = {
                            'dataset': dataset_name,
                            'sample_id': sample['sample_id'],
                            'layer_idx': layer_idx,
                            'allocation_strategy': layer_data['allocation_strategy'],
                            **head_detail,
                            'layer_total_budget': layer_data['total_budget'],
                            'layer_correlation': layer_data['score_budget_correlation']
                        }
                        head_rows.append(head_row)
        
        # Save CSVs
        if head_rows:
            head_df = pd.DataFrame(head_rows)
            head_df.to_csv(self.save_dir / "longbench_head_level_analysis.csv", index=False)
        
        if sample_rows:
            sample_df = pd.DataFrame(sample_rows)
            sample_df.to_csv(self.save_dir / "longbench_sample_level_analysis.csv", index=False)
        
        print(f"[CAKE Logger] Exported analysis CSVs to {self.save_dir}")
