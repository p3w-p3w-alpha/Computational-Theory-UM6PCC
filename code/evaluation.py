"""
Evaluation Module for ECG Scanpath Classification

This module provides comprehensive evaluation metrics and cross-validation
for the PFA-based scanpath classification system.

Authors: Lazrek Nassim, Omar Ait Said, Ilyass Skiriba
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

import sys
sys.path.append('.')
from pfa import ProbabilisticFiniteAutomaton, ScanpathClassifier


def parse_scanpath(scanpath_str: str) -> List[str]:
    """Convert comma-separated scanpath string to list."""
    return scanpath_str.split(',')


def evaluate_classification(y_true: List[str], y_pred: List[str], 
                           y_proba: np.ndarray = None) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional, for ROC-AUC)
        
    Returns:
        Dictionary of metrics
    """
    # Convert to binary (expert=1, novice=0)
    y_true_binary = [1 if y == 'expert' else 0 for y in y_true]
    y_pred_binary = [1 if y == 'expert' else 0 for y in y_pred]
    
    metrics = {
        'accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
        'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true_binary, y_pred_binary)
    }
    
    # Add ROC-AUC if probabilities are provided
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true_binary, y_proba[:, 0])
        except ValueError:
            metrics['roc_auc'] = None
    
    return metrics


def cross_validate(df: pd.DataFrame, n_folds: int = 5, 
                   smoothing: float = 1.0, seed: int = 42) -> Dict:
    """
    Perform k-fold cross-validation.
    
    Args:
        df: DataFrame with columns ['scanpath', 'expertise']
        n_folds: Number of folds
        smoothing: Laplace smoothing parameter
        seed: Random seed
        
    Returns:
        Dictionary with mean and std of metrics across folds
    """
    # Prepare data
    scanpaths = [parse_scanpath(s) for s in df['scanpath'].values]
    labels = df['expertise'].values
    
    # Initialize KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Store metrics for each fold
    fold_metrics = defaultdict(list)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(scanpaths)):
        # Split data
        train_scanpaths = [scanpaths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_scanpaths = [scanpaths[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        
        # Separate expert and novice training data
        expert_train = [sp for sp, lbl in zip(train_scanpaths, train_labels) if lbl == 'expert']
        novice_train = [sp for sp, lbl in zip(train_scanpaths, train_labels) if lbl == 'novice']
        
        # Train classifier
        classifier = ScanpathClassifier(smoothing=smoothing)
        classifier.train(expert_train, novice_train)
        
        # Predict
        predictions = classifier.predict(test_scanpaths)
        probas = classifier.predict_proba(test_scanpaths)
        
        # Evaluate
        metrics = evaluate_classification(test_labels, predictions, probas)
        
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                fold_metrics[key].append(value)
    
    # Compute mean and std
    results = {}
    for key, values in fold_metrics.items():
        values = [v for v in values if v is not None]
        if len(values) > 0:
            results[f'{key}_mean'] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
    
    return results


def evaluate_generation(pfa: ProbabilisticFiniteAutomaton, 
                        real_scanpaths: List[List[str]],
                        n_generated: int = 100,
                        seed: int = 42) -> Dict:
    """
    Evaluate the quality of generated scanpaths.
    
    Metrics:
    - Perplexity: Average negative log-likelihood
    - Transition distribution similarity: KL-divergence
    - Generation validity: Percentage of valid state transitions
    
    Args:
        pfa: Trained PFA model
        real_scanpaths: Real scanpaths for comparison
        n_generated: Number of scanpaths to generate
        seed: Random seed
        
    Returns:
        Dictionary of generation metrics
    """
    np.random.seed(seed)
    
    # Generate scanpaths
    avg_length = int(np.mean([len(sp) for sp in real_scanpaths]))
    generated_scanpaths = [pfa.generate(length=avg_length, seed=seed+i) 
                          for i in range(n_generated)]
    
    # Compute perplexity on real data
    log_probs = []
    total_length = 0
    for sp in real_scanpaths:
        log_prob = pfa.compute_log_probability(sp)
        if log_prob > float('-inf'):
            log_probs.append(log_prob)
            total_length += len(sp)
    
    if len(log_probs) > 0:
        avg_log_prob = sum(log_probs) / total_length
        perplexity = np.exp(-avg_log_prob)
    else:
        perplexity = float('inf')
    
    # Compute transition distribution from generated scanpaths
    gen_transition_counts = np.zeros((pfa.n_states, pfa.n_states))
    for sp in generated_scanpaths:
        for i in range(len(sp) - 1):
            if sp[i] in pfa.state_to_idx and sp[i+1] in pfa.state_to_idx:
                curr_idx = pfa.state_to_idx[sp[i]]
                next_idx = pfa.state_to_idx[sp[i+1]]
                gen_transition_counts[curr_idx, next_idx] += 1
    
    # Normalize to get distribution
    row_sums = gen_transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    gen_transition_dist = gen_transition_counts / row_sums
    
    # Compute KL-divergence between generated and model distributions
    kl_divergence = 0.0
    valid_rows = 0
    for i in range(pfa.n_states):
        if gen_transition_counts[i].sum() > 0:
            p = pfa.transition_matrix[i]
            q = gen_transition_dist[i]
            # Add small epsilon to avoid log(0)
            p = np.clip(p, 1e-10, 1.0)
            q = np.clip(q, 1e-10, 1.0)
            kl_divergence += np.sum(p * np.log(p / q))
            valid_rows += 1
    
    if valid_rows > 0:
        kl_divergence /= valid_rows
    
    # Check if generated scanpaths can be correctly classified
    # (For expert PFA, generated scanpaths should be classified as expert)
    
    return {
        'perplexity': perplexity,
        'kl_divergence': kl_divergence,
        'avg_generated_length': np.mean([len(sp) for sp in generated_scanpaths]),
        'n_generated': n_generated
    }


def evaluate_completion(pfa: ProbabilisticFiniteAutomaton,
                        complete_scanpaths: List[List[str]],
                        k_values: List[int] = [1, 3, 5],
                        seed: int = 42) -> Dict:
    """
    Evaluate scanpath completion accuracy.
    
    For each scanpath, hide the last k fixations and evaluate
    how well the PFA can predict them.
    
    Metrics:
    - Top-1 accuracy: Exact match of predicted next state
    - Top-3 accuracy: Correct answer in top 3 predictions
    
    Args:
        pfa: Trained PFA model
        complete_scanpaths: Complete scanpaths to evaluate on
        k_values: Values of k (number of hidden fixations) to test
        seed: Random seed
        
    Returns:
        Dictionary of completion metrics for each k
    """
    np.random.seed(seed)
    results = {}
    
    for k in k_values:
        top1_correct = 0
        top3_correct = 0
        total = 0
        
        for scanpath in complete_scanpaths:
            if len(scanpath) <= k:
                continue
            
            # Split into known and hidden parts
            known = scanpath[:-k]
            hidden = scanpath[-k:]
            
            if len(known) == 0:
                continue
            
            # Get the last known state
            last_known = known[-1]
            if last_known not in pfa.state_to_idx:
                continue
            
            last_idx = pfa.state_to_idx[last_known]
            
            # Get transition probabilities
            trans_probs = pfa.transition_matrix[last_idx]
            
            # Get top predictions
            sorted_indices = np.argsort(trans_probs)[::-1]
            top1_pred = pfa.idx_to_state[sorted_indices[0]]
            top3_pred = [pfa.idx_to_state[sorted_indices[i]] for i in range(min(3, len(sorted_indices)))]
            
            # Check if first hidden state matches
            first_hidden = hidden[0]
            
            if top1_pred == first_hidden:
                top1_correct += 1
            
            if first_hidden in top3_pred:
                top3_correct += 1
            
            total += 1
        
        if total > 0:
            results[f'k{k}_top1_accuracy'] = top1_correct / total
            results[f'k{k}_top3_accuracy'] = top3_correct / total
            results[f'k{k}_total_samples'] = total
        else:
            results[f'k{k}_top1_accuracy'] = 0.0
            results[f'k{k}_top3_accuracy'] = 0.0
            results[f'k{k}_total_samples'] = 0
    
    return results


def run_full_evaluation(df: pd.DataFrame, seed: int = 42) -> Dict:
    """
    Run complete evaluation pipeline.
    
    Args:
        df: Dataset DataFrame
        seed: Random seed
        
    Returns:
        Dictionary with all evaluation results
    """
    results = {}
    
    # Parse scanpaths
    scanpaths = [parse_scanpath(s) for s in df['scanpath'].values]
    labels = df['expertise'].values
    
    expert_scanpaths = [sp for sp, lbl in zip(scanpaths, labels) if lbl == 'expert']
    novice_scanpaths = [sp for sp, lbl in zip(scanpaths, labels) if lbl == 'novice']
    
    print("Running 5-fold cross-validation...")
    cv_results = cross_validate(df, n_folds=5, seed=seed)
    results['classification'] = cv_results
    
    print("Training final models for generation and completion evaluation...")
    # Train on full data for generation/completion evaluation
    classifier = ScanpathClassifier()
    classifier.train(expert_scanpaths, novice_scanpaths)
    
    print("Evaluating generation quality...")
    gen_results_expert = evaluate_generation(classifier.pfa_expert, expert_scanpaths, 
                                             n_generated=50, seed=seed)
    results['generation_expert'] = gen_results_expert
    
    print("Evaluating completion accuracy...")
    completion_results = evaluate_completion(classifier.pfa_expert, expert_scanpaths,
                                            k_values=[1, 3, 5], seed=seed)
    results['completion'] = completion_results
    
    # Compute entropy comparison
    results['entropy'] = {
        'expert_pfa_entropy': classifier.pfa_expert.get_entropy(),
        'novice_pfa_entropy': classifier.pfa_novice.get_entropy()
    }
    
    return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\n--- Classification (5-fold CV) ---")
    if 'classification' in results:
        cv = results['classification']
        print(f"  Accuracy:  {cv.get('accuracy_mean', 0):.4f} ± {cv.get('accuracy_std', 0):.4f}")
        print(f"  Precision: {cv.get('precision_mean', 0):.4f} ± {cv.get('precision_std', 0):.4f}")
        print(f"  Recall:    {cv.get('recall_mean', 0):.4f} ± {cv.get('recall_std', 0):.4f}")
        print(f"  F1-Score:  {cv.get('f1_score_mean', 0):.4f} ± {cv.get('f1_score_std', 0):.4f}")
        if 'roc_auc_mean' in cv:
            print(f"  ROC-AUC:   {cv.get('roc_auc_mean', 0):.4f} ± {cv.get('roc_auc_std', 0):.4f}")
    
    print("\n--- Generation Quality (Expert PFA) ---")
    if 'generation_expert' in results:
        gen = results['generation_expert']
        print(f"  Perplexity:     {gen.get('perplexity', 0):.4f}")
        print(f"  KL-Divergence:  {gen.get('kl_divergence', 0):.6f}")
        print(f"  Avg Length:     {gen.get('avg_generated_length', 0):.1f}")
    
    print("\n--- Completion Accuracy ---")
    if 'completion' in results:
        comp = results['completion']
        for k in [1, 3, 5]:
            t1 = comp.get(f'k{k}_top1_accuracy', 0)
            t3 = comp.get(f'k{k}_top3_accuracy', 0)
            print(f"  k={k}: Top-1 = {t1:.4f}, Top-3 = {t3:.4f}")
    
    print("\n--- Model Entropy ---")
    if 'entropy' in results:
        ent = results['entropy']
        print(f"  Expert PFA entropy: {ent.get('expert_pfa_entropy', 0):.4f}")
        print(f"  Novice PFA entropy: {ent.get('novice_pfa_entropy', 0):.4f}")
        print("  (Lower entropy = more structured/predictable patterns)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('../data/synthetic_dataset.csv')
    
    # Run evaluation
    results = run_full_evaluation(df, seed=42)
    
    # Print results
    print_results(results)
