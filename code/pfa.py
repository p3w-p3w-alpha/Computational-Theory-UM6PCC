"""
Probabilistic Finite Automata (PFA) Implementation for ECG Scanpath Analysis

This module implements the PFA model for:
1. Training from scanpath data
2. Computing sequence probabilities
3. Classification (Expert vs Novice)
4. Generation of new scanpaths
5. Completion of partial scanpaths

Authors: Lazrek Nassim, Omar Ait Said, Ilyass Skiriba
Date: December 2025
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


# Define the state space (12-lead ECG)
STATES = [
    'I', 'II', 'III',
    'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
]

STATE_TO_IDX = {state: idx for idx, state in enumerate(STATES)}
IDX_TO_STATE = {idx: state for idx, state in enumerate(STATES)}


class ProbabilisticFiniteAutomaton:
    """
    Probabilistic Finite Automaton for scanpath modeling.
    
    A PFA is defined as a tuple (Q, Σ, T, π₀) where:
    - Q: Finite set of states (ECG leads)
    - Σ: Alphabet (same as Q for our application)
    - T: Transition probability matrix T[i,j] = P(j|i)
    - π₀: Initial state distribution
    
    Attributes:
        states: List of state names
        n_states: Number of states
        transition_matrix: Transition probability matrix
        initial_dist: Initial state distribution
        smoothing: Laplace smoothing parameter
    """
    
    def __init__(self, states: List[str] = None, smoothing: float = 1.0):
        """
        Initialize the PFA.
        
        Args:
            states: List of state names (default: 12-lead ECG states)
            smoothing: Laplace smoothing parameter (default: 1.0)
        """
        self.states = states if states is not None else STATES.copy()
        self.n_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        
        self.smoothing = smoothing
        
        # Initialize with uniform distributions (will be updated during training)
        self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        self.initial_dist = np.ones(self.n_states) / self.n_states
        
        # Store counts for analysis
        self.transition_counts = None
        self.initial_counts = None
        self.is_trained = False
    
    def train(self, scanpaths: List[List[str]]) -> 'ProbabilisticFiniteAutomaton':
        """
        Train the PFA using Maximum Likelihood Estimation.
        
        The MLE for transition probabilities is:
            T[i,j] = (C[i,j] + α) / (Σ_k C[i,k] + α|Q|)
        
        where:
            - C[i,j] = count of transitions from state i to state j
            - α = smoothing parameter (Laplace smoothing)
            - |Q| = number of states
        
        Args:
            scanpaths: List of scanpaths, each scanpath is a list of state names
            
        Returns:
            self (for method chaining)
        """
        # Initialize count matrices
        self.transition_counts = np.zeros((self.n_states, self.n_states))
        self.initial_counts = np.zeros(self.n_states)
        
        # Count transitions and initial states
        for scanpath in scanpaths:
            if len(scanpath) == 0:
                continue
            
            # Count initial state
            first_state = scanpath[0]
            if first_state in self.state_to_idx:
                self.initial_counts[self.state_to_idx[first_state]] += 1
            
            # Count transitions
            for i in range(len(scanpath) - 1):
                current = scanpath[i]
                next_state = scanpath[i + 1]
                
                if current in self.state_to_idx and next_state in self.state_to_idx:
                    curr_idx = self.state_to_idx[current]
                    next_idx = self.state_to_idx[next_state]
                    self.transition_counts[curr_idx, next_idx] += 1
        
        # Apply Laplace smoothing and normalize
        # Initial distribution
        smoothed_initial = self.initial_counts + self.smoothing
        self.initial_dist = smoothed_initial / np.sum(smoothed_initial)
        
        # Transition matrix
        smoothed_transitions = self.transition_counts + self.smoothing
        row_sums = np.sum(smoothed_transitions, axis=1, keepdims=True)
        self.transition_matrix = smoothed_transitions / row_sums
        
        self.is_trained = True
        return self
    
    def compute_log_probability(self, scanpath: List[str]) -> float:
        """
        Compute the log-probability of a scanpath.
        
        Using the Markov property:
            P(S) = π₀(s₁) × Π_{t=1}^{n-1} T[s_t, s_{t+1}]
        
        In log-space (for numerical stability):
            log P(S) = log π₀(s₁) + Σ_{t=1}^{n-1} log T[s_t, s_{t+1}]
        
        Args:
            scanpath: List of state names
            
        Returns:
            Log-probability of the scanpath
        """
        if len(scanpath) == 0:
            return float('-inf')
        
        # Initial state probability
        first_state = scanpath[0]
        if first_state not in self.state_to_idx:
            return float('-inf')
        
        log_prob = np.log(self.initial_dist[self.state_to_idx[first_state]])
        
        # Transition probabilities
        for i in range(len(scanpath) - 1):
            current = scanpath[i]
            next_state = scanpath[i + 1]
            
            if current not in self.state_to_idx or next_state not in self.state_to_idx:
                return float('-inf')
            
            curr_idx = self.state_to_idx[current]
            next_idx = self.state_to_idx[next_state]
            
            trans_prob = self.transition_matrix[curr_idx, next_idx]
            if trans_prob <= 0:
                return float('-inf')
            
            log_prob += np.log(trans_prob)
        
        return log_prob
    
    def compute_probability(self, scanpath: List[str]) -> float:
        """
        Compute the probability of a scanpath.
        
        Args:
            scanpath: List of state names
            
        Returns:
            Probability of the scanpath
        """
        log_prob = self.compute_log_probability(scanpath)
        if log_prob == float('-inf'):
            return 0.0
        return np.exp(log_prob)
    
    def generate(self, length: int, seed: int = None) -> List[str]:
        """
        Generate a new scanpath by sampling from the PFA.
        
        Uses ancestral sampling:
            1. Sample s₁ ~ π₀
            2. For t = 2, ..., n: Sample s_t ~ T[s_{t-1}, ·]
        
        Args:
            length: Desired length of the generated scanpath
            seed: Random seed for reproducibility
            
        Returns:
            Generated scanpath as list of state names
        """
        if seed is not None:
            np.random.seed(seed)
        
        scanpath = []
        
        # Sample initial state
        current_idx = np.random.choice(self.n_states, p=self.initial_dist)
        scanpath.append(self.idx_to_state[current_idx])
        
        # Sample subsequent states
        for _ in range(length - 1):
            transition_probs = self.transition_matrix[current_idx]
            next_idx = np.random.choice(self.n_states, p=transition_probs)
            scanpath.append(self.idx_to_state[next_idx])
            current_idx = next_idx
        
        return scanpath
    
    def complete_greedy(self, partial_scanpath: List[str], num_steps: int) -> List[str]:
        """
        Complete a partial scanpath using greedy selection.
        
        At each step, selects the most likely next state:
            s_{t+1} = argmax_s T[s_t, s]
        
        Args:
            partial_scanpath: Known prefix of the scanpath
            num_steps: Number of steps to complete
            
        Returns:
            Completed scanpath
        """
        if len(partial_scanpath) == 0:
            return []
        
        completed = partial_scanpath.copy()
        current_state = partial_scanpath[-1]
        
        if current_state not in self.state_to_idx:
            return completed
        
        current_idx = self.state_to_idx[current_state]
        
        for _ in range(num_steps):
            # Select most likely next state
            next_idx = np.argmax(self.transition_matrix[current_idx])
            completed.append(self.idx_to_state[next_idx])
            current_idx = next_idx
        
        return completed
    
    def complete_sampling(self, partial_scanpath: List[str], num_steps: int, 
                         seed: int = None) -> List[str]:
        """
        Complete a partial scanpath using probabilistic sampling.
        
        Samples from the transition distribution:
            s_{t+1} ~ T[s_t, ·]
        
        Args:
            partial_scanpath: Known prefix of the scanpath
            num_steps: Number of steps to complete
            seed: Random seed
            
        Returns:
            Completed scanpath
        """
        if seed is not None:
            np.random.seed(seed)
        
        if len(partial_scanpath) == 0:
            return []
        
        completed = partial_scanpath.copy()
        current_state = partial_scanpath[-1]
        
        if current_state not in self.state_to_idx:
            return completed
        
        current_idx = self.state_to_idx[current_state]
        
        for _ in range(num_steps):
            transition_probs = self.transition_matrix[current_idx]
            next_idx = np.random.choice(self.n_states, p=transition_probs)
            completed.append(self.idx_to_state[next_idx])
            current_idx = next_idx
        
        return completed
    
    def complete_viterbi(self, partial_scanpath: List[str], num_steps: int) -> List[str]:
        """
        Complete a partial scanpath using Viterbi-style dynamic programming.
        
        Finds the most likely completion:
            S* = argmax_{s_{k+1},...,s_{k+m}} P(s_{k+1},...,s_{k+m} | s_k)
        
        Uses dynamic programming:
            V[t][j] = max probability of reaching state j at step t
            V[t][j] = max_i { V[t-1][i] × T[i][j] }
        
        Args:
            partial_scanpath: Known prefix of the scanpath
            num_steps: Number of steps to complete
            
        Returns:
            Completed scanpath with maximum probability
        """
        if len(partial_scanpath) == 0 or num_steps == 0:
            return partial_scanpath.copy()
        
        last_state = partial_scanpath[-1]
        if last_state not in self.state_to_idx:
            return partial_scanpath.copy()
        
        last_idx = self.state_to_idx[last_state]
        
        # Initialize DP tables
        # V[t][j] = max log-probability of reaching state j at step t
        V = np.full((num_steps + 1, self.n_states), float('-inf'))
        backpointer = np.zeros((num_steps + 1, self.n_states), dtype=int)
        
        # Base case: start from last known state
        V[0, last_idx] = 0.0  # log(1) = 0
        
        # Forward pass
        for t in range(1, num_steps + 1):
            for j in range(self.n_states):
                best_prob = float('-inf')
                best_pred = 0
                
                for i in range(self.n_states):
                    if V[t-1, i] > float('-inf'):
                        prob = V[t-1, i] + np.log(self.transition_matrix[i, j])
                        if prob > best_prob:
                            best_prob = prob
                            best_pred = i
                
                V[t, j] = best_prob
                backpointer[t, j] = best_pred
        
        # Find best final state
        best_final = np.argmax(V[num_steps])
        
        # Backtrack to find optimal path
        completion = []
        current = best_final
        
        for t in range(num_steps, 0, -1):
            completion.append(self.idx_to_state[current])
            current = backpointer[t, current]
        
        completion.reverse()
        
        return partial_scanpath.copy() + completion
    
    def get_transition_matrix(self) -> np.ndarray:
        """Return the transition matrix."""
        return self.transition_matrix.copy()
    
    def get_initial_distribution(self) -> np.ndarray:
        """Return the initial state distribution."""
        return self.initial_dist.copy()
    
    def get_entropy(self) -> float:
        """
        Compute the entropy of the transition matrix.
        
        Higher entropy indicates more random/uniform transitions.
        Lower entropy indicates more structured/predictable patterns.
        
        Returns:
            Average entropy across all states
        """
        entropies = []
        for i in range(self.n_states):
            probs = self.transition_matrix[i]
            # Avoid log(0) by filtering out zero probabilities
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            entropies.append(entropy)
        return np.mean(entropies)


class ScanpathClassifier:
    """
    Binary classifier for Expert vs Novice scanpaths using two PFAs.
    
    Classification is based on log-likelihood ratio:
        Λ(S) = log P(S|Expert) - log P(S|Novice)
        
    Decision rule:
        Class = Expert if Λ(S) > 0, else Novice
        
    Confidence is computed using sigmoid transformation:
        confidence = σ(Λ(S)) = 1 / (1 + exp(-Λ(S)))
    """
    
    def __init__(self, smoothing: float = 1.0):
        """
        Initialize the classifier.
        
        Args:
            smoothing: Laplace smoothing parameter for PFAs
        """
        self.pfa_expert = ProbabilisticFiniteAutomaton(smoothing=smoothing)
        self.pfa_novice = ProbabilisticFiniteAutomaton(smoothing=smoothing)
        self.is_trained = False
    
    def train(self, expert_scanpaths: List[List[str]], 
              novice_scanpaths: List[List[str]]) -> 'ScanpathClassifier':
        """
        Train both PFAs on their respective datasets.
        
        Args:
            expert_scanpaths: List of expert scanpaths
            novice_scanpaths: List of novice scanpaths
            
        Returns:
            self (for method chaining)
        """
        self.pfa_expert.train(expert_scanpaths)
        self.pfa_novice.train(novice_scanpaths)
        self.is_trained = True
        return self
    
    def classify(self, scanpath: List[str]) -> Tuple[str, float]:
        """
        Classify a scanpath as Expert or Novice.
        
        Args:
            scanpath: Scanpath to classify
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        log_prob_expert = self.pfa_expert.compute_log_probability(scanpath)
        log_prob_novice = self.pfa_novice.compute_log_probability(scanpath)
        
        # Log-likelihood ratio
        log_ratio = log_prob_expert - log_prob_novice
        
        # Decision
        if log_ratio > 0:
            predicted_class = 'expert'
        else:
            predicted_class = 'novice'
        
        # Confidence using sigmoid
        # Clip to avoid overflow
        log_ratio_clipped = np.clip(log_ratio, -500, 500)
        confidence = 1.0 / (1.0 + np.exp(-log_ratio_clipped))
        
        # Adjust confidence to reflect predicted class
        if predicted_class == 'novice':
            confidence = 1.0 - confidence
        
        return predicted_class, confidence
    
    def predict(self, scanpaths: List[List[str]]) -> List[str]:
        """
        Predict classes for multiple scanpaths.
        
        Args:
            scanpaths: List of scanpaths
            
        Returns:
            List of predicted classes
        """
        predictions = []
        for scanpath in scanpaths:
            pred_class, _ = self.classify(scanpath)
            predictions.append(pred_class)
        return predictions
    
    def predict_proba(self, scanpaths: List[List[str]]) -> np.ndarray:
        """
        Predict class probabilities for multiple scanpaths.
        
        Args:
            scanpaths: List of scanpaths
            
        Returns:
            Array of shape (n_samples, 2) with [P(expert), P(novice)]
        """
        probas = []
        for scanpath in scanpaths:
            log_prob_expert = self.pfa_expert.compute_log_probability(scanpath)
            log_prob_novice = self.pfa_novice.compute_log_probability(scanpath)
            
            log_ratio = log_prob_expert - log_prob_novice
            log_ratio_clipped = np.clip(log_ratio, -500, 500)
            
            p_expert = 1.0 / (1.0 + np.exp(-log_ratio_clipped))
            p_novice = 1.0 - p_expert
            
            probas.append([p_expert, p_novice])
        
        return np.array(probas)
    
    def get_log_likelihood_ratio(self, scanpath: List[str]) -> float:
        """
        Compute the log-likelihood ratio for a scanpath.
        
        Args:
            scanpath: Scanpath to evaluate
            
        Returns:
            Log-likelihood ratio Λ(S) = log P(S|Expert) - log P(S|Novice)
        """
        log_prob_expert = self.pfa_expert.compute_log_probability(scanpath)
        log_prob_novice = self.pfa_novice.compute_log_probability(scanpath)
        return log_prob_expert - log_prob_novice


def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL-divergence between two probability distributions.
    
    KL(P || Q) = Σ P(x) log(P(x) / Q(x))
    
    Args:
        p: First distribution (reference)
        q: Second distribution
        
    Returns:
        KL-divergence value
    """
    # Avoid division by zero and log(0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    
    return np.sum(p * np.log(p / q))


def compute_matrix_kl_divergence(T1: np.ndarray, T2: np.ndarray) -> float:
    """
    Compute average KL-divergence between two transition matrices.
    
    Args:
        T1: First transition matrix
        T2: Second transition matrix
        
    Returns:
        Average KL-divergence across all rows
    """
    n = T1.shape[0]
    kl_sum = 0.0
    
    for i in range(n):
        kl_sum += compute_kl_divergence(T1[i], T2[i])
    
    return kl_sum / n


if __name__ == "__main__":
    # Simple test
    print("Testing PFA Implementation...")
    
    # Create sample data
    expert_data = [
        ['II', 'III', 'aVF', 'V1', 'V2', 'V3'],
        ['II', 'III', 'aVF', 'I', 'V1', 'V2'],
        ['II', 'aVF', 'III', 'V1', 'V2', 'V3'],
    ]
    
    novice_data = [
        ['V3', 'I', 'V5', 'II', 'aVR', 'V1'],
        ['aVL', 'V2', 'I', 'V6', 'III', 'aVF'],
        ['V1', 'aVF', 'V4', 'I', 'II', 'V3'],
    ]
    
    # Train classifier
    classifier = ScanpathClassifier()
    classifier.train(expert_data, novice_data)
    
    # Test classification
    test_scanpath = ['II', 'III', 'aVF', 'V1', 'V2']
    pred_class, confidence = classifier.classify(test_scanpath)
    print(f"Test scanpath: {test_scanpath}")
    print(f"Predicted class: {pred_class} (confidence: {confidence:.2%})")
    
    # Test generation
    generated = classifier.pfa_expert.generate(length=8, seed=42)
    print(f"\nGenerated expert scanpath: {generated}")
    
    # Test completion
    partial = ['II', 'III']
    completed = classifier.pfa_expert.complete_viterbi(partial, num_steps=4)
    print(f"\nPartial scanpath: {partial}")
    print(f"Completed (Viterbi): {completed}")
    
    print("\nPFA Implementation test complete!")
