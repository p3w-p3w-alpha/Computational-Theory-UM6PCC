"""
Synthetic ECG Scanpath Dataset Generator

This module generates synthetic scanpath data for expert and novice ECG readers.
Expert patterns are based on clinical guidelines for systematic ECG interpretation.
Novice patterns exhibit more random, less structured viewing behavior.

Authors: Lazrek Nassim, Omar Ait Said, Ilyass Skiriba
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import random

# Define the 12-lead ECG state space
ECG_LEADS = [
    'I', 'II', 'III',           # Limb leads
    'aVR', 'aVL', 'aVF',        # Augmented limb leads
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6'  # Precordial leads
]

# Lead indices for easier reference
LEAD_TO_IDX = {lead: idx for idx, lead in enumerate(ECG_LEADS)}
IDX_TO_LEAD = {idx: lead for idx, lead in enumerate(ECG_LEADS)}

# Clinical lead groupings
INFERIOR_LEADS = ['II', 'III', 'aVF']
LATERAL_LEADS = ['I', 'aVL', 'V5', 'V6']
SEPTAL_LEADS = ['V1', 'V2']
ANTERIOR_LEADS = ['V3', 'V4']
RHYTHM_LEAD = 'II'  # Primary rhythm assessment lead


class ExpertPatternGenerator:
    """
    Generates expert-like scanpath patterns based on clinical ECG reading guidelines.
    
    Expert patterns follow systematic approaches with high probability of:
    - Starting at Lead II (rhythm assessment)
    - Following structured transition patterns
    - Visiting all critical leads
    """
    
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Expert initial state distribution (high probability for Lead II)
        self.initial_dist = self._create_expert_initial_distribution()
        
        # Expert transition matrices for different strategies
        self.systematic_transitions = self._create_systematic_transitions()
        self.rhythm_first_transitions = self._create_rhythm_first_transitions()
        self.chest_focus_transitions = self._create_chest_focus_transitions()
        self.shortcut_transitions = self._create_shortcut_transitions()
    
    def _create_expert_initial_distribution(self) -> np.ndarray:
        """Create initial state distribution favoring Lead II."""
        dist = np.zeros(12)
        dist[LEAD_TO_IDX['II']] = 0.75   # High probability for Lead II
        dist[LEAD_TO_IDX['I']] = 0.10    # Some start with Lead I
        dist[LEAD_TO_IDX['V1']] = 0.10   # Some start with V1
        dist[LEAD_TO_IDX['aVL']] = 0.05  # Rare start with aVL
        return dist
    
    def _create_systematic_transitions(self) -> np.ndarray:
        """
        Create transition matrix for systematic clinical approach.
        Follows: Rate → Rhythm → Axis → P-wave → PR → QRS → ST → T → Lead-by-lead
        """
        T = np.zeros((12, 12))
        
        # From Lead II (rhythm assessment) - go to Lead III or aVF (inferior leads)
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['III']] = 0.50
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['aVF']] = 0.30
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['I']] = 0.15
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['V1']] = 0.05
        
        # From Lead III - continue inferior or move to axis
        T[LEAD_TO_IDX['III'], LEAD_TO_IDX['aVF']] = 0.55
        T[LEAD_TO_IDX['III'], LEAD_TO_IDX['II']] = 0.20
        T[LEAD_TO_IDX['III'], LEAD_TO_IDX['I']] = 0.20
        T[LEAD_TO_IDX['III'], LEAD_TO_IDX['aVL']] = 0.05
        
        # From aVF - move to lateral or precordial
        T[LEAD_TO_IDX['aVF'], LEAD_TO_IDX['I']] = 0.35
        T[LEAD_TO_IDX['aVF'], LEAD_TO_IDX['aVL']] = 0.25
        T[LEAD_TO_IDX['aVF'], LEAD_TO_IDX['V1']] = 0.25
        T[LEAD_TO_IDX['aVF'], LEAD_TO_IDX['II']] = 0.10
        T[LEAD_TO_IDX['aVF'], LEAD_TO_IDX['III']] = 0.05
        
        # From Lead I - axis determination, then precordial
        T[LEAD_TO_IDX['I'], LEAD_TO_IDX['aVL']] = 0.35
        T[LEAD_TO_IDX['I'], LEAD_TO_IDX['aVF']] = 0.25
        T[LEAD_TO_IDX['I'], LEAD_TO_IDX['V1']] = 0.25
        T[LEAD_TO_IDX['I'], LEAD_TO_IDX['II']] = 0.10
        T[LEAD_TO_IDX['I'], LEAD_TO_IDX['V5']] = 0.05
        
        # From aVR - typically brief, move on
        T[LEAD_TO_IDX['aVR'], LEAD_TO_IDX['aVL']] = 0.40
        T[LEAD_TO_IDX['aVR'], LEAD_TO_IDX['V1']] = 0.35
        T[LEAD_TO_IDX['aVR'], LEAD_TO_IDX['I']] = 0.15
        T[LEAD_TO_IDX['aVR'], LEAD_TO_IDX['II']] = 0.10
        
        # From aVL - move to precordial
        T[LEAD_TO_IDX['aVL'], LEAD_TO_IDX['V1']] = 0.40
        T[LEAD_TO_IDX['aVL'], LEAD_TO_IDX['aVF']] = 0.20
        T[LEAD_TO_IDX['aVL'], LEAD_TO_IDX['I']] = 0.15
        T[LEAD_TO_IDX['aVL'], LEAD_TO_IDX['V5']] = 0.15
        T[LEAD_TO_IDX['aVL'], LEAD_TO_IDX['aVR']] = 0.10
        
        # Precordial leads - sequential pattern V1 → V2 → V3 → V4 → V5 → V6
        T[LEAD_TO_IDX['V1'], LEAD_TO_IDX['V2']] = 0.65
        T[LEAD_TO_IDX['V1'], LEAD_TO_IDX['V3']] = 0.15
        T[LEAD_TO_IDX['V1'], LEAD_TO_IDX['II']] = 0.10
        T[LEAD_TO_IDX['V1'], LEAD_TO_IDX['aVR']] = 0.10
        
        T[LEAD_TO_IDX['V2'], LEAD_TO_IDX['V3']] = 0.60
        T[LEAD_TO_IDX['V2'], LEAD_TO_IDX['V1']] = 0.20
        T[LEAD_TO_IDX['V2'], LEAD_TO_IDX['V4']] = 0.15
        T[LEAD_TO_IDX['V2'], LEAD_TO_IDX['II']] = 0.05
        
        T[LEAD_TO_IDX['V3'], LEAD_TO_IDX['V4']] = 0.60
        T[LEAD_TO_IDX['V3'], LEAD_TO_IDX['V2']] = 0.20
        T[LEAD_TO_IDX['V3'], LEAD_TO_IDX['V5']] = 0.15
        T[LEAD_TO_IDX['V3'], LEAD_TO_IDX['V1']] = 0.05
        
        T[LEAD_TO_IDX['V4'], LEAD_TO_IDX['V5']] = 0.55
        T[LEAD_TO_IDX['V4'], LEAD_TO_IDX['V3']] = 0.20
        T[LEAD_TO_IDX['V4'], LEAD_TO_IDX['V6']] = 0.15
        T[LEAD_TO_IDX['V4'], LEAD_TO_IDX['II']] = 0.10
        
        T[LEAD_TO_IDX['V5'], LEAD_TO_IDX['V6']] = 0.55
        T[LEAD_TO_IDX['V5'], LEAD_TO_IDX['V4']] = 0.20
        T[LEAD_TO_IDX['V5'], LEAD_TO_IDX['I']] = 0.15
        T[LEAD_TO_IDX['V5'], LEAD_TO_IDX['aVL']] = 0.10
        
        T[LEAD_TO_IDX['V6'], LEAD_TO_IDX['II']] = 0.35
        T[LEAD_TO_IDX['V6'], LEAD_TO_IDX['V5']] = 0.25
        T[LEAD_TO_IDX['V6'], LEAD_TO_IDX['I']] = 0.20
        T[LEAD_TO_IDX['V6'], LEAD_TO_IDX['V1']] = 0.20
        
        return T
    
    def _create_rhythm_first_transitions(self) -> np.ndarray:
        """
        Create transition matrix for rhythm-first approach.
        Prioritizes inferior leads (II, III, aVF) for rhythm assessment.
        """
        T = np.zeros((12, 12))
        
        # Strong focus on inferior leads first
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['III']] = 0.60
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['aVF']] = 0.30
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['V1']] = 0.10
        
        T[LEAD_TO_IDX['III'], LEAD_TO_IDX['aVF']] = 0.65
        T[LEAD_TO_IDX['III'], LEAD_TO_IDX['II']] = 0.25
        T[LEAD_TO_IDX['III'], LEAD_TO_IDX['I']] = 0.10
        
        T[LEAD_TO_IDX['aVF'], LEAD_TO_IDX['II']] = 0.30
        T[LEAD_TO_IDX['aVF'], LEAD_TO_IDX['III']] = 0.20
        T[LEAD_TO_IDX['aVF'], LEAD_TO_IDX['I']] = 0.25
        T[LEAD_TO_IDX['aVF'], LEAD_TO_IDX['V1']] = 0.25
        
        # Fill remaining transitions
        for i in range(12):
            if np.sum(T[i]) == 0:
                T[i] = self.systematic_transitions[i].copy()
            elif np.sum(T[i]) < 0.99:
                remaining = 1.0 - np.sum(T[i])
                T[i] += self.systematic_transitions[i] * remaining / max(np.sum(self.systematic_transitions[i]), 1e-6)
        
        # Normalize rows
        for i in range(12):
            if np.sum(T[i]) > 0:
                T[i] /= np.sum(T[i])
            else:
                T[i] = np.ones(12) / 12
        
        return T
    
    def _create_chest_focus_transitions(self) -> np.ndarray:
        """
        Create transition matrix for chest-lead focus approach.
        Prioritizes precordial leads (V1-V6) earlier in the examination.
        """
        T = np.zeros((12, 12))
        
        # Quick move to precordial leads
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['V1']] = 0.50
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['III']] = 0.30
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['V2']] = 0.20
        
        # Strong sequential precordial pattern
        T[LEAD_TO_IDX['V1'], LEAD_TO_IDX['V2']] = 0.70
        T[LEAD_TO_IDX['V1'], LEAD_TO_IDX['V3']] = 0.20
        T[LEAD_TO_IDX['V1'], LEAD_TO_IDX['II']] = 0.10
        
        T[LEAD_TO_IDX['V2'], LEAD_TO_IDX['V3']] = 0.70
        T[LEAD_TO_IDX['V2'], LEAD_TO_IDX['V4']] = 0.20
        T[LEAD_TO_IDX['V2'], LEAD_TO_IDX['V1']] = 0.10
        
        T[LEAD_TO_IDX['V3'], LEAD_TO_IDX['V4']] = 0.70
        T[LEAD_TO_IDX['V3'], LEAD_TO_IDX['V5']] = 0.20
        T[LEAD_TO_IDX['V3'], LEAD_TO_IDX['V2']] = 0.10
        
        T[LEAD_TO_IDX['V4'], LEAD_TO_IDX['V5']] = 0.65
        T[LEAD_TO_IDX['V4'], LEAD_TO_IDX['V6']] = 0.25
        T[LEAD_TO_IDX['V4'], LEAD_TO_IDX['V3']] = 0.10
        
        T[LEAD_TO_IDX['V5'], LEAD_TO_IDX['V6']] = 0.65
        T[LEAD_TO_IDX['V5'], LEAD_TO_IDX['I']] = 0.25
        T[LEAD_TO_IDX['V5'], LEAD_TO_IDX['V4']] = 0.10
        
        T[LEAD_TO_IDX['V6'], LEAD_TO_IDX['I']] = 0.40
        T[LEAD_TO_IDX['V6'], LEAD_TO_IDX['II']] = 0.30
        T[LEAD_TO_IDX['V6'], LEAD_TO_IDX['aVL']] = 0.20
        T[LEAD_TO_IDX['V6'], LEAD_TO_IDX['V5']] = 0.10
        
        # Fill remaining with systematic transitions
        for i in range(12):
            if np.sum(T[i]) == 0:
                T[i] = self.systematic_transitions[i].copy()
            elif np.sum(T[i]) < 0.99:
                remaining = 1.0 - np.sum(T[i])
                T[i] += self.systematic_transitions[i] * remaining / max(np.sum(self.systematic_transitions[i]), 1e-6)
        
        # Normalize
        for i in range(12):
            if np.sum(T[i]) > 0:
                T[i] /= np.sum(T[i])
            else:
                T[i] = np.ones(12) / 12
        
        return T
    
    def _create_shortcut_transitions(self) -> np.ndarray:
        """
        Create transition matrix for experienced shortcut approach.
        Efficient pattern focusing on critical leads with fewer fixations.
        """
        T = np.zeros((12, 12))
        
        # Efficient pattern: II → V1 → V2 → V5 → I → aVF
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['V1']] = 0.45
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['I']] = 0.30
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['aVF']] = 0.25
        
        T[LEAD_TO_IDX['V1'], LEAD_TO_IDX['V2']] = 0.50
        T[LEAD_TO_IDX['V1'], LEAD_TO_IDX['V5']] = 0.30
        T[LEAD_TO_IDX['V1'], LEAD_TO_IDX['II']] = 0.20
        
        T[LEAD_TO_IDX['I'], LEAD_TO_IDX['aVF']] = 0.45
        T[LEAD_TO_IDX['I'], LEAD_TO_IDX['V1']] = 0.35
        T[LEAD_TO_IDX['I'], LEAD_TO_IDX['II']] = 0.20
        
        # Fill remaining
        for i in range(12):
            if np.sum(T[i]) == 0:
                T[i] = self.systematic_transitions[i].copy()
            elif np.sum(T[i]) < 0.99:
                remaining = 1.0 - np.sum(T[i])
                T[i] += self.systematic_transitions[i] * remaining / max(np.sum(self.systematic_transitions[i]), 1e-6)
        
        # Normalize
        for i in range(12):
            if np.sum(T[i]) > 0:
                T[i] /= np.sum(T[i])
            else:
                T[i] = np.ones(12) / 12
        
        return T
    
    def generate_scanpath(self, min_length: int = 15, max_length: int = 25) -> List[str]:
        """
        Generate a single expert scanpath.
        
        Args:
            min_length: Minimum number of fixations
            max_length: Maximum number of fixations
            
        Returns:
            List of lead names representing the scanpath
        """
        # Select strategy based on distribution
        strategy_choice = np.random.random()
        if strategy_choice < 0.60:
            transitions = self.systematic_transitions
        elif strategy_choice < 0.75:
            transitions = self.rhythm_first_transitions
        elif strategy_choice < 0.90:
            transitions = self.chest_focus_transitions
        else:
            transitions = self.shortcut_transitions
        
        # Determine scanpath length
        length = np.random.randint(min_length, max_length + 1)
        
        # Generate scanpath
        scanpath = []
        
        # Sample initial state
        current_state = np.random.choice(12, p=self.initial_dist)
        scanpath.append(IDX_TO_LEAD[current_state])
        
        # Generate remaining fixations
        for _ in range(length - 1):
            probs = transitions[current_state]
            next_state = np.random.choice(12, p=probs)
            scanpath.append(IDX_TO_LEAD[next_state])
            current_state = next_state
        
        return scanpath


class NovicePatternGenerator:
    """
    Generates novice-like scanpath patterns.
    
    Novice patterns exhibit:
    - Random starting points
    - Less structured transitions
    - Potential to miss critical leads
    - Higher variability
    """
    
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Novice initial distribution (more uniform)
        self.initial_dist = self._create_novice_initial_distribution()
        
        # Novice transition matrix (more random)
        self.transitions = self._create_novice_transitions()
    
    def _create_novice_initial_distribution(self) -> np.ndarray:
        """Create initial distribution with significant partial learning."""
        # More uniform but with considerable learned tendency
        dist = np.ones(12) / 12
        
        # Many novices learn to start at Lead II (creates significant overlap)
        dist[LEAD_TO_IDX['II']] += 0.25  # Significant partial learning
        dist[LEAD_TO_IDX['V1']] += 0.12  # Some start at V1
        dist[LEAD_TO_IDX['I']] += 0.08   # Some start at Lead I
        
        # Small random perturbation
        dist += np.random.uniform(-0.01, 0.01, 12)
        dist = np.clip(dist, 0.02, None)
        dist /= np.sum(dist)
        return dist
    
    def _create_novice_transitions(self) -> np.ndarray:
        """
        Create transition matrix for novice patterns.
        More uniform with some spatial locality, but with significant partial expert-like patterns
        to create realistic overlap (many novices learn partial patterns from training).
        """
        T = np.zeros((12, 12))
        
        # Base: nearly uniform distribution
        base_prob = 0.055  # Slightly less than uniform
        
        for i in range(12):
            T[i] = np.ones(12) * base_prob
            
            # Add slight spatial bias (novices might follow visual layout)
            if i < 3:  # Limb leads
                T[i, :3] += 0.02
            elif i < 6:  # Augmented leads
                T[i, 3:6] += 0.02
            else:  # Precordial leads
                T[i, 6:] += 0.015
                if i < 11:
                    T[i, i+1] += 0.03
                if i > 6:
                    T[i, i-1] += 0.02
            
            # Add randomness
            T[i] += np.random.uniform(0, 0.012, 12)
            
            # Normalize first
            T[i] /= np.sum(T[i])
        
        # Add significant expert-like patterns for partial learning (creates realistic overlap)
        # Many novices partially learn patterns from training/observation
        
        # Inferior lead sequence (commonly taught)
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['III']] += 0.15
        T[LEAD_TO_IDX['II'], LEAD_TO_IDX['aVF']] += 0.10
        T[LEAD_TO_IDX['III'], LEAD_TO_IDX['aVF']] += 0.12
        T[LEAD_TO_IDX['aVF'], LEAD_TO_IDX['V1']] += 0.10
        
        # Precordial sequence (visually obvious)
        T[LEAD_TO_IDX['V1'], LEAD_TO_IDX['V2']] += 0.18
        T[LEAD_TO_IDX['V2'], LEAD_TO_IDX['V3']] += 0.15
        T[LEAD_TO_IDX['V3'], LEAD_TO_IDX['V4']] += 0.12
        T[LEAD_TO_IDX['V4'], LEAD_TO_IDX['V5']] += 0.10
        T[LEAD_TO_IDX['V5'], LEAD_TO_IDX['V6']] += 0.08
        
        # Some axis-related patterns
        T[LEAD_TO_IDX['I'], LEAD_TO_IDX['aVF']] += 0.08
        T[LEAD_TO_IDX['I'], LEAD_TO_IDX['aVL']] += 0.06
        
        # Normalize again
        for i in range(12):
            T[i] /= np.sum(T[i])
        
        return T
    
    def generate_scanpath(self, min_length: int = 10, max_length: int = 30) -> List[str]:
        """
        Generate a single novice scanpath.
        
        Args:
            min_length: Minimum number of fixations
            max_length: Maximum number of fixations
            
        Returns:
            List of lead names representing the scanpath
        """
        # Higher variability in length for novices
        length = np.random.randint(min_length, max_length + 1)
        
        scanpath = []
        
        # Sample initial state (more random)
        current_state = np.random.choice(12, p=self.initial_dist)
        scanpath.append(IDX_TO_LEAD[current_state])
        
        # Generate remaining fixations
        for _ in range(length - 1):
            probs = self.transitions[current_state]
            next_state = np.random.choice(12, p=probs)
            scanpath.append(IDX_TO_LEAD[next_state])
            current_state = next_state
        
        return scanpath


def generate_dataset(
    n_expert: int = 100,
    n_novice: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate complete synthetic dataset.
    
    Args:
        n_expert: Number of expert scanpaths
        n_novice: Number of novice scanpaths
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: id, expertise, scanpath, length
    """
    expert_gen = ExpertPatternGenerator(seed=seed)
    novice_gen = NovicePatternGenerator(seed=seed + 1000)
    
    data = []
    
    # Generate expert scanpaths
    for i in range(n_expert):
        scanpath = expert_gen.generate_scanpath(min_length=15, max_length=25)
        data.append({
            'id': f'E{i+1:03d}',
            'expertise': 'expert',
            'scanpath': ','.join(scanpath),
            'length': len(scanpath)
        })
    
    # Generate novice scanpaths
    for i in range(n_novice):
        scanpath = novice_gen.generate_scanpath(min_length=10, max_length=30)
        data.append({
            'id': f'N{i+1:03d}',
            'expertise': 'novice',
            'scanpath': ','.join(scanpath),
            'length': len(scanpath)
        })
    
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df


def parse_scanpath(scanpath_str: str) -> List[str]:
    """Convert scanpath string to list."""
    return scanpath_str.split(',')


if __name__ == "__main__":
    # Generate and save dataset
    print("Generating synthetic ECG scanpath dataset...")
    df = generate_dataset(n_expert=100, n_novice=100, seed=42)
    
    # Save to CSV
    df.to_csv('../data/synthetic_dataset.csv', index=False)
    print(f"Dataset saved with {len(df)} scanpaths")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Expert scanpaths: {len(df[df['expertise'] == 'expert'])}")
    print(f"  Novice scanpaths: {len(df[df['expertise'] == 'novice'])}")
    print(f"  Average length (expert): {df[df['expertise'] == 'expert']['length'].mean():.1f}")
    print(f"  Average length (novice): {df[df['expertise'] == 'novice']['length'].mean():.1f}")
    
    # Show sample scanpaths
    print("\nSample Expert Scanpath:")
    expert_sample = df[df['expertise'] == 'expert'].iloc[0]
    print(f"  {expert_sample['scanpath'][:80]}...")
    
    print("\nSample Novice Scanpath:")
    novice_sample = df[df['expertise'] == 'novice'].iloc[0]
    print(f"  {novice_sample['scanpath'][:80]}...")

