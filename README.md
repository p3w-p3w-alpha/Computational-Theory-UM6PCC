# ECG Scanpath Pattern Recognition Using Probabilistic Finite Automata

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/p3w-p3w-alpha/Computational-Theory-UM6PCC)

## Project Overview

This project applies Probabilistic Finite Automata (PFA) to analyze and classify eye-tracking scanpaths of medical professionals reading 12-lead electrocardiograms (ECGs). The framework supports:

1. **Classification**: Distinguishing expert from novice reading patterns (99% accuracy)
2. **Generation**: Creating realistic synthetic expert scanpaths (3.54 perplexity)
3. **Completion**: Predicting the continuation of partial scanpaths (84-87% Top-3)

## Authors

- **Nassim Lazrek** (nassim.lazrek@um6p.ma) - Theoretical framework & proofs
- **Omar Ait Said** (omar.aitsaid@um6p.ma) - Implementation & experiments
- **Ilyass Skiriba** (ilyass.skiriba@um6p.ma) - Data generation & evaluation

Mohammed VI Polytechnic University (UM6P), College of Computing, Rabat, Morocco

**Course**: Computational Theory - Fall 2025

## Key Results

| Metric | Value |
|--------|-------|
| Classification Accuracy | 99.0% ± 1.2% |
| Precision | 98.2% ± 2.2% |
| Recall | 100% |
| F1-Score | 99.1% ± 1.1% |
| Expert PFA Entropy | 2.16 bits |
| Novice PFA Entropy | 3.47 bits |
| Entropy Reduction | 38% |
| Generation Perplexity | 3.54 |
| Top-3 Completion Accuracy | 84-87% |

## Project Structure

```
ecg_scanpath_project/
├── README.md                 # This file
├── paper/
│   ├── paper_draft.tex       # LaTeX source
│   └── paper_draft.pdf       # Compiled paper
├── src/
│   ├── pfa.py                # PFA implementation
│   ├── data_generator.py     # Synthetic data generation
│   ├── evaluation.py         # Evaluation metrics
│   └── visualizations.py     # Figure generation
├── data/
│   └── synthetic_dataset.csv # Generated dataset (200 scanpaths)
├── experiments/
│   └── results.json          # Experimental results
├── visualizations/
│   ├── fig1_ecg_layout.png
│   ├── fig3_transition_heatmaps.png
│   ├── fig4_confidence_distribution.png
│   ├── fig5_scanpath_examples.png
│   ├── fig6_completion_accuracy.png
│   └── fig7_entropy_comparison.png
└── docs/
    └── api_documentation.md
```

## Installation

```bash
# Required packages
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Quick Start

### 1. Generate Dataset

```python
from src.data_generator import generate_dataset

# Generate 100 expert + 100 novice scanpaths
df = generate_dataset(n_expert=100, n_novice=100, seed=42)
df.to_csv('data/synthetic_dataset.csv', index=False)
```

### 2. Train and Classify

```python
from src.pfa import ScanpathClassifier

# Prepare data
expert_scanpaths = [sp.split(',') for sp in df[df['expertise']=='expert']['scanpath']]
novice_scanpaths = [sp.split(',') for sp in df[df['expertise']=='novice']['scanpath']]

# Train classifier
classifier = ScanpathClassifier(smoothing=1.0)
classifier.train(expert_scanpaths, novice_scanpaths)

# Classify a scanpath
test_scanpath = ['II', 'III', 'aVF', 'V1', 'V2', 'V3']
prediction, confidence = classifier.classify(test_scanpath)
print(f"Prediction: {prediction}, Confidence: {confidence:.2%}")
```

### 3. Generate Scanpaths

```python
# Generate an expert-like scanpath of length 20
generated = classifier.pfa_expert.generate(length=20, seed=42)
print(f"Generated: {generated}")
```

### 4. Complete Partial Scanpath

```python
# Complete a partial scanpath
partial = ['II', 'III', 'aVF']
completed = classifier.pfa_expert.complete_viterbi(partial, num_steps=5)
print(f"Completed: {completed}")
```

## Key Results

| Metric | Value |
|--------|-------|
| Classification Accuracy | 99.0% ± 1.2% |
| Precision | 98.2% ± 2.2% |
| Recall | 100% ± 0.0% |
| F1-Score | 99.1% ± 1.1% |
| Expert PFA Entropy | 2.16 bits |
| Novice PFA Entropy | 3.47 bits |
| Top-3 Completion Accuracy | 84-87% |

## Theoretical Foundation

### Markov Property Justification

ECG scanpaths satisfy the first-order Markov property:
```
P(X_{t+1} | X_1, ..., X_t) = P(X_{t+1} | X_t)
```

This is justified by:
1. **Cognitive locality**: Attention decisions depend on current fixation
2. **Anatomical constraints**: Lead transitions follow anatomical relationships
3. **Saccade mechanics**: Eye movements planned from current position

### PFA Definition

A PFA is defined as (Q, Σ, T, π₀) where:
- Q = Σ = {I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6}
- T[i,j] = P(next state = j | current state = i)
- π₀[i] = P(initial state = i)

### Sequence Probability

```
P(S) = π₀(s₁) × ∏_{t=1}^{n-1} T[s_t, s_{t+1}]
```

## Run Full Evaluation

```python
from src.evaluation import run_full_evaluation, print_results
import pandas as pd

df = pd.read_csv('data/synthetic_dataset.csv')
results = run_full_evaluation(df, seed=42)
print_results(results)
```

## Generate Figures

```python
from src.visualizations import generate_all_figures
import json

df = pd.read_csv('data/synthetic_dataset.csv')
with open('experiments/results.json', 'r') as f:
    results = json.load(f)

generate_all_figures(df, results, 'visualizations/')
```

## Conference Submission

This work is prepared for submission to:
- CHI 2026 Posters (Barcelona, Spain) - Deadline: January 22, 2026
- HCII 2026 (Montreal, Canada) - Deadline: February 2026
- ETRA 2026 (Marrakech, Morocco)

## License

This project is developed for academic purposes as part of the Computational Theory course at UM6P.

## Acknowledgments

We acknowledge the use of LLMs for code debugging and manuscript preparation. All core intellectual contributions, experimental design, and result interpretation are the original work of the authors.

## Citation

```bibtex
@inproceedings{lazrek2025scanpath,
  title={Scanpath Pattern Recognition Using Probabilistic Finite Automata: 
         An Application to ECG Interpretation},
  author={Lazrek, Nassim and Ait Said, Omar and Skiriba, Ilyass},
  booktitle={Proceedings of CHI 2026},
  year={2026},
  organization={ACM}
}
```
# Computational-Theory-UM6PCC
# Computational-Theory-UM6PCC
