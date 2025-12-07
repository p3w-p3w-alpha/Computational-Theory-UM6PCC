# ECG Scanpath Pattern Recognition Using Probabilistic Finite Automata

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)
[![UM6P](https://img.shields.io/badge/UM6P-College%20of%20Computing-orange.svg)](https://um6p.ma)

## Project Overview

This project applies **Probabilistic Finite Automata (PFA)** to analyze and classify eye-tracking scanpaths of medical professionals reading 12-lead electrocardiograms (ECGs). The framework supports:

- **Classification**: Distinguishing expert from novice reading patterns (99% accuracy)
- **Generation**: Creating realistic synthetic expert scanpaths (3.54 perplexity)
- **Completion**: Predicting the continuation of partial scanpaths (84-87% Top-3)

## Authors

| Name | Email | Role |
|------|-------|------|
| Nassim Lazrek | nassim.lazrek@um6p.ma | Theoretical framework & proofs |
| Omar Ait Said | omar.aitsaid@um6p.ma | Implementation & experiments |
| Ilyass Skiriba | ilyass.skiriba@um6p.ma | Data generation & evaluation |

**Institution**: Mohammed VI Polytechnic University (UM6P), College of Computing, Rabat, Morocco  
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

## Repository Structure

```
Computational-Theory-UM6PCC/
│
├── README.md                                    # This file
├── Computational Theory Project Description.pdf # Project requirements
│
├── code/                                        # Python implementation
│   ├── pfa.py                                   # Core PFA implementation
│   ├── data_generator.py                        # Synthetic data generation
│   ├── evaluation.py                            # Evaluation metrics
│   └── visualizations.py                        # Figure generation
│
├── Dataset/
│   └── synthetic_dataset.csv                    # Generated dataset (200 scanpaths)
│
├── Results/
│   └── results.json                             # Experimental results
│
├── Figures/                                     # Generated visualizations
│   ├── fig3_transition_heatmaps.png
│   ├── fig4_confidence_distribution.png
│   ├── fig5_scanpath_examples.png
│   ├── fig6_completion_accuracy.png
│   └── fig7_entropy_comparison.png
│
├── HMM_Final_Paper/
│   └── FINAL_PAPER_HMM.pdf                      # Final research paper
│
├── PFA_State_Digram/
│   └── STATE_DIAGRAM_GITHUB.png                 # PFA state diagram visualization
│
└── Proposal_Document/
    └── PROJECT_PROPOSAL.pdf                     # Initial project proposal
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/p3w-p3w-alpha/Computational-Theory-UM6PCC.git
cd Computational-Theory-UM6PCC

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Quick Start

### 1. Generate Dataset

```python
import sys
sys.path.append('code')
from data_generator import generate_dataset

# Generate 100 expert + 100 novice scanpaths
df = generate_dataset(n_expert=100, n_novice=100, seed=42)
df.to_csv('Dataset/synthetic_dataset.csv', index=False)
```

### 2. Train and Classify

```python
import sys
sys.path.append('code')
import pandas as pd
from pfa import ScanpathClassifier

# Load dataset
df = pd.read_csv('Dataset/synthetic_dataset.csv')

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

### 5. Run Full Evaluation

```python
import sys
sys.path.append('code')
import pandas as pd
from evaluation import run_full_evaluation, print_results

df = pd.read_csv('Dataset/synthetic_dataset.csv')
results = run_full_evaluation(df, seed=42)
print_results(results)
```

### 6. Generate Figures

```python
import sys
sys.path.append('code')
import json
import pandas as pd
from visualizations import generate_all_figures

df = pd.read_csv('Dataset/synthetic_dataset.csv')
with open('Results/results.json', 'r') as f:
    results = json.load(f)

generate_all_figures(df, results, 'Figures/')
```

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
- **Q = Σ** = {I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6}
- **T[i,j]** = P(next state = j | current state = i)
- **π₀[i]** = P(initial state = i)

### Sequence Probability

```
P(S) = π₀(s₁) × ∏_{t=1}^{n-1} T[s_t, s_{t+1}]
```

## File Descriptions

| File | Description |
|------|-------------|
| `code/pfa.py` | Core PFA implementation with classification, generation, and completion |
| `code/data_generator.py` | Generates synthetic expert/novice scanpath data |
| `code/evaluation.py` | Cross-validation, metrics calculation, and result analysis |
| `code/visualizations.py` | Creates all figures (heatmaps, distributions, comparisons) |
| `Dataset/synthetic_dataset.csv` | 200 synthetic scanpaths (100 expert + 100 novice) |
| `Results/results.json` | Saved experimental results and metrics |

## Conference Targets

This work is prepared for submission to:
- **CHI 2026** Posters (Barcelona, Spain) - Deadline: January 22, 2026
- **HCII 2026** (Montreal, Canada) - Deadline: February 2026
- **ETRA 2026** (Marrakech, Morocco)

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

## Contact

For questions or collaboration inquiries, please contact any of the authors via their UM6P email addresses listed above.
