"""
Visualization Module for ECG Scanpath Analysis

Generates all figures for the research paper:
- Figure 1: ECG 12-lead layout
- Figure 2: System architecture
- Figure 3: Expert vs Novice transition heatmaps
- Figure 4: Classification confidence distribution
- Figure 5: Example generated scanpaths
- Figure 6: Completion accuracy

Authors: Lazrek Nassim, Omar Ait Said, Ilyass Skiriba
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
import sys

sys.path.append('.')
from pfa import ProbabilisticFiniteAutomaton, ScanpathClassifier
from data_generator import ECG_LEADS, LEAD_TO_IDX, parse_scanpath

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 14


def create_ecg_layout_figure(save_path: str = None):
    """
    Create Figure 1: ECG 12-lead layout diagram.
    Shows the standard 4x3 arrangement of ECG leads.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Define ECG lead positions in 4x3 grid
    lead_positions = {
        # Row 1
        'I': (0, 3), 'aVR': (1, 3), 'V1': (2, 3), 'V4': (3, 3),
        # Row 2  
        'II': (0, 2), 'aVL': (1, 2), 'V2': (2, 2), 'V5': (3, 2),
        # Row 3
        'III': (0, 1), 'aVF': (1, 1), 'V3': (2, 1), 'V6': (3, 1),
    }
    
    # Colors for different lead groups
    colors = {
        'limb': '#3498db',      # Blue for limb leads
        'augmented': '#9b59b6', # Purple for augmented
        'precordial': '#e74c3c' # Red for precordial
    }
    
    lead_groups = {
        'I': 'limb', 'II': 'limb', 'III': 'limb',
        'aVR': 'augmented', 'aVL': 'augmented', 'aVF': 'augmented',
        'V1': 'precordial', 'V2': 'precordial', 'V3': 'precordial',
        'V4': 'precordial', 'V5': 'precordial', 'V6': 'precordial'
    }
    
    # Draw lead boxes
    for lead, (x, y) in lead_positions.items():
        color = colors[lead_groups[lead]]
        rect = mpatches.FancyBboxPatch(
            (x * 2.5, y * 2), 2, 1.5,
            boxstyle="round,pad=0.05",
            facecolor=color, alpha=0.3,
            edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x * 2.5 + 1, y * 2 + 0.75, lead,
                ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['limb'], alpha=0.3, 
                      edgecolor=colors['limb'], label='Limb Leads (I, II, III)'),
        mpatches.Patch(facecolor=colors['augmented'], alpha=0.3,
                      edgecolor=colors['augmented'], label='Augmented Leads (aVR, aVL, aVF)'),
        mpatches.Patch(facecolor=colors['precordial'], alpha=0.3,
                      edgecolor=colors['precordial'], label='Precordial Leads (V1-V6)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10)
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Standard 12-Lead ECG Layout\n(State Space for PFA Model)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.close()


def create_transition_heatmaps(classifier: ScanpathClassifier, save_path: str = None):
    """
    Create Figure 3: Expert vs Novice transition matrix heatmaps.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Expert transition matrix
    T_expert = classifier.pfa_expert.get_transition_matrix()
    
    # Novice transition matrix
    T_novice = classifier.pfa_novice.get_transition_matrix()
    
    # Create custom colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Plot expert heatmap
    sns.heatmap(T_expert, ax=axes[0], cmap=cmap,
                xticklabels=ECG_LEADS, yticklabels=ECG_LEADS,
                annot=False, fmt='.2f', cbar_kws={'label': 'P(to | from)'},
                vmin=0, vmax=0.7)
    axes[0].set_title('Expert Transition Probabilities', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('To State', fontsize=10)
    axes[0].set_ylabel('From State', fontsize=10)
    
    # Plot novice heatmap
    sns.heatmap(T_novice, ax=axes[1], cmap=cmap,
                xticklabels=ECG_LEADS, yticklabels=ECG_LEADS,
                annot=False, fmt='.2f', cbar_kws={'label': 'P(to | from)'},
                vmin=0, vmax=0.7)
    axes[1].set_title('Novice Transition Probabilities', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('To State', fontsize=10)
    axes[1].set_ylabel('From State', fontsize=10)
    
    plt.suptitle('Transition Matrix Comparison: Expert vs Novice', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.close()


def create_confidence_distribution(classifier: ScanpathClassifier, 
                                   scanpaths: list, labels: list,
                                   save_path: str = None):
    """
    Create Figure 4: Classification confidence distribution.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    expert_confidences = []
    novice_confidences = []
    
    for sp, label in zip(scanpaths, labels):
        _, conf = classifier.classify(sp)
        if label == 'expert':
            expert_confidences.append(conf)
        else:
            novice_confidences.append(conf)
    
    # Plot histograms
    bins = np.linspace(0, 1, 21)
    ax.hist(expert_confidences, bins=bins, alpha=0.6, label='Expert Scanpaths',
            color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.hist(novice_confidences, bins=bins, alpha=0.6, label='Novice Scanpaths',
            color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    
    ax.set_xlabel('Classification Confidence (for predicted class)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Classification Confidence Scores', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.close()


def create_scanpath_visualization(scanpaths: list, labels: list,
                                  n_samples: int = 3, save_path: str = None):
    """
    Create Figure 5: Example scanpath visualizations.
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(14, 8))
    
    # Get samples
    expert_samples = [(sp, lbl) for sp, lbl in zip(scanpaths, labels) if lbl == 'expert'][:n_samples]
    novice_samples = [(sp, lbl) for sp, lbl in zip(scanpaths, labels) if lbl == 'novice'][:n_samples]
    
    # Lead positions for visualization
    lead_positions = {
        'I': (0, 2), 'II': (0, 1), 'III': (0, 0),
        'aVR': (1, 2), 'aVL': (1, 1), 'aVF': (1, 0),
        'V1': (2, 2), 'V2': (2, 1), 'V3': (2, 0),
        'V4': (3, 2), 'V5': (3, 1), 'V6': (3, 0)
    }
    
    def plot_scanpath(ax, scanpath, title, color):
        # Draw all lead positions
        for lead, (x, y) in lead_positions.items():
            ax.scatter(x, y, s=300, c='lightgray', edgecolors='black', zorder=1)
            ax.text(x, y, lead, ha='center', va='center', fontsize=8, zorder=3)
        
        # Draw scanpath
        for i in range(len(scanpath) - 1):
            if scanpath[i] in lead_positions and scanpath[i+1] in lead_positions:
                x1, y1 = lead_positions[scanpath[i]]
                x2, y2 = lead_positions[scanpath[i+1]]
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=color, 
                                          lw=1.5, alpha=0.6))
        
        # Highlight start and end
        if scanpath[0] in lead_positions:
            x, y = lead_positions[scanpath[0]]
            ax.scatter(x, y, s=400, c='green', edgecolors='black', 
                      zorder=2, marker='s', label='Start')
        
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{title}\n(Length: {len(scanpath)})', fontsize=10)
    
    # Plot expert samples
    for i, (sp, _) in enumerate(expert_samples):
        plot_scanpath(axes[0, i], sp, f'Expert Sample {i+1}', '#2ecc71')
    
    # Plot novice samples
    for i, (sp, _) in enumerate(novice_samples):
        plot_scanpath(axes[1, i], sp, f'Novice Sample {i+1}', '#e74c3c')
    
    # Row labels
    axes[0, 0].text(-1.5, 1, 'EXPERT', rotation=90, va='center', ha='center',
                    fontsize=12, fontweight='bold', color='#2ecc71')
    axes[1, 0].text(-1.5, 1, 'NOVICE', rotation=90, va='center', ha='center',
                    fontsize=12, fontweight='bold', color='#e74c3c')
    
    plt.suptitle('Sample Scanpath Visualizations', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.close()


def create_completion_accuracy_plot(results: dict, save_path: str = None):
    """
    Create Figure 6: Completion accuracy vs steps.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    k_values = [1, 3, 5]
    top1_acc = [results['completion'][f'k{k}_top1_accuracy'] for k in k_values]
    top3_acc = [results['completion'][f'k{k}_top3_accuracy'] for k in k_values]
    
    x = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, top1_acc, width, label='Top-1 Accuracy',
                   color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, top3_acc, width, label='Top-3 Accuracy',
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Number of Hidden Fixations (k)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Scanpath Completion Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k = {k}' for k in k_values])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.close()


def create_entropy_comparison(classifier: ScanpathClassifier, save_path: str = None):
    """
    Create a bar plot comparing expert vs novice PFA entropy.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    expert_entropy = classifier.pfa_expert.get_entropy()
    novice_entropy = classifier.pfa_novice.get_entropy()
    
    categories = ['Expert PFA', 'Novice PFA']
    values = [expert_entropy, novice_entropy]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Average Entropy (bits)', fontsize=11)
    ax.set_title('Transition Matrix Entropy Comparison\n(Lower = More Structured)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 4)
    
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.close()


def generate_all_figures(df: pd.DataFrame, results: dict, output_dir: str):
    """Generate all figures for the paper."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse data
    scanpaths = [parse_scanpath(s) for s in df['scanpath'].values]
    labels = list(df['expertise'].values)
    
    expert_scanpaths = [sp for sp, lbl in zip(scanpaths, labels) if lbl == 'expert']
    novice_scanpaths = [sp for sp, lbl in zip(scanpaths, labels) if lbl == 'novice']
    
    # Train classifier
    print("Training classifier for visualizations...")
    classifier = ScanpathClassifier()
    classifier.train(expert_scanpaths, novice_scanpaths)
    
    # Generate figures
    print("Generating figures...")
    
    create_ecg_layout_figure(
        save_path=os.path.join(output_dir, 'fig1_ecg_layout.png'))
    
    create_transition_heatmaps(
        classifier, 
        save_path=os.path.join(output_dir, 'fig3_transition_heatmaps.png'))
    
    create_confidence_distribution(
        classifier, scanpaths, labels,
        save_path=os.path.join(output_dir, 'fig4_confidence_distribution.png'))
    
    create_scanpath_visualization(
        scanpaths, labels, n_samples=3,
        save_path=os.path.join(output_dir, 'fig5_scanpath_examples.png'))
    
    create_completion_accuracy_plot(
        results,
        save_path=os.path.join(output_dir, 'fig6_completion_accuracy.png'))
    
    create_entropy_comparison(
        classifier,
        save_path=os.path.join(output_dir, 'fig7_entropy_comparison.png'))
    
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    import json
    
    # Load data and results
    df = pd.read_csv('../data/synthetic_dataset.csv')
    
    with open('../experiments/results.json', 'r') as f:
        results = json.load(f)
    
    # Generate all figures
    generate_all_figures(df, results, '../visualizations/')
