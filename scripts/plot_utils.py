#!/usr/bin/env python3
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(results_file):
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    with open(results_file) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("results.json must contain a list")
    return data


def valid_entries(data):
    return [entry for entry in data if entry.get('row') or entry.get('tile')]


def save_no_data_figure(out_path, title, message):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis('off')
    ax.text(0.5, 0.62, title, ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax.text(0.5, 0.40, message, ha='center', va='center',
            fontsize=11, color='dimgray')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

