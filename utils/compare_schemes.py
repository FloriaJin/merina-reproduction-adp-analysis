"""Aggregate per-trace reward logs into a scheme x dataset mean-QoE comparison.

Usage: python compare_schemes.py
Produces Results/comparison_table.csv and Results/comparison_chart.png
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'Results', 'test', 'log')

DATASETS = {
    'fcc': 'FCC',
    '3gp': 'HSDPA',
    'oboe': 'Oboe',
    'puffer': 'Puffer-2017',
    'puffer2': 'Puffer-2018',
}

SCHEMES = {
    'merina': 'MERINA',
    'bola': 'BOLA',
    'mpc': 'RobustMPC',
}


def mean_qoe_for_logs(log_paths):
    """Reward is the 7th whitespace-separated field of every non-blank line.

    Matches algos/test_v5.py: per-trace mean reward skipping the first 4
    chunks (startup), then averaged across traces (mean-of-means).
    """
    per_trace_means = []
    for path in log_paths:
        rewards = []
        with open(path) as f:
            for line in f:
                fields = line.split()
                if len(fields) >= 7:
                    rewards.append(float(fields[6]))
        if len(rewards) > 4:
            per_trace_means.append(np.mean(rewards[4:]))
    return float(np.mean(per_trace_means)) if per_trace_means else np.nan


def main():
    rows = []
    for dataset_key, dataset_label in DATASETS.items():
        dataset_dir = os.path.join(RESULTS_ROOT, dataset_key)
        row = {'dataset': dataset_label}
        for scheme_key, scheme_label in SCHEMES.items():
            log_paths = glob.glob(os.path.join(dataset_dir, f'log_test_{scheme_key}_*'))
            row[scheme_label] = mean_qoe_for_logs(log_paths)
        rows.append(row)

    table = pd.DataFrame(rows).set_index('dataset')
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'Results')
    table.to_csv(os.path.join(out_dir, 'comparison_table.csv'))
    print(table.round(3))

    ax = table.plot(kind='bar', figsize=(8, 5))
    ax.set_ylabel('Mean QoE (log form)')
    ax.set_title('MERINA vs. baselines across bandwidth-trace datasets')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison_chart.png'), dpi=150)
    print('Saved table to Results/comparison_table.csv and chart to Results/comparison_chart.png')


if __name__ == '__main__':
    main()
