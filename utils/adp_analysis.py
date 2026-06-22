"""Analyse the meta-adaptation (--adp) run on the Puffer-2017 domain.

Reads the validation log produced during `--adp` training and renders:
  1. Results/adp_curve.png       -- validation mean QoE vs. epoch
  2. Results/adp_puffer_bar.png  -- Puffer-2017 QoE: baselines vs. adapted MERINA

The Puffer test-set numbers for the non-adapted schemes are read from
comparison_table.csv (produced by compare_schemes.py); the adapted numbers are
passed in below from the explicit checkpoint evaluations documented in README.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, '..')
VALID_LOG = os.path.join(ROOT, 'Results', 'sim', 'merina_adp', 'log_test_2')
OUT = os.path.join(ROOT, 'Results')

# Mean QoE on the Puffer-2017 test set from explicit `--test --tp` evaluations.
ADAPTED = {
    'MERINA\n+adapt (ep170)': 0.821,
    'MERINA\n+adapt (ep300)': 0.561,
}


def read_valid_curve(path):
    epochs, means = [], []
    with open(path) as f:
        for line in f:
            fields = line.split('\t')
            if len(fields) >= 4:
                try:
                    epochs.append(int(fields[0]))
                    means.append(float(fields[3]))  # column 3 = mean QoE
                except ValueError:
                    continue  # header / hyper-parameter line
    return np.array(epochs), np.array(means)


def main():
    epochs, means = read_valid_curve(VALID_LOG)

    # --- Plot 1: adaptation learning curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, means, marker='o', label='Adapted MERINA (valid mean QoE)')
    ax.axhline(0.803, ls='--', color='gray', label='Zero-shot MERINA (0.803)')
    ax.axhline(0.875, ls=':', color='tab:orange', label='BOLA (0.875)')
    ax.axvline(150, color='red', alpha=0.4)
    ax.text(152, means.min(), 'actor updates start (epoch 150)',
            color='red', fontsize=8, va='bottom')
    best = epochs[int(np.argmax(means))]
    ax.scatter([best], [means.max()], color='green', zorder=5, s=80)
    ax.annotate(f'peak: ep{best} = {means.max():.3f}',
                (best, means.max()), textcoords='offset points',
                xytext=(10, 8), color='green')
    ax.set_xlabel('Adaptation epoch')
    ax.set_ylabel('Validation mean QoE (Puffer-2017)')
    ax.set_title('MERINA meta-adaptation on Puffer-2017: brief gain, then over-fitting')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'adp_curve.png'), dpi=150)

    # --- Plot 2: Puffer-2017 bar comparison ---
    table = pd.read_csv(os.path.join(OUT, 'comparison_table.csv'), index_col=0)
    puffer = table.loc['Puffer-2017']
    labels = ['RobustMPC', 'MERINA\n(zero-shot)', 'BOLA'] + list(ADAPTED.keys())
    values = [puffer['RobustMPC'], puffer['MERINA'], puffer['BOLA']] + list(ADAPTED.values())
    colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:red']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt='%.3f')
    ax.axhline(puffer['BOLA'], ls=':', color='tab:orange', alpha=0.6)
    ax.set_ylabel('Mean QoE (log form)')
    ax.set_title('Puffer-2017 (out-of-distribution): does adaptation close the gap?')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'adp_puffer_bar.png'), dpi=150)

    print('Best adaptation epoch:', best, 'valid mean QoE:', round(float(means.max()), 3))
    print('Saved Results/adp_curve.png and Results/adp_puffer_bar.png')


if __name__ == '__main__':
    main()
