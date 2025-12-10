import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_roc(data, path, show=False, inset=True, logy=False):
    """
    Main plotting logic for ROC curves.
    
    Args:
        data (dict): Dictionary containing ROC data (fpr, tpr per fold) and aggregate stats.
        path (Path or str): Output path for the plot.
        show (bool): Whether to show the plot interactively.
        inset (bool): Whether to include a zoomed inset.
        logy (bool): Log scale Y-axis (rarely used for ROC, but supported).
    """
    
    # Extract aggregate statistics
    mean_fpr   = data['mean_fpr']
    mean_tpr   = data['mean_tpr']
    tprs_lower = data['tprs_lower']
    tprs_upper = data['tprs_upper']
    mean_auc   = data['mean_auc']
    std_auc    = data['std_auc']
    
    # Extract fold data
    fold_data  = {k:v for k,v in data.items() if k.startswith('Fold ')}

    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    
    # Setup axes list (Main + Optional Inset)
    axes = [ax]
    if inset:
        # Matches the location in your original utils class
        inset_ax = ax.inset_axes([0.25, 0.4, 0.65, 0.3], xlim=[.001, .8], ylim=[.9, 1])
        axes.append(inset_ax)

    # Plot on all active axes (Main + Inset)
    for axis in axes:
        # 1. Plot Chance Level
        axis.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)

        # 2. Plot Individual Folds (Faint)
        for i, (name, (fpr, tpr)) in enumerate(fold_data.items()):
            fold_label = 'ROC (Fold)' if i == 0 else None
            axis.plot(fpr, tpr, color='red', alpha=0.3, lw=1, label=fold_label)

        # 3. Plot Mean ROC
        axis.plot(
            mean_fpr,
            mean_tpr,
            color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        # 4. Plot Variance (Std Dev)
        axis.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color='blue',
            lw=0,
            alpha=0.2,
            label=r'$\pm$ 1$\sigma$',
        )

    # Formatting Main Axis
    ax.set_xlabel('False Positive Rate', loc='right')
    ax.set_ylabel('True Positive Rate', loc='top')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    # Format legend
    handles, labels = ax.get_legend_handles_labels()
    target_indices = [3, 2, 0, 1]
    ordered_handles = [handle for (idx, handle) in sorted(zip(target_indices, handles))]
    ordered_labels = [label for (idx, label) in sorted(zip(target_indices, labels))]
    ax.legend(ordered_handles, ordered_labels, loc='lower right', fontsize='medium')

    # Format inset
    if inset:
        # Connect inset to main plot
        ax.indicate_inset_zoom(inset_ax, edgecolor='black')
        
        # Clean up inset labels
        inset_ax.set_xlabel('')
        inset_ax.set_ylabel('')
        inset_ax.tick_params(axis='both', which='both', labelsize='small')
        # inset_ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    if logy:
        ax.set_yscale('log')
        ax.set_ylim([1E-5, 2.])

    if show:
        fig.show()
    
    fig.savefig(path)
    plt.close(fig)


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    output_params = argparse.Namespace(**cfg['output'])
    output_params.output_dir = Path(output_params.output_dir)

    if args.input_file:
        data_file = Path(args.input_file)
        assert data_file.is_file(), f'Cannot find data file: {data_file}'
    else:
        data_file = output_params.output_dir / 'plots' / 'roc.pkl'
        assert data_file.is_file(), f'Cannot find data file: {data_file}'

    if args.output:
        output_file = Path(args.output) / 'roc.pdf'
    else:
        output_file = output_params.output_dir / 'plots' / 'roc.pdf'

    if args.label:
        output_file = output_file.with_stem(f"{output_file.stem}_{args.label}")

    with open(data_file, 'rb') as f:
        roc_data = pickle.load(f)

    plot_roc(roc_data, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', 
        type=str, required=True, help='BDT configuration file (.yml)')
    parser.add_argument('-f', '--file', dest='input_file', 
        type=str, help='pickle data file')
    parser.add_argument('-o', '--output', dest='output', 
        type=str, help='output file path')
    parser.add_argument('-l', '--label', dest='label', 
        type=str, help='output file label')
    args, _ = parser.parse_known_args()

    main(args)