import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_scores(data, path, show=False, logy=False):
    """
    Main plotting logic.
    
    Args:
        data (dict): Dictionary containing fold data.
        path (Path or str): Output path for the plot.
        show (bool): Whether to show the plot interactively.
        logy (bool): Plot Y axis on log scale.
    """

    # Define binning (matches your utils preference, adjust range if needed)    # pprint(data.values())
    score_arrays = [vals for idata in data.values() for key, vals in idata.items() if key.startswith('scores_')]
    if score_arrays:
        all_scores = np.concatenate([np.ravel(arr) for arr in score_arrays])
    else:
        all_scores = np.array([])

    max_score = np.ceil(all_scores.max())
    min_score = np.floor(all_scores.min())
    bins = np.linspace(min_score * 1.1 if min_score < 0 else min_score * 0.9, max_score  * 1.1, int((max_score - min_score) * 2 + 1))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')

    # Plot validation folds independently
    for i, (label, vals) in enumerate(data.items()):
        if vals['scores_val_sig'].size > 0:
            w_sig = (np.ones_like(vals['weights_val_sig']) / vals['weights_val_sig'].sum()) * vals['weights_val_sig']
            h_sig, _ = np.histogram(vals['scores_val_sig'], bins=bins, weights=w_sig, density=False)
            fold_label = 'Validation, Signal (Single Fold)' if i == 0 else None
            fold_sig = ax.step(bin_centers, h_sig, where='mid', color='b', alpha=0.15, linewidth=1, label=fold_label)
        
        if vals['scores_val_bkg'].size > 0:
            w_bkg = (np.ones_like(vals['weights_val_bkg']) / vals['weights_val_bkg'].sum()) * vals['weights_val_bkg']
            h_bkg, _ = np.histogram(vals['scores_val_bkg'], bins=bins, weights=w_bkg, density=False)
            fold_label = 'Validation, Background (Single Fold)' if i == 0 else None
            fold_bkg = ax.step(bin_centers, h_bkg, where='mid', color='r', alpha=0.15, linewidth=1, label=fold_label)

    # Plot aggregated stats
    scores_train_sig, weights_train_sig = [], []
    scores_train_bkg, weights_train_bkg = [], []
    scores_val_sig, weights_val_sig = [], []
    scores_val_bkg, weights_val_bkg = [], []

    for i, (label, vals) in enumerate(data.items()):
        scores_train_sig.append(vals['scores_train_sig'])
        weights_train_sig.append(vals['weights_train_sig'])
        
        scores_train_bkg.append(vals['scores_train_bkg'])
        weights_train_bkg.append(vals['weights_train_bkg'])

        scores_val_sig.append(vals['scores_val_sig'])
        weights_val_sig.append(vals['weights_val_sig'])

        scores_val_bkg.append(vals['scores_val_bkg'])
        weights_val_bkg.append(vals['weights_val_bkg'])

    # Flatten arrays
    scores_train_sig = np.concatenate(scores_train_sig)
    w_train_sig = np.concatenate(weights_train_sig)
    
    scores_train_bkg = np.concatenate(scores_train_bkg)
    w_train_bkg = np.concatenate(weights_train_bkg)

    scores_val_sig = np.concatenate(scores_val_sig)
    w_val_sig = np.concatenate(weights_val_sig)

    scores_val_bkg = np.concatenate(scores_val_bkg)
    w_val_bkg = np.concatenate(weights_val_bkg)

    # Normalize
    w_train_sig = (np.ones_like(w_train_sig) / w_train_sig.sum()) * w_train_sig
    w_train_bkg = (np.ones_like(w_train_bkg) / w_train_bkg.sum()) * w_train_bkg
    w_val_sig = (np.ones_like(w_val_sig) / w_val_sig.sum()) * w_val_sig
    w_val_bkg = (np.ones_like(w_val_bkg) / w_val_bkg.sum()) * w_val_bkg

    # Histograms
    h_train_sig, _ = np.histogram(scores_train_sig, bins=bins, weights=w_train_sig, density=False)
    h_train_bkg, _ = np.histogram(scores_train_bkg, bins=bins, weights=w_train_bkg, density=False)
    h_val_sig, _ = np.histogram(scores_val_sig, bins=bins, weights=w_val_sig, density=False)
    h_val_bkg, _ = np.histogram(scores_val_bkg, bins=bins, weights=w_val_bkg, density=False)

    # Errors (Poisson: sqrt(N) scaled by weight)
    err_train_sig = np.sqrt(np.histogram(scores_train_sig, bins=bins, weights=w_train_sig**2)[0])
    err_train_bkg = np.sqrt(np.histogram(scores_train_bkg, bins=bins, weights=w_train_bkg**2)[0])
    err_val_sig = np.sqrt(np.histogram(scores_val_sig, bins=bins, weights=w_val_sig**2)[0])
    err_val_bkg = np.sqrt(np.histogram(scores_val_bkg, bins=bins, weights=w_val_bkg**2)[0])

    # Plotting
    ax.errorbar(bin_centers, h_train_sig, yerr=err_train_sig, 
                color='b', marker='', drawstyle='steps-mid', label='Train, Signal (Global)', lw=2)
    ax.errorbar(bin_centers, h_train_bkg, yerr=err_train_bkg, 
                color='r', marker='', drawstyle='steps-mid', label='Train, Background (Global)', lw=2)
    
    ax.errorbar(bin_centers, h_val_sig, yerr=err_val_sig, 
                color='b', marker='o', fillstyle='none', linestyle='', label='Validation, Signal (Global)')
    ax.errorbar(bin_centers, h_val_bkg, yerr=err_val_bkg, 
                color='r', marker='o', fillstyle='none', linestyle='', label='Validation, Background (Global)')
        
    ax.set_xlabel('BDT Score', loc='right')
    ax.set_ylabel('nEvents (A.U.)', loc='top')

    # Optional: set Y scale
    if logy:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-3)

    # Format legend
    handles, labels = ax.get_legend_handles_labels()
    target_indices = [2, 5, 0, 3, 1, 4]
    ordered_handles = [handle for (idx, handle) in sorted(zip(target_indices, handles))]
    ordered_labels = [label for (idx, label) in sorted(zip(target_indices, labels))]
    ax.legend(ordered_handles, ordered_labels, loc='upper right', fontsize='small')


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
        data_file = output_params.output_dir / 'plots' / 'scores.pkl'
        assert data_file.is_file(), f'Cannot find data file: {data_file}'

    if args.output:
        output_file = Path(args.output) / 'scores.pdf'
    else:
        output_file = output_params.output_dir / 'plots' / 'scores.pdf'

    if args.label:
        output_file = output_file.with_stem(f'{output_file.stem}_{args.label}')

    with open(data_file, 'rb') as f:
        score_data = pickle.load(f)

    plot_scores(score_data, output_file, show=False)

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