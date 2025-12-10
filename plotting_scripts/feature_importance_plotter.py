import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

def plot_feature_importance(data, path, show=False):
    """
    Main plotting logic for Feature Importance.
    
    Args:
        data (dict): Dictionary containing 'features' (list) and 'feature_imps' (dict of arrays).
        path (Path or str): Output path for the plot.
        show (bool): Whether to show the plot interactively.
    """

    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')

    features = data['features']
    feature_imps = data['feature_imps']
    
    n_folds = len(feature_imps)
    n_features = len(features)
    
    # Calculate bar height and positions to group
    total_width = 0.8
    bar_height = total_width / n_folds
    y_indices = np.arange(n_features)
    
    # Generate colors
    colors = cm.get_cmap('viridis', n_folds)

    for i, (label, vals) in enumerate(feature_imps.items()):
        # Calculate offset so bars are centered around the tick
        offset = (i - n_folds/2) * bar_height + bar_height/2
        
        ax.barh(
            y_indices + offset, 
            vals, 
            height=bar_height, 
            label=label, 
            align='center',
            alpha=0.8,
            color=colors(i)
        )
        
    ax.set_xlabel('Feature Importance (Gain)', loc='right')
    ax.set_ylabel('Features', loc='top')
    
    # Set Y-ticks to feature names
    ax.set_yticks(y_indices)
    ax.set_yticklabels(features)
    
    # Invert Y axis so the first feature (often most important) is at the top
    ax.invert_yaxis() 
    
    ax.legend(title='Folds', loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.5)

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
        data_file = output_params.output_dir / 'plots' / 'feature_importance.pkl'
        assert data_file.is_file(), f'Cannot find data file: {data_file}'

    if args.output:
        output_file = Path(args.output) / 'feature_importance.pdf'
    else:
        output_file = output_params.output_dir / 'plots' / 'feature_importance.pdf'

    if args.label:
        output_file = output_file.with_stem(f"{output_file.stem}_{args.label}")

    with open(data_file, 'rb') as f:
        feature_importance_data = pickle.load(f)

    plot_feature_importance(feature_importance_data, output_file, show=False)

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