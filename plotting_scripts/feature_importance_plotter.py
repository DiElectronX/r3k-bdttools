import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def feature_importance_plotter(data, path, show=False):
    fig, ax = plt.subplots(figsize=(8, 6),layout='constrained')

    width = 0.25
    x = np.arange(len(data['features']))
    for i, (label, vals) in enumerate(data['feature_imps'].items()):
        offset = width * i
        rects = ax.barh(x+offset, vals, width, label=label, align='center')
        
    ax.set_xlabel('Feature Importance', loc='right')
    ax.set_ylabel('Features', loc='top')
    ax.set_yticks(x + width, data['features'])
    ax.legend()

    if show:
        fig.show()

    fig.savefig(path)

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    model_params = argparse.Namespace(**cfg['model'])
    output_params = argparse.Namespace(**cfg['output'])
    
    output_params.output_dir = Path(output_params.output_dir)

    if args.input_file:
        data_file = Path(args.input_file)
        assert data_file.is_file(), 'Cannot find data file'
    else:
        data_file = output_params.output_dir / 'plots' / 'feature_importance.pkl'
        assert data_file.is_file(), 'Cannot find data file'

    if args.output:
        output_file = Path(args.output)
        output_file.mkdir(exist_ok=True)
    else:
        output_file = output_params.output_dir / 'plots' / 'feature_importance.pdf'

    if args.label:
        output_file = output_file.with_stem('_'.join([str(output_file.stem), args.label]))

    with open(data_file, 'rb') as f:
        feature_importance_data = pickle.load(f)

    feature_importance_plotter(feature_importance_data, output_file)

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