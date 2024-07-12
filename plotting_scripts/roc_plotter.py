import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def roc_plotter(data, path, show=False, inset=True):
    mean_fpr   = data['mean_fpr']
    mean_tpr   = data['mean_tpr']
    tprs_lower = data['tprs_lower']
    tprs_upper = data['tprs_upper']
    mean_auc   = data['mean_auc']
    std_auc    = data['std_auc']
    fold_data  = {k:v for k,v in data.items() if 'KFold' in k}

    fig, ax = plt.subplots(figsize=(8, 6),layout='constrained')
    mean_fpr = np.linspace(0, 1, 100)

    if inset:
        inset_ax = ax.inset_axes([0.25, 0.4, 0.65, 0.3],xlim=[.001,.8], ylim=[.9, 1])
        axes = [ax, inset_ax]
    else:
        axes = [ax]

    for axis in axes:
        for name, (data_x, data_y) in fold_data.items():
            axis.plot(data_x, data_y, label=name, linewidth=1, alpha=0.3)

        axis.plot(
            mean_fpr,
            mean_tpr,
            color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        axis.fill_between(
            mean_fpr,
            tprs_upper,
            tprs_lower,
            color='grey',
            alpha=0.2,
            label=r'$\pm$ 1 std. dev.',
        )

        axis.axline(
            (mean_fpr[0],mean_tpr[0]), 
            (mean_fpr[-1],mean_tpr[-1]), 
            color='black', 
            linestyle='--',
            label='Chance level (AUC = 0.5)',
        )

    ax.set_xlabel('False Positive Rate', loc='right')
    ax.set_ylabel('True Positive Rate', loc='top')
    ax.legend(loc='lower right')

    if inset:
        ax.indicate_inset_zoom(inset_ax, edgecolor='black')
        inset_ax.grid(True)
        inset_ax.set_xlabel('')
        inset_ax.set_ylabel('')
        inset_ax.set_xlabel('')
        inset_ax.set_ylabel('')

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
        data_file = output_params.output_dir / 'plots' / 'roc.pkl'
        assert data_file.is_file(), 'Cannot find data file'

    if args.output:
        output_file = Path(args.output)
        output_file.mkdir(exist_ok=True)
    else:
        output_file = output_params.output_dir / 'plots' / 'roc.pdf'

    if args.label:
        output_file = output_file.with_stem('_'.join([str(output_file.stem), args.label]))

    with open(data_file, 'rb') as f:
        roc_data = pickle.load(f)

    roc_plotter(roc_data, output_file)

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