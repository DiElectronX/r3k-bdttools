import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def score_plotter(data, path, show=False):
    bins = np.linspace(-20,15,60)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    scores_train_sig = np.array([])
    scores_train_bkg = np.array([])
    scores_val_sig = np.array([])
    scores_val_bkg = np.array([])

    for i, (label, vals) in enumerate(data.items()):
        scores_train_sig = np.append(scores_train_sig, vals['scores_train_sig'])
        scores_train_bkg = np.append(scores_train_bkg, vals['scores_train_bkg'])
        scores_val_sig = np.append(scores_val_sig, vals['scores_val_sig'])
        scores_val_bkg = np.append(scores_val_bkg, vals['scores_val_bkg'])

    scores_train_sig = scores_train_sig.flatten()
    scores_train_bkg = scores_train_bkg.flatten()
    scores_val_sig = scores_val_sig.flatten()
    scores_val_bkg = scores_val_bkg.flatten()

    scores_train_sig_wgts = np.abs(np.ones_like(scores_train_sig) / scores_train_sig.size)
    scores_train_bkg_wgts = np.abs(np.ones_like(scores_train_bkg) / scores_train_bkg.size)
    scores_val_sig_wgts = np.abs(np.ones_like(scores_val_sig) / scores_val_sig.size)
    scores_val_bkg_wgts = np.abs(np.ones_like(scores_val_bkg) / scores_val_bkg.size)

    train_sig_hist,_ = np.histogram(scores_train_sig, bins=bins, weights=scores_train_sig_wgts)
    train_bkg_hist,_ = np.histogram(scores_train_bkg, bins=bins, weights=scores_train_bkg_wgts)
    val_sig_hist,_ = np.histogram(scores_val_sig, bins=bins, weights=scores_val_sig_wgts)
    val_bkg_hist,_ = np.histogram(scores_val_bkg, bins=bins, weights=scores_val_bkg_wgts)

    train_sig_hist_err = np.sqrt(np.histogram(scores_train_sig, bins=bins, weights=scores_train_sig_wgts**2)[0])
    train_bkg_hist_err = np.sqrt(np.histogram(scores_train_bkg, bins=bins, weights=scores_train_bkg_wgts**2)[0])
    val_sig_hist_err = np.sqrt(np.histogram(scores_val_sig, bins=bins, weights=scores_val_sig_wgts**2)[0])
    val_bkg_hist_err = np.sqrt(np.histogram(scores_val_bkg, bins=bins, weights=scores_val_bkg_wgts**2)[0])

    fig, ax = plt.subplots(figsize=(8, 6),layout='constrained')

    ax.errorbar(bin_centers, train_sig_hist, yerr=train_sig_hist_err, marker = '', drawstyle = 'steps-mid', label='Train, Signal')
    ax.errorbar(bin_centers, train_bkg_hist, yerr=train_bkg_hist_err, marker = '', drawstyle = 'steps-mid', label='Train, Background')
    ax.errorbar(bin_centers, val_sig_hist, yerr=val_sig_hist_err, marker = 'o', fillstyle='none', linestyle = '', label='Validation, Signal')
    ax.errorbar(bin_centers, val_bkg_hist, yerr=val_bkg_hist_err, marker = 'o', fillstyle='none', linestyle = '', label='Validation, Background')
        
    ax.set_xlabel('BDT Score', loc='right')
    ax.set_ylabel('A.U.', loc='top')
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
        data_file = output_params.output_dir / 'plots' / 'scores.pkl'
        assert data_file.is_file(), 'Cannot find data file'

    if args.output:
        output_file = Path(args.output)
        # output_file.mkdir(exist_ok=True)
    else:
        output_file = output_params.output_dir / 'plots' / 'scores.pdf'

    if args.label:
        output_file = output_file.with_stem('_'.join([str(output_file.stem), args.label]))

    with open(data_file, 'rb') as f:
        score_data = pickle.load(f)

    score_plotter(score_data, output_file)

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