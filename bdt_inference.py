import os
import time
import argparse
import yaml
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from logging import DEBUG, INFO, WARNING, ERROR
from utils import R3KLogger, ROCPlotterKFold, read_bdt_arrays, save_bdt_arrays, load_external_model, edit_filename


def bdt_inference(dataset_params, model_params, output_params, args):
    debug_n_evts = None

    # configuration for outputs & logging
    os.makedirs(output_params.output_dir, exist_ok=True)
    base_filename = os.path.join(output_params.output_dir, 'measurement.root')
    lgr = R3KLogger(
        os.path.join(output_params.output_dir, output_params.log_file), 
        verbose=args.verbose
    )
    
    with open(args.config, 'r') as f:
        cfg_str = f'cfg_path: {args.config}\n{f.read()}'

    lgr.log(f'Configuration File:\n{cfg_str}')

    # # load data & mc arrays from input files
    X_mc, cutvars_mc, weights_mc = read_bdt_arrays(
        dataset_params.rare_file, 
        dataset_params.tree_name, 
        model_params.features, 
        model_params.sample_weights, 
        model_params.preselection, 
        (dataset_params.b_mass_branch, dataset_params.ll_mass_branch),
        n_evts=debug_n_evts
    )

    X_data, cutvars_data, weights_data = read_bdt_arrays(
        dataset_params.data_file,
        dataset_params.tree_name,
        model_params.features,
        None, 
        model_params.preselection, 
        (dataset_params.b_mass_branch, dataset_params.ll_mass_branch),
        n_evts=debug_n_evts
    )

    y_mc = np.ones(X_mc.shape[0])
    y_data = np.zeros(X_data.shape[0])

    # create array masks for training
    mask_sig = np.logical_and(
        cutvars_mc[dataset_params.ll_mass_branch] > 1.05, 
        cutvars_mc[dataset_params.ll_mass_branch] < 2.45
    )
    mask_bkg_lowq2 = np.logical_and(
        cutvars_data[dataset_params.ll_mass_branch] > 1.05, 
        cutvars_data[dataset_params.ll_mass_branch] < 2.45
    )
    mask_bkg_sideband = np.logical_or(
        np.logical_and(
            cutvars_data[dataset_params.b_mass_branch] > 4.8, 
            cutvars_data[dataset_params.b_mass_branch] < 5.
        ), 
        np.logical_and(
            cutvars_data[dataset_params.b_mass_branch] > 5.4,
            cutvars_data[dataset_params.b_mass_branch] < 5.6
        )
    ) 
    mask_bkg = np.logical_and(mask_bkg_lowq2, mask_bkg_sideband)

    # load bdt from template file
    model = load_external_model(model_params.template_file)
    
    # set K-fold validation scheme for retraining model
    skf = StratifiedKFold(n_splits=3)

    # K-fold loop for data measurement
    event_idxs = np.array([], dtype=np.int64)
    scores = np.array([], dtype=np.float64)
    if args.plot:
        roc = ROCPlotterKFold(skf)

    lgr.log(f'Training Model for Data Measurement ({skf.get_n_splits()}-fold x-val)', just_print=True)
    start = time.perf_counter()
    for fold, (train_data, test_data) in enumerate(skf.split(X_data, y_data)):
        # mix all MC signal events with fold of background data
        X = np.concatenate((X_mc, X_data[train_data]))
        y = np.concatenate((y_mc, y_data[train_data]))
        weights = np.concatenate((weights_mc, weights_data[train_data]))

        # mask all training events (low-q2 + mass sidebands for data)
        train_mask = np.concatenate((mask_sig, mask_bkg[train_data]))

        # further split events into training and validation sets before fitting
        split = train_test_split(X[train_mask], y[train_mask], weights[train_mask], test_size=0.05, random_state=271996)
        X_train, X_val, y_train, y_val, weights_train, weights_val = split
        eval_set = ((X_train, y_train), (X_val, y_val))

        # fit model
        model.fit(
            X_train, 
            y_train, 
            sample_weight=weights_train, 
            eval_set=eval_set, 
            verbose=2 if args.verbose else 0
        )

        # predict bdt scores on data events designated for testing in fold
        scores = np.append(scores, np.array([x[1] for x in model.predict_proba(X_data[test_data])], dtype=np.float64))
        event_idxs = np.append(event_idxs, np.array(test_data, dtype=np.int64))

        # add line to roc plot for fold
        if args.plot:
            roc.add_fold(model, X_val, y_val)

        lgr.log(f'Finished fold {fold+1} of {skf.get_n_splits()}', just_print=True)

    lgr.log(f'Elapsed Training/Inference Time on Data = {round(time.perf_counter() - start)}s', just_print=True)

    # save plots
    if args.plot:
        roc.save(os.path.join(output_params.output_dir, 'roc.png'), show=False)

    # save data measurement file
    output_filename = edit_filename(base_filename, suffix='data')
    output_branch_names = output_params.output_branches['common'] + output_params.output_branches['common']
    save_bdt_arrays(
        dataset_params.data_file, 
        dataset_params.tree_name, 
        output_filename, 
        dataset_params.tree_name, 
        output_branch_names, 
        output_params.score_branch, 
        scores, 
        event_idxs, 
        preselection=model_params.preselection, 
        n_evts=debug_n_evts,
    )
    lgr.log(f'Data Measurement File: {output_filename}')

    # TODO down-sample background events
    downsample = False
    if downsample:
        pass

    # K-fold loop for rare mc measurement
    event_idxs = np.array([], dtype=np.int64)
    scores = np.array([], dtype=np.float64)
    lgr.log(f'Training Model for MC Measurement ({skf.get_n_splits()}-fold x-val)', just_print=True)
    start = time.perf_counter()
    for fold, (train_mc, test_mc) in enumerate(skf.split(X_mc, y_mc)):
        # add all data background events to fold of mc
        X = np.concatenate((X_mc[train_mc], X_data))
        y = np.concatenate((y_mc[train_mc], y_data))
        weights = np.concatenate((weights_mc[train_mc], weights_data))

        # mask all training events (low-q2 + mass sidebands for data)
        train_mask = np.concatenate((mask_sig[train_mc], mask_bkg))

        # further split events into training and validation sets before fitting
        X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(X[train_mask], y[train_mask], weights[train_mask], test_size=0.05, random_state=271996)
        eval_set = ((X_train, y_train), (X_val, y_val))

        # fit model
        model.fit(
            X_train, 
            y_train, 
            sample_weight=weights_train, 
            eval_set=eval_set, 
            verbose=2 if args.verbose else 0
        )

        # predict bdt scores on data events designated for testing in fold
        scores = np.append(scores, np.array([x[1] for x in model.predict_proba(X_mc[test_mc])], dtype=np.float64))
        event_idxs = np.append(event_idxs, np.array(test_mc, dtype=np.int64))
        lgr.log(f'Finished fold {fold+1} of {skf.get_n_splits()}', just_print=True)

    lgr.log(f'Elapsed Training/Inference Time = {round(time.perf_counter() - start)}s', just_print=True)
    
    # save rare mc measurement file
    output_filename = edit_filename(base_filename, suffix='rare')
    output_branch_names = output_params.output_branches['common'] + output_params.output_branches['mc']
    save_bdt_arrays(
        dataset_params.rare_file, 
        dataset_params.tree_name, 
        output_filename, 
        dataset_params.tree_name, 
        output_branch_names, 
        output_params.score_branch, 
        scores, 
        event_idxs, 
        preselection=model_params.preselection, 
        n_evts=debug_n_evts,
    )
    lgr.log(f'MC (Rare) Measurement File: {output_filename}')

    if dataset_params.jpsi_file:
        # load data & mc arrays from input files
        X_jpsi, cutvars_mc, weights_mc = read_bdt_arrays(
            dataset_params.jpsi_file, 
            dataset_params.tree_name, 
            model_params.features, 
            model_params.sample_weights, 
            model_params.preselection, 
            (dataset_params.b_mass_branch, dataset_params.ll_mass_branch),
            n_evts=debug_n_evts,
        )

        # bdt inference
        scores = np.array([x[1] for x in model.predict_proba(X_jpsi)], dtype=np.float64)

        # save jpsi mc measurement file
        output_filename = edit_filename(base_filename, suffix='jpsi')
        output_branch_names = output_params.output_branches['common'] + output_params.output_branches['mc']
        save_bdt_arrays(
            dataset_params.jpsi_file, 
            dataset_params.tree_name, 
            output_filename, 
            dataset_params.tree_name, 
            output_branch_names, 
            output_params.score_branch, 
            scores, 
            None, 
            preselection=model_params.preselection, 
            n_evts=debug_n_evts,
        )
        lgr.log(f'MC (JPsi) Measurement File: {output_filename}')
 
    if dataset_params.psi2s_file:
        # load data & mc arrays from input files
        X_psi2s, cutvars_mc, weights_mc = read_bdt_arrays(
            dataset_params.psi2s_file, 
            dataset_params.tree_name, 
            model_params.features, 
            model_params.sample_weights, 
            model_params.preselection, 
            (dataset_params.b_mass_branch, dataset_params.ll_mass_branch),
            n_evts=debug_n_evts,
        )

        # bdt inference
        scores = np.array([x[1] for x in model.predict_proba(X_psi2s)], dtype=np.float64)

        # save psi2s mc measurement file
        output_filename = edit_filename(base_filename, suffix='psi2s')
        output_branch_names = output_params.output_branches['common'] + output_params.output_branches['mc']
        save_bdt_arrays(
            dataset_params.psi2s_file, 
            dataset_params.tree_name, 
            output_filename, 
            dataset_params.tree_name, 
            output_branch_names, 
            output_params.score_branch, 
            scores, 
            None, 
            preselection=model_params.preselection, 
            n_evts=debug_n_evts,
        )
        lgr.log(f'MC (Psi2S) Measurement File: {output_filename}')

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    model_params = argparse.Namespace(**cfg['model'])
    output_params = argparse.Namespace(**cfg['output'])

    bdt_inference(dataset_params, model_params ,output_params, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, required=True, help='BDT configuration file (.yml)')
    parser.add_argument('-np', '--no_plot', dest='plot', action='store_false', help='dont add loss plot to output directory')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print parameters and training progress to stdout')
    args, _ = parser.parse_known_args()

    main(args)