import os
from pathlib import Path
import shutil
import time
import argparse
import yaml
import gc
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from logging import DEBUG, INFO, WARNING, ERROR
from utils import R3KLogger, ROCPlotterKFold, read_bdt_arrays, \
    save_bdt_arrays, load_external_model, edit_filename, get_branches


def bdt_inference(dataset_params, model_params, output_params, args):
    debug_n_evts = 10000 if args.debug else None

    # configuration for outputs & logging
    os.makedirs(output_params.output_dir, exist_ok=True)
    base_filename = os.path.join(output_params.output_dir, 'measurement.root')
    lgr = R3KLogger(
        os.path.join(output_params.output_dir, output_params.log_file), 
        verbose=args.verbose,
        append=True if args.cached_model else False,
    )
    
    with open(args.config, 'r') as f:
        cfg_str = f'cfg_path: {args.config}\n{f.read()}'

    if args.cached_model:
        cached_filepath = os.path.join(output_params.output_dir, 'model.json') if isinstance(args.cached_model,bool) else args.cached_model
        assert os.path.isfile(cached_filepath), 'No Valid Model File Found'
        # Load XGBoost model if stored in output x
        model = XGBClassifier()
        model.load_model(cached_filepath)
        lgr.log(f'Running Additional Inference with Model {cached_filepath}')

    else:
        lgr.log(f'Configuration File:\n{cfg_str}')

        # load data & mc arrays from input files
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
        model = load_external_model(model_params.template_file, debug=args.debug)
        
        # set K-fold validation scheme for retraining model
        skf = StratifiedKFold(n_splits=3)

        # K-fold loop for data measurement
        scores = np.empty(X_data.shape[0], dtype=np.float64)
        event_idxs = np.empty(X_data.shape[0], dtype=np.int64)
        last_idx = -1
        
        if args.plot:
            roc = ROCPlotterKFold(skf)

        lgr.log(f'Training Model for Data Measurement ({skf.get_n_splits()}-fold x-val)', just_print=True)
        start = time.perf_counter()
        for fold, (train_data, test_data) in enumerate(skf.split(X_data, y_data)):
            # mask all training events (low-q2 + mass sidebands for data)
            train_mask = np.concatenate((mask_sig, mask_bkg[train_data]))

            # split events into training and validation sets from mc and data fold
            split = train_test_split(
                np.concatenate((X_mc, X_data[train_data]))[train_mask], 
                np.concatenate((y_mc, y_data[train_data]))[train_mask], 
                np.concatenate((weights_mc, weights_data[train_data]))[train_mask],
                test_size=0.05, random_state=271996
            )
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
            fill_idxs = slice(last_idx+1,last_idx+test_data.size+1)
            scores[fill_idxs] = model.predict_proba(X_data[test_data])[:,1].astype(np.float64)
            event_idxs[fill_idxs] = test_data
            last_idx = last_idx+test_data.size

            # add line to roc plot for fold
            if args.plot:
                roc.add_fold(model, X_val, y_val)

            lgr.log(f'Finished fold {fold+1} of {skf.get_n_splits()}', just_print=True)
            del split, train_mask, X_train, X_val, y_train, y_val, weights_train, \
                weights_val, eval_set, train_data, test_data
            gc.collect()

        lgr.log(f'Elapsed Training/Inference Time on Data = {round(time.perf_counter() - start)}s', just_print=True)

        # save plots
        if args.plot:
            roc.save(os.path.join(output_params.output_dir, 'roc.png'), zoom=True, show=False)

        # save data measurement file
        output_filename = edit_filename(base_filename, suffix='data')
        output_branch_names = get_branches(output_params, ['common','data'])
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
        del scores, event_idxs
        gc.collect()

        # # TODO down-sample background events
        # downsample = False
        # if downsample:
        #     pass

        # K-fold loop for rare mc measurement
        scores = np.empty(X_mc.shape[0], dtype=np.float64)
        event_idxs = np.empty(X_mc.shape[0], dtype=np.int64)
        last_idx = -1

        lgr.log(f'Training Model for MC Measurement ({skf.get_n_splits()}-fold x-val)', just_print=True)
        start = time.perf_counter()
        for fold, (train_mc, test_mc) in enumerate(skf.split(X_mc, y_mc)):
            # mask all training events (low-q2 + mass sidebands for data)
            train_mask = np.concatenate((mask_sig[train_mc], mask_bkg))

            # split events into training and validation sets from mc fold and data
            split = train_test_split(
                np.concatenate((X_mc[train_mc], X_data))[train_mask], 
                np.concatenate((y_mc[train_mc], y_data))[train_mask], 
                np.concatenate((weights_mc[train_mc], weights_data))[train_mask], 
                test_size=0.05, random_state=271996
            )
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
            fill_idxs = slice(last_idx+1,last_idx+test_mc.size+1)
            scores[fill_idxs] = model.predict_proba(X_mc[test_mc])[:,1].astype(np.float64)
            event_idxs[fill_idxs] = test_mc
            last_idx = last_idx+test_mc.size

            lgr.log(f'Finished fold {fold+1} of {skf.get_n_splits()}', just_print=True)
            del split, train_mask, X_train, X_val, y_train, y_val, weights_train, \
                weights_val, eval_set, train_mc, test_mc
            gc.collect()

        lgr.log(f'Elapsed Training/Inference Time = {round(time.perf_counter() - start)}s', just_print=True)

        # save rare mc measurement file
        output_filename = edit_filename(base_filename, suffix='rare')
        output_branch_names = get_branches(output_params, ['common','mc'])
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
        del scores, event_idxs
        gc.collect()

        # Delete arrays needed for training, save memory
        del X_mc, cutvars_mc, weights_mc, X_data, cutvars_data, weights_data, \
            y_mc, y_data, mask_sig, mask_bkg_lowq2, mask_bkg_sideband
        gc.collect()

        # Save XGBoost model
        model.save_model(os.path.join(output_params.output_dir, 'model.json'))

    # predict bdt scores for other data files in config
    if dataset_params.other_data_files:
        for data_name, data_file in dataset_params.other_data_files.items():
            # load arrays from input file
            X_data_extra, _, _ = read_bdt_arrays(
                data_file, 
                dataset_params.tree_name, 
                model_params.features, 
                None, 
                model_params.preselection, 
                (dataset_params.b_mass_branch, dataset_params.ll_mass_branch),
                n_evts=debug_n_evts,
            )

            # bdt inference
            scores = model.predict_proba(X_data_extra)[:,1].astype(np.float64)

            # save jpsi mc measurement file
            output_filename = edit_filename(base_filename, suffix=data_name)
            output_branch_names = get_branches(output_params, ['common','data'])
            save_bdt_arrays(
                data_file, 
                dataset_params.tree_name, 
                output_filename, 
                dataset_params.tree_name, 
                output_branch_names, 
                output_params.score_branch, 
                scores, 
                preselection=model_params.preselection, 
                n_evts=debug_n_evts,
            )
            lgr.log(f'Data ({data_name}) Measurement File: {output_filename}')
            del X_data_extra, scores
            gc.collect()

    # predict bdt scores for other mc files in config
    if dataset_params.other_mc_files:
        for mc_name, mc_file in dataset_params.other_mc_files.items():
            X_mc_extra, _, _ = read_bdt_arrays(
                mc_file, 
                dataset_params.tree_name, 
                model_params.features, 
                model_params.sample_weights, 
                model_params.preselection, 
                (dataset_params.b_mass_branch, dataset_params.ll_mass_branch),
                n_evts=debug_n_evts,
            )

            # bdt inference
            scores = model.predict_proba(X_mc_extra)[:,1].astype(np.float64)

            # save jpsi mc measurement file
            output_filename = edit_filename(base_filename, suffix=mc_name)
            output_branch_names = get_branches(output_params, ['common','mc'])
            save_bdt_arrays(
                mc_file, 
                dataset_params.tree_name, 
                output_filename, 
                dataset_params.tree_name, 
                output_branch_names, 
                output_params.score_branch, 
                scores, 
                preselection=model_params.preselection, 
                n_evts=debug_n_evts,
            )
            lgr.log(f'MC ({mc_name}) Measurement File: {output_filename}')
            del X_mc_extra, scores
            gc.collect()


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    model_params = argparse.Namespace(**cfg['model'])
    output_params = argparse.Namespace(**cfg['output'])

    if args.debug:
        args.verbose = True
        output_params.output_dir = Path('outputs/tmp')

    bdt_inference(dataset_params, model_params ,output_params, args)

    if args.debug:
        if output_params.output_dir.exists() and output_params.output_dir.is_dir():
            shutil.rmtree(output_params.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', 
        type=str, required=True, help='BDT configuration file (.yml)')
    parser.add_argument('-np', '--no_plot', dest='plot', 
        action='store_false', help='dont add loss plot to output directory')
    parser.add_argument('-v', '--verbose', dest='verbose', 
        action='store_true', help='print parameters and training progress to stdout')
    parser.add_argument('-cm', '--cached_model', dest='cached_model', nargs='?',
        const=True, default=False, help='only run prediction on extra samples with cached model, no retraining')
    parser.add_argument('-db', '--debug', dest='debug', 
        action='store_true', help='debug mode: run in verbose mode with scaled-down model & datasets')
    args, _ = parser.parse_known_args()

    main(args)