import os
from pathlib import Path
import shutil
import time
import argparse
import yaml
import json
import gc
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from logging import DEBUG, INFO, WARNING, ERROR
from utils import R3KLogger, ROCPlotterKFold, FeatureImportancePlotterKFold, ScorePlotterKFold, \
                  read_bdt_arrays, save_bdt_arrays, load_external_model, edit_filename, get_branches, save_kfold_model

rand_seed = 271996

def bdt_inference(dataset_params, model_params, output_params, args):
    debug_n_evts = 100000 if args.debug else None

    # configuration for outputs & logging
    os.makedirs(output_params.output_dir, exist_ok=True)
    os.makedirs(output_params.output_dir / 'models', exist_ok=True)
    base_filename = output_params.output_dir / 'measurement.root'
    lgr = R3KLogger(
        output_params.output_dir / output_params.log_file,
        verbose=args.verbose,
        append=True if args.cached_model else False,
    )
    
    with open(args.config, 'r') as f:
        cfg_str = f'cfg_path: {args.config}\n{f.read()}'

    if args.cached_model:
        cached_model = Path(args.cached_model)
        if cached_model.is_dir():
            cached_filepath = cached_model / 'models' / 'model_final.json'
            assert cached_filepath.is_file(), 'No Valid Model File Found'

        elif cached_model.is_file():
            cached_filepath = cached_model

        # Load XGBoost model if stored in output x
        model = XGBClassifier()
        model.load_model(cached_filepath)
        lgr.log(f'Running Inference with Model {cached_filepath}')
        formatted_model_info = json.dumps({k:v for k,v in model.get_params().items() if v is not None}, indent=4)
        lgr.log(f'Model Architecture:\n{type(model)}\n{formatted_model_info}\n')

    else:
        lgr.log(f'Configuration File:\n{cfg_str}\n')

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
        formatted_model_info = json.dumps({k:v for k,v in model.get_params().items() if v is not None}, indent=4)
        lgr.log(f'Model Architecture:\n{type(model)}\n{formatted_model_info}\n')
        
        # set K-fold validation scheme for retraining model
        # skf = StratifiedKFold(n_splits=3)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rand_seed)

        # K-fold loop for data measurement
        scores = np.empty(X_data.shape[0], dtype=np.float64)
        event_idxs = np.empty(X_data.shape[0], dtype=np.int64)
        last_idx = -1
        
        # initialize plotters
        if args.plot:
            roc = ROCPlotterKFold(skf)
            feat_imp = FeatureImportancePlotterKFold(skf, features=model_params.feature_labels)
            score_dist = ScorePlotterKFold(skf)

        lgr.log(f'Training Model for Data Measurement ({skf.get_n_splits()}-fold x-val)', just_print=True)
        start = time.perf_counter()
        for fold, (train_data, test_data) in enumerate(skf.split(X_data, y_data)):
            # mask all training events (low-q2 + mass sidebands for data)
            train_mask = np.concatenate((mask_sig, mask_bkg[train_data]))

            # split events into training and validation sets from mc and data fold
            combined_X = np.concatenate((X_mc, X_data[train_data]))[train_mask]
            combined_y = np.concatenate((y_mc, y_data[train_data]))[train_mask]
            combined_w = np.concatenate((weights_mc, weights_data[train_data]))[train_mask]

            split = train_test_split(
                combined_X, 
                combined_y, 
                combined_w,
                test_size=0.05, 
                random_state=rand_seed,
                stratify=combined_y
            )
            X_train, X_val, y_train, y_val, weights_train, weights_val = split

            # Reweight signal events
            sum_w_sig_train = np.sum(weights_train[y_train == 1])
            sum_w_bkg_train = np.sum(weights_train[y_train == 0])
            train_sf = sum_w_bkg_train / sum_w_sig_train if sum_w_sig_train > 0 else 1.0
            weights_train[y_train == 1] *= train_sf

            sum_w_sig_val = np.sum(weights_val[y_val == 1])
            sum_w_bkg_val = np.sum(weights_val[y_val == 0])
            val_sf = sum_w_bkg_val / sum_w_sig_val if sum_w_sig_val > 0 else 1.0
            weights_val[y_val == 1] *= val_sf

            # Clip any weirdly weighted signal events
            assert np.all(weights_train[y_train == 0] == 1.)
            max_w_sig = np.percentile(weights_train[y_train==1], 99) * 2
            weights_train[y_train==1] = np.clip(weights_train[y_train==1], 0, max_w_sig)            
            weights_val[y_val==1] = np.clip(weights_val[y_val==1], 0, max_w_sig)

            eval_set = ((X_train, y_train), (X_val, y_val))

            # fit model
            model.fit(
                X_train, 
                y_train, 
                eval_set=eval_set, 
                sample_weight=weights_train,
                sample_weight_eval_set=[weights_train, weights_val],
                verbose=100 if args.verbose else 0
            )

            # predict bdt scores on data events designated for testing in fold
            fill_idxs = slice(last_idx+1,last_idx+test_data.size+1)
            scores[fill_idxs] = model.predict_proba(X_data[test_data])[:,1].astype(np.float64)
            event_idxs[fill_idxs] = test_data
            last_idx = last_idx+test_data.size

            # add data to to plots for fold
            if args.plot:
                roc.add_fold(model, X_val, y_val)
                feat_imp.add_fold(model)
                score_dist.add_fold(
                    model, 
                    X_train, 
                    y_train, 
                    X_val, 
                    y_val, 
                    w_train=weights_train, 
                    w_val=weights_val
                )

            # Save data fold model
            save_kfold_model(model, output_params.output_dir / 'models', fold, prefix='model_data_fold')

            lgr.log(f'Finished fold {fold+1} of {skf.get_n_splits()}', just_print=True)
            del split, train_mask, X_train, X_val, y_train, y_val, weights_train, \
                weights_val, eval_set, train_data, test_data
            gc.collect()
        lgr.log(f'Elapsed Training/Inference Time on Data = {round(time.perf_counter() - start)}s', just_print=True)

        # Reorder events after KFold split
        orig_order = np.argsort(event_idxs)
        scores = scores[orig_order]

        # save plots
        if args.plot:
            plot_dir = output_params.output_dir / 'plots'
            plot_dir.mkdir(exist_ok=True)
            roc.save(plot_dir / 'roc.pdf')
            feat_imp.save(plot_dir / 'feature_importance.pdf')
            score_dist.save(plot_dir / 'scores.pdf')

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
            None, 
            preselection=model_params.preselection, 
            n_evts=debug_n_evts,
        )
        lgr.log(f'Data Measurement File: {output_filename}')
        del scores, event_idxs
        gc.collect()

        # K-fold loop for rare mc measurement
        scores = np.empty(X_mc.shape[0], dtype=np.float64)
        event_idxs = np.empty(X_mc.shape[0], dtype=np.int64)
        last_idx = -1

        lgr.log(f'Training Model for MC Measurement ({skf.get_n_splits()}-fold x-val)', just_print=True)
        start = time.perf_counter()
        for fold, (train_mc, test_mc) in enumerate(skf.split(X_mc, y_mc)):
            
            train_mask = np.concatenate((mask_sig[train_mc], mask_bkg))
            combined_X = np.concatenate((X_mc[train_mc], X_data))[train_mask]
            combined_y = np.concatenate((y_mc[train_mc], y_data))[train_mask]
            combined_w = np.concatenate((weights_mc[train_mc], weights_data))[train_mask]

            split = train_test_split(
                combined_X, 
                combined_y, 
                combined_w,
                test_size=0.05, 
                random_state=rand_seed,
                stratify=combined_y
            )
            X_train, X_val, y_train, y_val, weights_train, weights_val = split

            sum_w_sig = np.sum(weights_train[y_train == 1])
            sum_w_bkg = np.sum(weights_train[y_train == 0])
            train_sf = sum_w_bkg / sum_w_sig if sum_w_sig > 0 else 1.0
            weights_train[y_train == 1] *= train_sf

            sum_w_sig_val = np.sum(weights_val[y_val == 1])
            sum_w_bkg_val = np.sum(weights_val[y_val == 0])
            val_sf = sum_w_bkg_val / sum_w_sig_val if sum_w_sig_val > 0 else 1.0
            weights_val[y_val == 1] *= val_sf

            # Clip any weirdly weighted signal events
            assert np.all(weights_train[y_train == 0] == 1.)
            max_w_sig = np.percentile(weights_train[y_train==1], 99) * 2
            weights_train[y_train==1] = np.clip(weights_train[y_train==1], 0, max_w_sig)            
            weights_val[y_val==1] = np.clip(weights_val[y_val==1], 0, max_w_sig)

            eval_set = ((X_train, y_train), (X_val, y_val))

            model.fit(
                X_train, 
                y_train, 
                sample_weight=weights_train, 
                eval_set=eval_set, 
                sample_weight_eval_set=[weights_train, weights_val],
                verbose=100 if args.verbose else 0
            )

            fill_idxs = slice(last_idx+1,last_idx+test_mc.size+1)
            scores[fill_idxs] = model.predict_proba(X_mc[test_mc])[:,1].astype(np.float64)
            event_idxs[fill_idxs] = test_mc
            last_idx = last_idx+test_mc.size

            save_kfold_model(model, output_params.output_dir / 'models', fold, prefix='model_mc_fold')

            lgr.log(f'Finished fold {fold+1} of {skf.get_n_splits()}', just_print=True)
            del split, train_mask, X_train, X_val, y_train, y_val, weights_train, \
                weights_val, eval_set, train_mc, test_mc, combined_X, combined_y, combined_w
            gc.collect()

        lgr.log(f'Elapsed Training/Inference Time = {round(time.perf_counter() - start)}s', just_print=True)

        orig_order = np.argsort(event_idxs)
        scores = scores[orig_order]

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
            None,
            preselection=model_params.preselection, 
            n_evts=debug_n_evts,
        )
        lgr.log(f'MC (Rare) Measurement File: {output_filename}')
        del scores, event_idxs
        gc.collect()

        lgr.log('Training Final Production Model (Full Statistics)', just_print=True)
            
        # Construct the full training set
        train_mask_full = np.concatenate((mask_sig, mask_bkg))
        X_full = np.concatenate((X_mc, X_data))[train_mask_full]
        y_full = np.concatenate((y_mc, y_data))[train_mask_full]
        w_full = np.concatenate((weights_mc, weights_data))[train_mask_full]

        # Same signal reweighting
        sum_w_sig = np.sum(w_full[y_full == 1])
        sum_w_bkg = np.sum(w_full[y_full == 0])
        prod_sf = sum_w_bkg / sum_w_sig if sum_w_sig > 0 else 1.0
        w_full[y_full == 1] *= prod_sf

        # Clip any weirdly weighted signal events
        assert np.all(w_full[y_full == 0] == 1.)
        max_w_sig = np.percentile(w_full[y_full==1], 99) * 2
        w_full[y_full==1] = np.clip(w_full[y_full==1], 0, max_w_sig)

        # TODO: Calculate average best_ntrees from folds
        # n_trees = int(np.mean(best_iteration_list))
        n_trees = 6000
        model.set_params(early_stopping_rounds=False)
        model.set_params(n_estimators=n_trees)

        model.fit(
            X_full, 
            y_full, 
            sample_weight=w_full,
            verbose=100 if args.verbose else 0
        )

        # Delete arrays needed for training, save memory
        del X_mc, cutvars_mc, weights_mc, X_data, cutvars_data, weights_data, \
            y_mc, y_data, mask_sig, mask_bkg_lowq2, mask_bkg_sideband
        gc.collect()

        # Save XGBoost model
        model.save_model(output_params.output_dir / 'models' / 'model_final.json')

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
    else:
        output_params.output_dir = Path(output_params.output_dir)

    bdt_inference(dataset_params, model_params ,output_params, args)

    # Optional: remove debug outputs
    # if args.debug:
    #     if output_params.output_dir.exists() and output_params.output_dir.is_dir():
    #         shutil.rmtree(output_params.output_dir)


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