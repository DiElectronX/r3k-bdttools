import os
import time
import argparse
import numpy as np
import uproot as ur
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc, recall_score, precision_score, RocCurveDisplay
from sklearn.utils import resample
# from sklearn.utils.class_weight import compute_sample_weight
from utils import Logger, load_dir_args, load_bdt, make_file_name, edit_filename, check_rm_files, save_model, logloss_plot, auc_plot, aucpr_plot, roc_curve, pr_curve

BACKEND = 'np'
NTUPLE_TREE = 'mytree'

def xval_bdt(args):
    if args.fromdir:
        args.outdir = args.fromdir
    else:
        args.outdir = os.path.join(args.outdir if args.outdir else '.', args.modelname)
    os.makedirs(args.outdir, exist_ok=True)
    lgr = Logger(os.path.join(args.outdir, f'log_{make_file_name(args)}_xval.txt'), verbose=args.verbose)

    # Log info
    lgr.log(f'Signal File: {args.mcfile}')
    lgr.log(f'Background File: {args.datafile}')
    lgr.log(f'Model Name: {args.modelname}')
    lgr.log(f'Decay: {args.decay}')
    lgr.log(f'Inputs: {args.features}')
    
    if args.preselection:
        lgr.log('Preselection Cuts:')
        lgr.log(f'  - {args.preselection}')

    lgr.log('Model Hyperparameters:')
    for arg in list(vars(args))[5:]:
        lgr.log(f'  -{arg}: {getattr(args, arg)}')

    with ur.open(args.mcfile) as mcfile:
        features_mc = mcfile[NTUPLE_TREE].arrays(
            args.features, cut=args.preselection if args.preselection else None, library=BACKEND)
        X_mc = np.stack(list(features_mc.values())).T
        weights_mc = mcfile[NTUPLE_TREE].arrays(
            [args.sample_weights], cut=args.preselection if args.preselection else None, library=BACKEND)[args.sample_weights]
        cutvars_mc = mcfile[NTUPLE_TREE].arrays(
            ['Bmass', 'Mll'], cut=args.preselection if args.preselection else None, library=BACKEND)

    with ur.open(args.datafile) as datafile:
        features_data = datafile[NTUPLE_TREE].arrays(
            args.features, cut=args.preselection if args.preselection else None, library=BACKEND)
        X_data = np.stack(list(features_data.values())).T
        weights_data = np.ones(X_data.shape[0])
        cutvars_data = datafile[NTUPLE_TREE].arrays(
            ['Bmass', 'Mll'], cut=args.preselection if args.preselection else None, library=BACKEND)

    y_mc = np.ones(X_mc.shape[0])
    y_data = np.zeros(X_data.shape[0])

    mask_sig = np.logical_and(cutvars_mc['Mll'] > 1.05, cutvars_mc['Mll'] < 2.45)
    mask_bkg_lowq2 = np.logical_and(cutvars_data['Mll'] > 1.05, cutvars_data['Mll'] < 2.45)
    mask_bkg_sideband = np.logical_or(np.logical_and(cutvars_data['Bmass'] > 4.8, cutvars_data['Bmass'] < 5.), 
                                      np.logical_and(cutvars_data['Bmass'] > 5.4, cutvars_data['Bmass'] < 5.6)) 
    mask_bkg = np.logical_and(mask_bkg_lowq2, mask_bkg_sideband)

    # format input data
    model = XGBClassifier(
            max_depth=3,
            n_estimators=10,
            objective='binary:logitraw',
            eval_metric=['logloss'],
    )
    # model = args.bdt
    
    # set K-fold validation scheme for retraining model
    skf = StratifiedKFold(n_splits=2)

    # split 2022 data into K folds for trainig/validation/measuring data
    event_idxs = np.array([], dtype=np.int64)
    scores = np.array([], dtype=np.float64)
    start = time.perf_counter()
    if args.verbose:
        lgr.log(f'Training Model for Data Measurement ({skf.get_n_splits()}-fold x-val)')
    for fold, (train_data, test_data) in enumerate(skf.split(X_data, y_data)):
        # add all MC signal events to fold of data
        X = np.concatenate((X_mc, X_data[train_data]))
        y = np.concatenate((y_mc, y_data[train_data]))
        weights = np.concatenate((weights_mc, weights_data[train_data]))

        # mask all training events (low-q2 + mass sidebands for data)
        train_mask = np.concatenate((mask_sig, mask_bkg[train_data]))

        # further split events into training and validation sets before fitting
        X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(X[train_mask], y[train_mask], weights[train_mask], test_size=0.05, random_state=271996)
        eval_set = ((X_train, y_train), (X_val, y_val))

        # fit model
        model.fit(X_train, y_train, sample_weight=weights_train, eval_set=eval_set, verbose=2 if args.verbose else 0)

        # predict bdt scores on data events designated for testing in fold
        scores = np.append(scores, np.array([x[1] for x in model.predict_proba(X_data[test_data])], dtype=np.float64))
        event_idxs = np.append(event_idxs, np.array(test_data, dtype=np.int64))

        if args.verbose:
            print(f'Finished fold {fold}')
    if args.verbose:
        print(f'Elapsed Training/Inference Time = {round(time.perf_counter() - start)}s')

    modelname = edit_filename(
        args.filepath, prefix='measurement', suffix=args.label)
    check_rm_files([modelname, modelname.replace(args.format, '.root')])

    with ur.open(args.datafile) as datafile:
        branchlist_data = list(set(args.output_branches['common']) | set(args.output_branches['data']))

        output_branches_data = datafile[NTUPLE_TREE].arrays(
            branchlist_data, cut=args.preselection if args.preselection else None, library=BACKEND)

        for arr in output_branches_data.values():
            arr = arr[event_idxs]

        output_branches_data['xgb'] = scores

        measurement_data_filename = (modelname.split('.')[0] if '.' in modelname else modelname)+'_xval.root'
        with ur.recreate(measurement_data_filename, compression=ur.LZMA(9)) as outfile:
            outfile['mytreefit'] = output_branches_data
    lgr.log(f'Data Measurement File: {measurement_data_filename}')

    # # down-sample background events
    # bkg_sig_ratio = 3
    # backgr, bkg_weights = resample(backgr.T, bkg_weights, n_samples=round(bkg_sig_ratio*signal.shape[1]), replace=False, random_state=271996)
    # backgr = backgr.T

    # split rare MC into K folds for trainig/validation/measuring MC
    event_idxs = np.array([], dtype=np.int64)
    scores = np.array([], dtype=np.float64)
    start = time.perf_counter()
    if args.verbose:
        lgr.log(f'Training Model for MC Measurement ({skf.get_n_splits()}-fold x-val)')
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
        model.fit(X_train, y_train, sample_weight=weights_train, eval_set=eval_set, verbose=2 if args.verbose else 0)

        # predict bdt scores on data events designated for testing in fold
        scores = np.append(scores, np.array([x[1] for x in model.predict_proba(X_mc[test_mc])], dtype=np.float64))
        event_idxs = np.append(event_idxs, np.array(test_mc, dtype=np.int64))

        if args.verbose:
            print(f'Finished fold {fold}')
    if args.verbose:
        print(f'Elapsed Training/Inference Time = {round(time.perf_counter() - start)}s')

    modelname = edit_filename(
        args.filepath, prefix='measurement', suffix=args.label+'rare')
    check_rm_files([modelname, modelname.replace(args.format, '.root')])

    with ur.open(args.mcfile) as mcfile:
        branchlist_mc = list(set(args.output_branches['common']) | set(args.output_branches['mc']))

        output_branches_mc = mcfile[NTUPLE_TREE].arrays(
            branchlist_mc, cut=args.preselection if args.preselection else None, library=BACKEND)

        for arr in output_branches_mc.values():
            arr = arr[event_idxs]

        output_branches_mc['xgb'] = scores

        measurement_mc_filename = (modelname.split('.')[0] if '.' in modelname else modelname)+'_xval.root'
        with ur.recreate(measurement_mc_filename, compression=ur.LZMA(9)) as outfile:
            outfile['mytreefit'] = output_branches_mc
    lgr.log(f'MC (Rare) Measurement File: {measurement_mc_filename}')

    if args.jpsifile:
        scores = np.array([], dtype=np.float64)
        modelname = edit_filename(
            args.filepath, prefix='measurement', suffix=args.label+'jpsi')
        check_rm_files([modelname, modelname.replace(args.format, '.root')])
        with ur.open(args.jpsifile) as jpsifile:
            features_jpsi = jpsifile[NTUPLE_TREE].arrays(
                args.features, cut=args.preselection if args.preselection else None, library=BACKEND)
            X_jpsi = np.stack(list(features_jpsi.values())).T
            
            scores = np.append(scores, np.array([x[1] for x in model.predict_proba(X_jpsi)], dtype=np.float64))

            branchlist_jpsi = list(set(args.output_branches['common']) | set(args.output_branches['mc']))

            output_branches_jpsi = jpsifile[NTUPLE_TREE].arrays(
                branchlist_jpsi, cut=args.preselection if args.preselection else None, library=BACKEND)

            output_branches_jpsi['xgb'] = scores

            measurement_mc_filename = (modelname.split('.')[0] if '.' in modelname else modelname)+'_xval.root'
            with ur.recreate(measurement_mc_filename, compression=ur.LZMA(9)) as outfile:
                outfile['mytreefit'] = output_branches_jpsi
        lgr.log(f'MC (JPsi) Measurement File: {measurement_mc_filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', dest='modelname',
                        default='xgbmodel', type=str, help='model name')
    parser.add_argument('--mcfile', dest='mcfile', type=str,
                        required=True,  help='file with signal examples')
    parser.add_argument('--jpsifile', dest='jpsifile', type=str,
                        help='file for additional jpsi measurement')
    parser.add_argument('--datafile', dest='datafile', type=str,
                        required=True, help='file with background examples')
    parser.add_argument('--fromdir', dest='fromdir', default=None,
                        type=str, help='load params from designated model directory')
    parser.add_argument('--label', dest='label', default='',
                        type=str, help='output file label')
    parser.add_argument('--no_plot', dest='plot', action='store_false',
                        help='dont add loss plot to output directory')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='print parameters and training progress to stdout')
    parser.add_argument('--format', dest='format', default='.pkl',
                        help='format of saved model file')
    args, unknown = parser.parse_known_args()

    # default input variables
    args.features = ['Bprob', 'BsLxy', 'L2iso/L2pt', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id']

    # load from directory
    if args.fromdir:
        load_dir_args(args)

    # load model
    args.bdt = load_bdt(args)

    # Select Input Variables
    args.sample_weights = 'trig_wgt'

    args.output_branches = {
        'common' : ['Bmass', 'Mll'],
        'data'   : [],
        'mc'     : ['trig_wgt'],
    }

    # Preselection Cuts
    args.preselection = '(KLmassD0 > 2.)'

    xval_bdt(args)