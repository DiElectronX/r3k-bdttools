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
from utils import Logger, make_file_name, save_model, logloss_plot, auc_plot, aucpr_plot, roc_curve, pr_curve


def train_bdt(args):
    args.outdir = os.path.join(args.outdir if args.outdir else '.', args.modelname)
    os.makedirs(args.outdir, exist_ok=True)
    lgr = Logger(os.path.join(args.outdir, f'log_{make_file_name(args)}.txt'), verbose=args.verbose)

    # read input data
    lgr.log(f'Signal File: {args.sigfile}')
    lgr.log(f'Background File: {args.bkgfile}')

    with ur.open(args.sigfile) as sigfile:
        sig_dict = sigfile['mytree'].arrays(
            args.features, cut=args.preselection if args.preselection else None, entry_stop=args.stop_sig, library='np')
        signal = np.stack(list(sig_dict.values()))
        sig_weights = sigfile['mytree'].arrays(
            [args.sample_weights], cut=args.preselection if args.preselection else None, entry_stop=args.stop_sig, library="np")[args.sample_weights]

    with ur.open(args.bkgfile) as bkgfile:
        bkg_dict = bkgfile['mytree'].arrays(
            args.features, cut=args.preselection if args.preselection else None, entry_stop=args.stop_bkg, library='np')
        backgr = np.stack(list(bkg_dict.values()))
        bkg_weights = np.ones(backgr.shape[1])

    # load model info
    lgr.log(f'Model Name: {args.modelname}')
    lgr.log(f'Decay: {args.decay}')
    lgr.log(f'Inputs: {args.features}')

    if args.preselection:
        lgr.log('Preselection Cuts:')
        lgr.log(f'  - {args.preselection}')

    lgr.log('Model Hyperparameters:')
    for arg in list(vars(args))[5:]:
        lgr.log(f'  -{arg}: {getattr(args, arg)}')

    # down-sample background events
    bkg_sig_ratio = 3
    backgr, bkg_weights = resample(backgr.T, bkg_weights, n_samples=round(bkg_sig_ratio*signal.shape[1]), replace=False, random_state=271996)
    backgr = backgr.T

    # format input data
    X = np.transpose(np.concatenate((signal, backgr), axis=1))
    y = np.concatenate((np.ones(signal.shape[1]), np.zeros(backgr.shape[1])))
    weights = np.concatenate((sig_weights, bkg_weights))

    # initialize model
    bdt = XGBClassifier(
            max_depth=args.depth,
            n_estimators=args.ntree,
            learning_rate=args.lrate,
            min_child_weight=args.nodeweight,
            gamma=args.gamma,
            subsample=args.subsample,
            scale_pos_weight=args.scaleweight,
            objective='binary:'+args.lossfunction,
            eval_metric=['logloss','error','auc','aucpr','map'],
    )

    # standard training
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.05, random_state=42)
    eval_set = ((X_train, y_train), (X_test, y_test))

    # train model
    start = time.perf_counter()
    bdt.fit(X_train, y_train, eval_set=eval_set,
            sample_weight=weights_train, verbose=2 if args.verbose else 0)
    lgr.log(f'Elapsed Training Time = {round(time.perf_counter() - start)}s')

    # cross-validation metrics
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=271996)
    cv_results = cross_val_score(bdt, X_test, y_test, cv=skf, n_jobs=-1, scoring='accuracy', fit_params={'sample_weight': weights_test},verbose=2 if args.verbose else 0)
    # save model
    save_model(bdt, args, ['.pkl', '.txt', '.text', '.json'], lgr)

    # save metrics
    results = bdt.evals_result()
    lgr.log(f'Accuracy: {round(cv_results.mean()*100,1)}%')

    # plot loss curve
    if args.plot:
        logloss_plot(results, args)
        auc_plot(results, args)
        aucpr_plot(results, args)
        roc_curve(bdt, X_test, y_test, weights_test, args, skf)
        pr_curve(bdt, X_test, y_test, weights_test, args, skf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', dest='modelname',
                        default='xgbmodel', type=str, help='model name')
    parser.add_argument('--outdir', dest='outdir',
                        default='.', type=str, help='output directory')
    parser.add_argument('--sigfile', dest='sigfile', type=str,
                        required=True,  help='file with signal examples')
    parser.add_argument('--bkgfile', dest='bkgfile', type=str,
                        required=True, help='file with background examples')
    parser.add_argument('--decay', dest='decay', default='kee',
                        type=str, choices=['kmumu', 'kee'], help='decay type')
    parser.add_argument('--label', dest='label', default='',
                        type=str, help='output file label')
    parser.add_argument('--ntree', dest='ntree', default=750,
                        type=int, help='number of trees')
    parser.add_argument('--depth', dest='depth', default=6,
                        type=int, help='tree depth')
    parser.add_argument('--lrate', dest='lrate', default=0.1,
                        type=float, help='learning rate')
    parser.add_argument('--subsample', dest='subsample',
                        default=1.0, type=float, help='fraction of evts')
    parser.add_argument('--gamma', dest='gamma', default=3.0,
                        type=float, help='gamma factor')
    parser.add_argument('--nodeweight', dest='nodeweight',
                        default=1.0, type=float, help='weight for node to be split')
    parser.add_argument('--scaleweight', dest='scaleweight', default=1.0,
                        type=float, help='scale of negative to positive classes')
    parser.add_argument('--lossfunction', dest='lossfunction',
                        default='logitraw', type=str, help='loss function')
    parser.add_argument('--nbkg', dest='stop_bkg', default=None,
                        type=int, help='number of background training examples')
    parser.add_argument('--nsig', dest='stop_sig', default=None,
                        type=int, help='number of signal training examples')
    parser.add_argument('--no_plot', dest='plot', action='store_false',
                        help='dont add loss plot to output directory')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='print parameters and training progress to stdout')
    args, unknown = parser.parse_known_args()

    # Select Input Variables
    args.features = ['Bprob', 'BsLxy', 'L2iso/L2pt', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id']
    args.sample_weights = 'trig_wgt'

    # Preselection Cuts
    args.preselection = '(KLmassD0 > 2.) & ((Mll>1.05) & (Mll<2.45))'

    train_bdt(args)
