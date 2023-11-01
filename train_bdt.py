import os
import time
import argparse
import numpy as np
import uproot as ur
import pandas as pd
from xgboost import XGBClassifier
from hep_ml.uboost import uBoostBDT
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from utils import Logger, make_file_name, save_model


def train_bdt(args):
    args.outdir = os.path.join(args.outdir if args.outdir else '.', args.modelname)
    os.makedirs(args.outdir, exist_ok=True)
    lgr = Logger(os.path.join(args.outdir, f'log_{make_file_name(args)}.txt'), verbose=args.verbose)

    # read input data
    lgr.log(f'Signal File: {args.sigfile}')
    lgr.log(f'Background File: {args.bkgfile}')
    with ur.open(args.sigfile) as sigfile, ur.open(args.bkgfile) as bkgfile:
        signal = sigfile['mytree'].arrays(
            args.features+args.uniform_features, cut=args.preselection if args.preselection else None, entry_stop=args.stop_sig, library='pd')
        backgr = bkgfile['mytree'].arrays(
            args.features+args.uniform_features, cut=args.preselection if args.preselection else None, entry_stop=args.stop_bkg, library='pd')

    # load model info
    lgr.log(f'Model Name: {args.modelname}')
    lgr.log(f'Decay: {args.decay}')
    lgr.log(f'Inputs: {args.features}')
    if args.preselection:
        lgr.log('Preselection Cuts:')
        lgr.log(f'  - {args.preselection}')
        # TODO reformat preselection cuts
        # for k, val in args.preselection.items():
        #     lgr.log(f'  -{k}: {val}')
    lgr.log('Model Hyperparameters:')
    for arg in list(vars(args))[5:]:
        lgr.log(f'  -{arg}: {getattr(args, arg)}')

    # load input data
    X = pd.concat((signal, backgr))
    Y = np.concatenate((np.ones(len(signal)), np.zeros(len(backgr))))
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.05, random_state=42)
    eval_set = ((X_train, Y_train), (X_test, Y_test))
    weightTrain = compute_sample_weight(class_weight='balanced', y=Y_train)
    weightTest = compute_sample_weight(class_weight='balanced', y=Y_test)

    # initialize model
    clf = DecisionTreeClassifier(max_depth=args.depth)
    bdt = uBoostBDT(
            uniform_features=args.uniform_features,
            train_features=args.features,
            n_neighbors=5,
            uniform_label=0,
            base_estimator=clf,
            n_estimators=args.ntree,
            learning_rate=args.lrate,
            # min_child_weight=args.nodeweight,
            # gamma=args.gamma,
            subsample=args.subsample,
            # scale_pos_weight=args.scaleweight,
            # objective= 'binary:'+args.lossfunction
    )

    # train model
    start = time.perf_counter()
    bdt.fit(
        X_train, 
        Y_train, 
        # eval_set=eval_set,
        sample_weight=weightTrain, 
        # verbose=2 if args.verbose else 0,
    )
    lgr.log(f'Elapsed Training Time = {round(time.perf_counter() - start)}s')

    # save model
    save_model(bdt, args, ['.pkl'], lgr)

    # save metrics
    y_pred = bdt.predict(X_test.drop(args.uniform_features, axis=1))
    predictions = [round(value) for value in y_pred]
    acc = accuracy_score(Y_test, predictions)
    lgr.log(f'Accuracy: {round(acc*100,1)}%')
    # results = bdt.evals_result()

    # if args.plot:
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots()
    #     ax.plot(np.arange(len(results['validation_0']['logloss'])),
    #             results['validation_0']['logloss'], label='Train')
    #     ax.plot(np.arange(len(results['validation_1']['logloss'])),
    #             results['validation_1']['logloss'], label='Test')
    #     ax.legend()
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Log Loss')
    #     plt.title('XGBoost Log Loss')
    #     plt.savefig(os.path.join(args.outdir, f'logloss_{make_file_name(args)}.png'))


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

    # select bdt input variables bdt
    args.features = ['Bprob', 'BsLxy', 'L2iso/L2pt', 'Kpt/Bmass', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Bpt/Bmass', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id']
    # select features to maintain uniformity
    args.uniform_features = ['Bmass']
    # preselection Cuts
    args.preselection = 'KLmassD0 > 2.'

    train_bdt(args)
