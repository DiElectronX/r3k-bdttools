import os
import time
import argparse
import numpy as np
import uproot as ur
from joblib import dump
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


class Logger():
    def __init__(self, filepath, verbose=True):
        self.filepath = filepath
        self.verbose = verbose
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def log(self, string):
        with open(self.filepath, 'a+') as f:
            f.write(string+'\n')
        if self.verbose:
            print(string)


def train_bdt(features, preselection, args, verbose=True):
    outdir = os.path.join(args.outdir if args.outdir else '.', args.modelname)
    os.makedirs(outdir, exist_ok=True)
    lgr = Logger(os.path.join(outdir, 'log.txt'), verbose=verbose)

    # read input data
    lgr.log(f'Signal File: {args.sigfile}')
    lgr.log(f'Background File: {args.bkgfile}')
    with ur.open(args.sigfile) as sigfile, ur.open(args.bkgfile) as bkgfile:
        sig_dict = sigfile['mytree'].arrays(
            features, cut=preselection if preselection else None, entry_stop=args.stop_sig, library='np')
        bkg_dict = bkgfile['mytree'].arrays(
            features, cut=preselection if preselection else None, entry_stop=args.stop_bkg, library='np')
        signal = np.stack(list(sig_dict.values()))
        backgr = np.stack(list(bkg_dict.values()))

    # load model info
    lgr.log(f'Model Name: {args.modelname}')
    lgr.log(f'Decay: {args.decay}')
    lgr.log(f'Inputs: {features}')
    if preselection:
        lgr.log('Preselection Cuts:')
        for k, val in preselection.items():
            lgr.log(f'  -{k}: {val}')
    lgr.log('Model Hyperparameters:')
    for arg in list(vars(args))[5:]:
        lgr.log(f'  -{arg}: {getattr(args, arg)}')

    # load input data
    X = np.transpose(np.concatenate((signal, backgr), axis=1))
    Y = np.concatenate((np.ones(signal.shape[1]), np.zeros(backgr.shape[1])))
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.05, random_state=42)
    eval_set = ((X_train, Y_train), (X_test, Y_test))
    weightTrain = compute_sample_weight(class_weight='balanced', y=Y_train)
    compute_sample_weight(class_weight='balanced', y=Y_test)

    # initialize model
    bdt = XGBClassifier(
            max_depth=args.depth,
            n_estimators=args.ntree,
            learning_rate=args.lrate,
            min_child_weight=args.nodeweight,
            gamma=args.gamma,
            subsample=args.subsample,
            scale_pos_weight=args.scaleweight,
            objective='binary:'+args.lossfunction
    )

    # train model
    start = time.perf_counter()
    bdt.fit(X_train, Y_train, eval_set=eval_set,
            sample_weight=weightTrain, verbose=2 if verbose else 3)
    lgr.log(f'Elapsed Training Time = {round(time.perf_counter() - start)}s')

    # save model
    name_blocks = [
            args.modelname,
            'Nsig'+str(args.stop_sig)+'_Nbkg' +
                       str(args.stop_bkg) if (
                           args.stop_sig or args.stop_bkg) else '',
            args.label,
    ]
    output_name = '_'.join(filter(None, name_blocks))+'.pkl'
    dump(bdt, os.path.join(outdir, output_name))
    lgr.log(f'Saving Model {os.path.join(outdir,output_name)}')

    # save metrics
    y_pred = bdt.predict(X_test)
    predictions = [round(value) for value in y_pred]
    acc = accuracy_score(Y_test, predictions)
    lgr.log(f'Accuracy: {round(acc*100,1)}%')
    results = bdt.evals_result()

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(results['validation_0']['logloss'])),
                results['validation_0']['logloss'], label='Train')
        ax.plot(np.arange(len(results['validation_1']['logloss'])),
                results['validation_1']['logloss'], label='Test')
        ax.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.title('XGBoost Log Loss')
        plt.savefig(os.path.join(outdir, 'logloss.png'))


if __name__ == '__main__':
    print('Imports Done')
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
    args, unknown = parser.parse_known_args()

    # Select Input Variables
    features = ['Bprob', 'BsLxy', 'L2iso/L2pt', 'Kpt/Bmass', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr',
                'Bpt/Bmass', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id', 'L1pt/Bmass', 'L2pt/Bmass', 'L1L2dr']

    # Preselection Cuts
    preselection = ''

    train_bdt(features, preselection, args)
