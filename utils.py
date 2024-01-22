import os
import logging
import ast
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from xgboost import Booster
from joblib import dump,load
from glob import glob
from pathlib import Path
from sklearn.metrics import auc, RocCurveDisplay, PrecisionRecallDisplay


class Logger():
    def __init__(self, filepath, verbose=True):        
        self.filepath = filepath
        self.verbose = verbose
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        
        log_format = '%(levelname)s | %(asctime)s | %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

    def log(self, string):
        with open(self.filepath, 'a+') as f:
            f.write(string+'\n')
        if self.verbose:
            logging.info(string)


def make_file_name(args):
    name_blocks = [
        args.modelname,
        'Nsig'+str(args.stop_sig)+'_Nbkg' +
        str(args.stop_bkg) if (args.stop_sig or args.stop_bkg) else '',
        args.label,
    ]
    return '_'.join(filter(None, name_blocks))


def save_model(model, args, formats, logger):
    output_name = make_file_name(args)

    if '.pkl' in formats:
        name = os.path.join(args.outdir, output_name+'.pkl')
        dump(model, name)
        if logger:
            logger.log(f'Saving Model {name}')
    if '.text' in formats:
        name = os.path.join(args.outdir, output_name+'.text')
        booster = model.get_booster()
        booster.dump_model(name, dump_format='text')
        if logger:
            logger.log(f'Saving Model {name}')
    if '.json' in formats:
        name = os.path.join(args.outdir, output_name+'.json')
        model.save_model(name)
        if logger:
            logger.log(f'Saving Model {name}')
    if '.txt' in formats:
        name = os.path.join(args.outdir, output_name+'.txt')
        model.save_model(name)
        if logger:
            logger.log(f'Saving Model {name}')

def load_bdt(args):
    args.filepath = args.model if args.format in args.model else args.model+args.format
    assert os.path.exists(args.filepath)

    if ('pkl' in args.format) or ('pickle' in args.format):
        return load(args.filepath)
    else:
        bdt = Booster()
        bdt.load_model(args.filepath)
        return bdt 

def preprocess_files(input_files, nparts, total):
    filelist = [input_files] if input_files.endswith('.root') else glob(input_files+'/**/*.root',recursive=True)[:total]
    if nparts==1:
        outfiles = filelist
    else:
        outfiles = np.array_split(np.array(filelist), nparts if nparts!=-1 else mp.cpu_count())

    if not outfiles: 
        raise ValueError('Invalid input path/file')
    return outfiles


def load_dir_args(args):
    if args.log:
        logname = args.log
    else:
        logs = [f for f in os.listdir(args.fromdir) if (('log_' in f) and ('.txt' in f))]
        if len(logs)==1:
            logname = logs[0]
        else:
            raise KeyError('Multiple viable log files, use "--log" flag to pick one')

    with open(os.path.join(args.fromdir, logname)) as f:
        for line in f:
            if 'Decay: ' in line:
                args.decay = line.split('Decay: ', 1)[1].strip()
            if 'Inputs: ' in line:
                args.features = ast.literal_eval(
                    line.split('Inputs: ', 1)[1].strip())
            if ('Saving Model' in line) and (args.format in line):
                args.model = line.split('Saving Model ', 1)[1].strip()

    print(f'Parsing {args.fromdir} Directory')
    print(f'Measuring {args.decay} Decay')
    print(f'Using Model {args.model}')
    print(f'Using Input Vector {args.features}')


def check_rm_files(files=[]):
    for fl in files:
        if os.path.isfile(fl):
            os.system('rm '+fl)


def edit_filename(path, prefix='', suffix=''):
    path = Path(path)
    return os.path.join(str(path.parent), (prefix+'_' if prefix else '') + str(path.stem) + ('_'+suffix if suffix else '') + str(path.suffix))

def logloss_plot(results, args):
    fig, ax = plt.subplots()
    train_logloss = results['validation_0']['logloss']
    test_logloss  = results['validation_1']['logloss']
    ax.plot(np.arange(len(train_logloss)), train_logloss, label='Train')
    ax.plot(np.arange(len(test_logloss)), test_logloss, label='Test')
    ax.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('Log Loss Curve')
    plt.savefig(os.path.join(args.outdir, f'logloss_{make_file_name(args)}.png'))

def auc_plot(results, args):
    fig, ax = plt.subplots()
    train_logloss = results['validation_0']['auc']
    test_logloss  = results['validation_1']['auc']
    ax.plot(np.arange(len(train_logloss)), train_logloss, label='Train')
    ax.plot(np.arange(len(test_logloss)), test_logloss, label='Test')
    ax.legend()
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC Curve')
    plt.savefig(os.path.join(args.outdir, f'auc_{make_file_name(args)}.png'))
    
def aucpr_plot(results, args):
    fig, ax = plt.subplots()
    train_logloss = results['validation_0']['aucpr']
    test_logloss  = results['validation_1']['aucpr']
    ax.plot(np.arange(len(train_logloss)), train_logloss, label='Train')
    ax.plot(np.arange(len(test_logloss)), test_logloss, label='Test')
    ax.legend()
    plt.xlabel('Epoch')
    plt.ylabel('PR AUC')
    plt.title('Precision-Recall AUC Curve')
    plt.savefig(os.path.join(args.outdir, f'aucpr_{make_file_name(args)}.png'))

def roc_curve(model, X, y, weights, args, cv):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        model.fit(X[train], y[train], sample_weight=weights[train])
        viz = RocCurveDisplay.from_estimator(
            model,
            X[test],
            y[test],
            name=f'ROC fold {fold}',
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == cv.get_n_splits() - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color='grey',
        alpha=0.2,
        label=r'$\pm$ 1 std. dev.',
    )

    ax.set(
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        title=f'Mean ROC curve with variability\n(Positive label "Signal")',
    )
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(args.outdir, f'roc_{make_file_name(args)}.png'))

def pr_curve(model, X, y, weights, args, cv):
    precisions = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        model.fit(X[train], y[train], sample_weight=weights[train])

        viz = PrecisionRecallDisplay.from_estimator(
            model,
            X[test],
            y[test],
            name=f'PR fold {fold}',
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == cv.get_n_splits() - 1),
        )

        interp_precision = np.interp(mean_recall, viz.recall[::-1], viz.precision[::-1])
        interp_precision[0] = 1.0
        precisions.append(interp_precision)
        aucs.append(auc(viz.recall, viz.precision))

    mean_precision = np.mean(precisions, axis=0)
    # mean_precision[-1] = 0.0
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)
    ax.plot(
        mean_recall,
        mean_precision,
        color='b',
        label=r'Mean PR (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_precision = np.std(precisions, axis=0)
    precisions_upper = np.minimum(mean_precision + std_precision, 1)
    precisions_lower = np.maximum(mean_precision - std_precision, 0)
    ax.fill_between(
        mean_recall,
        precisions_lower,
        precisions_upper,
        color='grey',
        alpha=0.2,
        label=r'$\pm$ 1 std. dev.',
    )

    ax.set(
        xlabel='Recall',
        ylabel='Precision',
        title=f'Mean PR curve with variability\n(Positive label "Signal")',
    )
    ax.legend(loc='center left')
    plt.savefig(os.path.join(args.outdir, f'pr_{make_file_name(args)}.png'))