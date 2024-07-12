import os
import sys
import logging
import pickle
import importlib.util
import numpy as np
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot as ur
from xgboost import Booster
from joblib import dump,load
from glob import glob
from pathlib import Path
from sklearn.metrics import auc, RocCurveDisplay
from logging import DEBUG, INFO, WARNING, ERROR

BACKEND = 'np'
MPL_BACKEND = 'TkAgg'
# mpl.use(MPL_BACKEND)

class R3KLogger():
    def __init__(self, filepath, verbose=True, append=False):        
        self.filepath = filepath
        self.verbose = verbose
        
        self.base_logger = logging.getLogger('base_logger')
        self.base_logger.setLevel(logging.INFO)
        self.stdout_logger = logging.getLogger('stdout_logger')
        self.stdout_logger.setLevel(logging.INFO)
        self.fout_logger = logging.getLogger('fout_logger')
        self.fout_logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(levelname)s | %(asctime)s | %(message)s')

        self.fh = logging.FileHandler(self.filepath, mode='a' if append else 'w')
        self.fh.setFormatter(self.formatter)
        self.fh.setLevel(logging.INFO)

        self.sh = logging.StreamHandler(sys.stdout)
        self.sh.setFormatter(self.formatter)
        self.sh.setLevel(logging.INFO)

        self.base_logger.addHandler(self.sh)
        self.stdout_logger.addHandler(self.sh)
        self.base_logger.addHandler(self.fh)
        self.fout_logger.addHandler(self.fh)


    def log(self, string, just_print=False, just_write=False, level=INFO):
        if just_print:
            if self.verbose:
                self.stdout_logger.log(level, string)
        elif just_write:
            self.fout_logger.log(level, string)
        else:
            if self.verbose:
                self.base_logger.log(level, string)
            else:
                self.fout_logger.log(level, string)


class ROCPlotterKFold():
    def __init__(self, kf):
        self.kf = kf
        self.ifold = 0
        self.tprs = []
        self.aucs = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.inset_ax = self.ax.inset_axes(
            [0.25, 0.4, 0.65, 0.3],
            xlim=[.001,.8], ylim=[.9, 1],
            # xticklabels=[], yticklabels=[]
        )

    def add_fold(self, model, X, y):
        self.ifold += 1

        _viz = RocCurveDisplay.from_estimator(
            model,
            X,
            y,
            name=f'ROC fold {self.ifold}',
            alpha=0.3,
            lw=1,
            ax=self.ax,
            plot_chance_level=(self.ifold ==self.kf.get_n_splits() - 1),
        )

        _viz = RocCurveDisplay.from_estimator(
            model,
            X,
            y,
            name=f'ROC fold {self.ifold}',
            alpha=0.3,
            lw=1,
            ax=self.inset_ax,
            plot_chance_level=(self.ifold ==self.kf.get_n_splits() - 1),
        )

        _interp_tpr = np.interp(self.mean_fpr, _viz.fpr, _viz.tpr)
        _interp_tpr[0] = 0.0
        self.tprs.append(_interp_tpr)
        self.aucs.append(_viz.roc_auc)


    def save_to_pickle(self, path):
        new_path = path.with_suffix('.pkl')
        with open(new_path,'wb') as pkl_file:
            pickle.dump(self.roc_data, pkl_file)


    def save(self, path, show=False, logy=False, zoom=False):
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        self.roc_data = {
            'mean_fpr' : self.mean_fpr,
            'mean_tpr' : mean_tpr,
            'tprs_lower' : tprs_lower,
            'tprs_upper' : tprs_upper,
            'mean_auc' : mean_auc,
            'std_auc' : std_auc,
        }
        
        axes = [self.ax]
        if zoom:
            axes.append(self.inset_ax)
        
        for ax in axes:
            ax.plot(
                self.roc_data['mean_fpr'],
                self.roc_data['mean_tpr'],
                color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (self.roc_data['mean_auc'], self.roc_data['std_auc']),
                lw=2,
                alpha=0.8,
            )

            ax.fill_between(
                self.roc_data['mean_fpr'],
                self.roc_data['tprs_upper'],
                self.roc_data['tprs_lower'],
                color='grey',
                alpha=0.2,
                label=r'$\pm$ 1 std. dev.',
            )

        self.ax.set_xlabel('False Positive Rate', loc='right')
        self.ax.set_ylabel('True Positive Rate', loc='top')
        self.ax.legend(loc='lower right')
    
        if zoom:
            self.ax.indicate_inset_zoom(self.inset_ax, edgecolor='black')
            self.inset_ax.get_legend().remove()
            self.inset_ax.set_xlabel('')
            self.inset_ax.set_ylabel('')
            self.inset_ax.set_xlabel('')
            self.inset_ax.set_ylabel('')

        if logy:
            self.ax.set_yscale('log')
            self.ax.set_ylim([1E-5,2.])

        if show:
            self.fig.show()
        
        self.save_to_pickle(path)
        self.fig.savefig(path)


class FeatureImportancePlotterKFold():
    def __init__(self, kf, features=None):
        self.kf = kf
        self.ifold = 0
        self.features = features if features else []
        self.feature_imps = {}
        self.fig, self.ax = plt.subplots(figsize=(8, 6),layout='constrained')


    def add_fold(self, model):
        self.ifold += 1

        if not self.features:
            self.features = range(len(model.feature_importances_))

        self.feature_imps[f'KFold {self.ifold}'] = model.feature_importances_


    def save_to_pickle(self, path):
        new_path = path.with_suffix('.pkl')
        with open(new_path,'wb') as pkl_file:
            pickle.dump(self.feature_imp_data, pkl_file)


    def save(self, path, show=False):
        self.feature_imp_data = {
            'features' : self.features,
            'feature_imps' : self.feature_imps,
        }

        width = 0.25
        x = np.arange(len(self.feature_imp_data['features']))
        for i, (label, vals) in enumerate(self.feature_imp_data['feature_imps'].items()):
            offset = width * i
            rects = self.ax.barh(x+offset, vals, width, label=label, align='center')
            
        self.ax.set_xlabel('Feature Importance', loc='right')
        self.ax.set_ylabel('Features', loc='top')
        self.ax.set_yticks(x + width, self.feature_imp_data['features'])
        self.ax.legend()

        if show:
            self.fig.show()

        self.save_to_pickle(path)
        self.fig.savefig(path)

class ScorePlotterKFold():
    def __init__(self, kf, features=None):
        self.kf = kf
        self.ifold = 0
        self.score_data = {}
        self.fig, self.ax = plt.subplots(figsize=(8, 6),layout='constrained')


    def add_fold(self, model, X_train, y_train, X_val, y_val):
        self.ifold += 1

        scores_train = model.predict_proba(X_train)[:,1].astype(np.float64)
        scores_val = model.predict_proba(X_val)[:,1].astype(np.float64)

        self.score_data[f'KFold {self.ifold}'] = {
            'scores_train_sig' : scores_train[y_train==1],
            'scores_train_bkg' : scores_train[y_train==0],
            'scores_val_sig' : scores_val[y_val==1],
            'scores_val_bkg' : scores_val[y_val==0],
        }


    def save_to_pickle(self, path):
        new_path = path.with_suffix('.pkl')
        with open(new_path,'wb') as pkl_file:
            pickle.dump(self.score_data, pkl_file)


    def save(self, path, show=False):
        bins = np.linspace(-5,5,40)
        bin_centers = 0.5*(bins[1:] + bins[:-1])

        scores_train_sig = np.array([])
        scores_train_bkg = np.array([])
        scores_val_sig = np.array([])
        scores_val_bkg = np.array([])

        for i, (label, vals) in enumerate(self.score_data.items()):
            scores_train_sig = np.append(scores_train_sig, vals['scores_train_sig'])
            scores_train_bkg = np.append(scores_train_bkg, vals['scores_train_bkg'])
            scores_val_sig = np.append(scores_val_sig, vals['scores_val_sig'])
            scores_val_bkg = np.append(scores_val_bkg, vals['scores_val_bkg'])

        scores_train_sig = scores_train_sig.flatten()
        scores_train_bkg = scores_train_bkg.flatten()
        scores_val_sig = scores_val_sig.flatten()
        scores_val_bkg = scores_val_bkg.flatten()

        scores_train_sig_wgts = np.abs(np.ones_like(scores_train_sig) / scores_train_sig.sum())
        scores_train_bkg_wgts = np.abs(np.ones_like(scores_train_bkg) / scores_train_bkg.sum())
        scores_val_sig_wgts = np.abs(np.ones_like(scores_val_sig) / scores_val_sig.sum())
        scores_val_bkg_wgts = np.abs(np.ones_like(scores_val_bkg) / scores_val_bkg.sum())

        train_sig_hist,_ = np.histogram(scores_train_sig, bins=bins, weights=scores_train_sig_wgts)
        train_bkg_hist,_ = np.histogram(scores_train_bkg, bins=bins, weights=scores_train_bkg_wgts)
        val_sig_hist,_ = np.histogram(scores_val_sig, bins=bins, weights=scores_val_sig_wgts)
        val_bkg_hist,_ = np.histogram(scores_val_bkg, bins=bins, weights=scores_val_bkg_wgts)

        train_sig_hist_err = np.sqrt(np.histogram(scores_train_sig, bins=bins, weights=scores_train_sig_wgts**2)[0])
        train_bkg_hist_err = np.sqrt(np.histogram(scores_train_bkg, bins=bins, weights=scores_train_bkg_wgts**2)[0])
        val_sig_hist_err = np.sqrt(np.histogram(scores_val_sig, bins=bins, weights=scores_val_sig_wgts**2)[0])
        val_bkg_hist_err = np.sqrt(np.histogram(scores_val_bkg, bins=bins, weights=scores_val_bkg_wgts**2)[0])

        self.ax.errorbar(bin_centers, train_sig_hist, yerr=train_sig_hist_err, marker = '', drawstyle = 'steps-mid', label='Train, Signal')
        self.ax.errorbar(bin_centers, train_bkg_hist, yerr=train_bkg_hist_err, marker = '', drawstyle = 'steps-mid', label='Train, Background')
        self.ax.errorbar(bin_centers, val_sig_hist, yerr=val_sig_hist_err, marker = 'o', fillstyle='none', linestyle = '', label='Validation, Signal')
        self.ax.errorbar(bin_centers, val_bkg_hist, yerr=val_bkg_hist_err, marker = 'o', fillstyle='none', linestyle = '', label='Validation, Background')
            
        self.ax.set_xlabel('BDT Score', loc='right')
        self.ax.set_ylabel('A.U.', loc='top')
        self.ax.legend()

        if show:
            self.fig.show()

        self.save_to_pickle(path)
        self.fig.savefig(path)

def read_bdt_arrays(file, tree, features, weights_branch=None, preselection=None, cutvar_branches=('Bmass', 'Mll'), n_evts=None):
    all_branches = list(set(features) | set(cutvar_branches))
    if weights_branch:
        all_branches += [weights_branch]

    with ur.open(file) as f:
        all_arrays = f[tree].arrays(all_branches, cut=preselection, entry_stop=n_evts, library=BACKEND)

    features_array = np.stack([all_arrays[k] for k in features]).T
    cutvars_dict  = {k:all_arrays[k] for k in cutvar_branches}
    weights_array  = all_arrays[weights_branch] if weights_branch else np.ones(features_array.shape[0])

    return features_array, cutvars_dict, weights_array


def save_bdt_arrays(input_file, input_tree, output_file, output_tree, output_branch_names, score_branch, scores, idxs=None, preselection=None, n_evts=None):    
    with ur.open(input_file) as f_in:
        output_branches = f_in[input_tree].arrays(output_branch_names, cut=preselection, entry_stop=n_evts, library=BACKEND)

        if idxs is not None:
            for br in output_branches.values():
                br = br[idxs]

        output_branches[score_branch] = scores
        output_branches['trigger_OR'] = np.ones_like(scores)

        with ur.recreate(output_file, compression=ur.LZMA(9)) as f_out:
            f_out[output_tree] = output_branches


def load_external_model(filepath, debug=False, model_name='model'):
    spec = importlib.util.spec_from_file_location('tmp_module', filepath)
    source_module = importlib.util.module_from_spec(spec)
    sys.modules['tmp_module'] = source_module
    spec.loader.exec_module(source_module)
    model = getattr(source_module,model_name)

    assert callable(getattr(model, 'fit')) and callable(getattr(model, 'predict_proba'))

    if debug:
        model.set_params(**{'n_estimators' : 5})

    return model

def get_branches(output_params, branch_names):
    output_branches = []
    for key in branch_names:
        if output_params.output_branches[key] is not None:
            output_branches.extend(output_params.output_branches[key])
    
    return output_branches


def save_model(output_name, model, args, formats, logger):
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


def check_rm_files(files=[]):
    for fl in files:
        if os.path.isfile(fl):
            os.system('rm '+fl)


def edit_filename(path, prefix='', suffix=''):
    path = Path(path)
    path = path.with_stem('_'.join(filter(None, [prefix, str(path.stem), suffix])))
    return path