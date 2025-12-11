import os
import sys
import logging
import pickle
import importlib.util
import numpy as np
import multiprocessing as mp
import uproot as ur
from joblib import dump, load
from glob import glob
from pathlib import Path
from sklearn.metrics import auc, RocCurveDisplay, roc_curve
from logging import DEBUG, INFO, WARNING, ERROR

from plotting_scripts.score_plotter import plot_scores
from plotting_scripts.feature_importance_plotter import plot_feature_importance
from plotting_scripts.roc_plotter import plot_roc

BACKEND = 'np'
MPL_BACKEND = 'TkAgg'

RENAME_MAPPING = {
    "BToKEE_mll_fullfit": "Mll",
    "BToKEE_fit_pt": "Bpt",
    "BToKEE_fit_mass": "Bmass",
    "BToKEE_fit_cos2D": "Bcos",
    "BToKEE_svprob": "Bprob",
    "BToKEE_fit_massErr": "BmassErr",
    "BToKEE_b_iso04": "Biso",
    "BToKEE_l_xy_sig": "BsLxy",
    "BToKEE_fit_l1_pt": "L1pt",
    "BToKEE_fit_l1_eta": "L1eta",
    "BToKEE_l1_iso04": "L1iso",
    "BToKEE_l1_PFMvaID_retrained": "L1id",
    "BToKEE_fit_l2_pt": "L2pt",
    "BToKEE_fit_l2_eta": "L2eta",
    "BToKEE_l2_iso04": "L2iso",
    "BToKEE_l2_PFMvaID_retrained": "L2id",
    "BToKEE_fit_k_pt": "Kpt",
    "BToKEE_k_iso04": "Kiso",
    "BToKEE_fit_k_eta": "Keta",
    "BToKEE_lKDz": "LKdz",
    "BToKEE_lKDr": "LKdr",
    "BToKEE_l1l2Dr": "L1L2dr",
    "BToKEE_k_svip3d": "Kip3d",
    "BToKEE_k_svip3d_err": "Kip3dErr",
    "BToKEE_l1_iso04_dca": "L1isoDca",
    "BToKEE_l2_iso04_dca": "L2isoDca",
    "BToKEE_k_iso04_dca": "KisoDca",
    "BToKEE_b_iso04_dca": "BisoDca",
    "BToKEE_k_dca_sig": "KsDca",
    "BToKEE_D0_mass_LepToPi_KToK": "KLmassD0_1",
    "BToKEE_D0_mass_LepToK_KToPi": "KLmassD0_2",
    "BToKEE_p_assymetry": "Passymetry",
    "total_weight": "total_weight",
    "PV_npvs": "Npv",
    "Presel_BDT": "presel_bdt",
    "trigger_sf_value": "trigger_sf_value",
    "trigger_sf_error": "trigger_sf_error",
    "candidate": "BToKEE"
}

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

        # Clear existing handlers to prevent duplicates if logger is re-initialized
        if self.base_logger.hasHandlers(): self.base_logger.handlers.clear()
        if self.stdout_logger.hasHandlers(): self.stdout_logger.handlers.clear()
        if self.fout_logger.hasHandlers(): self.fout_logger.handlers.clear()

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
        self.roc_data = {}
        self.tprs = []
        self.aucs = []
        self.mean_fpr = np.linspace(0, 1, 100)

    def add_fold(self, model, X, y):
        self.ifold += 1
                
        # Get scores
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]
        else:
            y_score = model.decision_function(X)
            
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)

        # Interpolate TPR
        interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        
        # Store data
        self.roc_data[f'Fold {self.ifold}'] = (fpr, tpr)
        self.tprs.append(interp_tpr)
        self.aucs.append(roc_auc)

    def agg_data(self):
        # Calculate aggregate stats before saving
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(self.aucs) # Simple mean of AUCs
        std_auc = np.std(self.aucs)
        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        # Update dict with aggregates
        self.roc_data.update({
            'mean_fpr' : self.mean_fpr,
            'mean_tpr' : mean_tpr,
            'tprs_lower' : tprs_lower,
            'tprs_upper' : tprs_upper,
            'mean_auc' : mean_auc,
            'std_auc' : std_auc,
        })

    def save_to_pickle(self, path):
        self.agg_data()
        assert self.roc_data, 'No data available'

        path = Path(path)
        new_path = path.with_suffix('.pkl')
        
        with open(new_path,'wb') as pkl_file:
            pickle.dump(self.roc_data, pkl_file)

    def save(self, path):
        self.agg_data()
        assert self.roc_data, 'No data available'

        self.save_to_pickle(path)
        plot_roc(self.roc_data, path)


class FeatureImportancePlotterKFold():
    def __init__(self, kf, features=None):
        self.kf = kf
        self.ifold = 0
        self.features = features if features else []
        self.feature_imps = {}
        self.feature_data = None

    def add_fold(self, model):
        self.ifold += 1
        if not self.features:
            self.features = [f'Feat {i}' for i in range(len(model.feature_importances_))]
        
        self.feature_imps[f'Fold {self.ifold}'] = model.feature_importances_

    def agg_data(self):
        if not self.feature_data:
            self.feature_data = {
                'features': self.features, 
                'feature_imps': self.feature_imps
            }

    def save_to_pickle(self, path):
        self.agg_data()
        assert self.feature_data, 'No data available'

        path = Path(path)
        new_path = path.with_suffix('.pkl')
        
        with open(new_path,'wb') as pkl_file:
            pickle.dump(self.feature_data, pkl_file)

    def save(self, path):
        self.agg_data()
        assert self.feature_data, 'No data available'

        self.save_to_pickle(path)
        plot_feature_importance(self.feature_data, path)


class ScorePlotterKFold():
    def __init__(self, kf):
        self.kf = kf
        self.ifold = 0
        self.score_data = {}

    def add_fold(self, model, X_train, y_train, X_val, y_val, w_train=None, w_val=None):
        self.ifold += 1
        scores_train = model.predict_proba(X_train)[:,1].astype(np.float64)
        scores_val = model.predict_proba(X_val)[:,1].astype(np.float64)

        # Handle case where weights = 1
        if w_train is None: w_train = np.ones(len(y_train))
        if w_val is None: w_val = np.ones(len(y_val))

        self.score_data[f'Fold {self.ifold}'] = {
            'scores_train_sig' : scores_train[y_train==1],
            'scores_train_bkg' : scores_train[y_train==0],
            'scores_val_sig' : scores_val[y_val==1],
            'scores_val_bkg' : scores_val[y_val==0],
            'weights_train_sig' : w_train[y_train==1],
            'weights_train_bkg' : w_train[y_train==0],
            'weights_val_sig' : w_val[y_val==1],
            'weights_val_bkg' : w_val[y_val==0],
        }

    def save_to_pickle(self, path):
        assert self.score_data, 'No data available'

        path = Path(path)
        new_path = path.with_suffix('.pkl')
        with open(new_path,'wb') as pkl_file:
            pickle.dump(self.score_data, pkl_file)

    def save(self, path, show=False):
        assert self.score_data, 'No data available'
        
        self.save_to_pickle(path)    
        plot_scores(self.score_data, path, show=show)


# --- Helper Functions ---

def save_kfold_model(model, output_dir, fold_idx, prefix='model_fold'):
    out_path = Path(output_dir) / f'{prefix}_{fold_idx}.json'
    model.save_model(out_path)


def read_bdt_arrays(file, tree, features, weights_branch=None, preselection=None, cutvar_branches=('Bmass', 'Mll'), n_evts=None):
    branch_list = list(set(features + list(cutvar_branches)))
    if weights_branch and weights_branch not in branch_list:
        branch_list.append(weights_branch)

    with ur.open(file) as f:
        # Handle tree cycle numbers
        keys = {k.split(';')[0]: k for k in f.keys()} 
        
        # Find valid tree name
        if tree in keys:
            tree_name = keys[tree] # Get exact name including cycle if needed
        elif 'Events' in keys:
            tree_name = keys['Events']
        elif 'mytree' in keys:
            tree_name = keys['mytree']
        else:
            raise KeyError(f"Tree '{tree}' not found in {file}. Available: {list(keys.keys())}")
            
        tree_obj = f[tree_name]
        
        # Allow branch aliases for different naming schemes
        available_branches = {b.decode('utf-8') if isinstance(b, bytes) else b for b in tree_obj.keys()}
        aliases = {}
        
        for orig, desired in RENAME_MAPPING.items():
            # Only create alias if desired name is MISSING and Original is PRESENT
            if desired not in available_branches and orig in available_branches:
                aliases[desired] = orig

        # Read arrays with aliases
        arrays = tree_obj.arrays(
            branch_list, 
            cut=preselection, 
            aliases=aliases,
            entry_stop=n_evts, 
            library=BACKEND
        )

    try:
        first_key = next(iter(arrays))
        n_events = len(arrays[first_key])
    except StopIteration:
        return np.empty((0, len(features)), dtype=np.float32), {k: np.empty(0) for k in cutvar_branches}, np.empty(0, dtype=np.float32)

    X = np.empty((n_events, len(features)), dtype=np.float32)
    for i, feat in enumerate(features):
        if feat not in arrays:
            raise KeyError(f"Feature '{feat}' missing in {file} (checked aliases and formulas)")
        X[:, i] = np.asarray(arrays[feat], dtype=np.float32)
    X = np.ascontiguousarray(X)

    cutvars_dict = {k: np.asarray(arrays[k]) for k in cutvar_branches}
    
    if weights_branch and weights_branch in arrays:
        weights_array = np.asarray(arrays[weights_branch], dtype=np.float32)
    else:
        weights_array = np.ones(n_events, dtype=np.float32)

    return X, cutvars_dict, weights_array


def save_bdt_arrays(input_file, input_tree, output_file, output_tree, output_branch_names, score_branch, scores, idxs=None, preselection=None, n_evts=None):    
    with ur.open(input_file) as f_in:
        # Handle tree cycle numbers
        keys = {k.split(';')[0]: k for k in f_in.keys()}

        # Find valid tree name        
        if input_tree in keys:
            valid_tree_name = keys[input_tree]
        elif 'Events' in keys:
            valid_tree_name = keys['Events']
        elif 'mytree' in keys:
            valid_tree_name = keys['mytree']
        else:
            raise KeyError(f"Tree '{input_tree}' not found in {input_file}. Available: {list(keys.keys())}")
            
        tree_obj = f_in[valid_tree_name]

        available_branches = {b.decode('utf-8') if isinstance(b, bytes) else b for b in tree_obj.keys()}
        aliases = {}
        for orig, desired in RENAME_MAPPING.items():
            if desired not in available_branches and orig in available_branches:
                aliases[desired] = orig

        output_branches = tree_obj.arrays(
            output_branch_names, 
            cut=preselection, 
            aliases=aliases,
            entry_stop=n_evts, 
            library=BACKEND
        )

    if idxs is not None:
        for k, v in list(output_branches.items()):
            output_branches[k] = np.asarray(v)[idxs]

    scores = np.asarray(scores)
    output_branches[score_branch] = scores

    if 'trigger_OR' not in output_branches:
        output_branches['trigger_OR'] = np.ones(scores.shape[0], dtype=np.uint8)

    with ur.recreate(output_file, compression=ur.LZMA(1)) as f_out:
        f_out[output_tree] = output_branches


def load_external_model(filepath, debug=False, model_name='model'):
    p = Path(filepath)
    if p.suffix in ('.pkl', '.joblib'):
        model = load(str(p))
    elif p.suffix in ('.json', '.model', '.bin', '.txt'):
        try:
            from xgboost import Booster as XGBBooster
            b = XGBBooster()
            b.load_model(str(p))
            return b
        except Exception:
            model = load(str(p)) if p.exists() else None
    elif p.suffix == '.py':
        spec = importlib.util.spec_from_file_location('tmp_module', filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules['tmp_module'] = module
        spec.loader.exec_module(module)
        if hasattr(module, model_name):
            model = getattr(module, model_name)
        elif hasattr(module, 'build_model'):
            model = module.build_model()
        else:
            raise ImportError(f"No attribute '{model_name}' in {filepath}")
    else:
        try:
            model = load(str(p))
        except Exception as e:
            raise ImportError(f"Could not load model {filepath}: {e}")

    if hasattr(model, 'predict_proba'):
        if debug and hasattr(model, 'set_params'):
            try:
                model.set_params(n_estimators=5)
            except Exception:
                pass
        return model

    # Try returning XGBoost Booster if not sklearn-wrapped
    try:
        from xgboost import Booster as XGBBooster
        if isinstance(model, XGBBooster): return model
    except Exception:
        pass
        
    raise AssertionError('Loaded object does not expose predict_proba and is not an XGBoost Booster')


def get_branches(output_params, branch_names):
    output_branches = []
    for key in branch_names:
        vals = output_params.output_branches.get(key)
        if not vals: continue
        for v in vals:
            if v not in output_branches: output_branches.append(v)
    return output_branches


def edit_filename(path, prefix='', suffix=''):
    path = Path(path)
    new_stem = '_'.join(filter(None, [prefix, str(path.stem), suffix]))
    try:
        return path.with_stem(new_stem)
    except AttributeError:
        return path.with_name(new_stem + path.suffix)