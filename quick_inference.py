import argparse
import time
import gc
import logging
import sys
import re
from pathlib import Path
from typing import Optional, Any, Dict, List, Set

import numpy as np
import uproot as ur

from utils import read_bdt_arrays, get_branches

# Setup Logging
logger = logging.getLogger('QuickInference')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')

# Branch name mapping works with regular inputs and preprocessed ones
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
    "event": "event",
    "PV_npvs": "Npv",
    "Presel_BDT": "presel_bdt",
    "trigger_sf_value": "trigger_sf_value",
    "trigger_sf_error": "trigger_sf_error",
}


def predict_in_chunks(model: Any, X: np.ndarray, batch_size: int = 100_000) -> np.ndarray:
    n_samples = X.shape[0]
    scores = np.empty(n_samples, dtype=np.float64)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        scores[start:end] = model.predict_proba(X[start:end])[:, 1].astype(np.float64)
    
    return scores


def resolve_branch_name(desired_name: str, available_keys: Set[str], rev_mapping: Dict[str, str]) -> Optional[str]:
    if desired_name in available_keys:
        return desired_name
    
    orig_name = rev_mapping.get(desired_name)
    if orig_name and orig_name in available_keys:
        return orig_name
        
    return None


def resolve_formula(expression: str, available_keys: Set[str], rev_mapping: Dict[str, str]) -> Optional[str]:
    tokens = re.split(r'([^\w\.]+)', expression)
    
    resolved_tokens = []
    found_any = False
    
    for token in tokens:
        if not token.strip():
            resolved_tokens.append(token)
            continue
            
        phys_name = resolve_branch_name(token, available_keys, rev_mapping)
        
        if phys_name:
            resolved_tokens.append(phys_name)
            found_any = True
        else:
            resolved_tokens.append(token)
            
    if not found_any:
        return None
        
    return "".join(resolved_tokens)


def run_pipeline(args, config: dict):
    infile = Path(args.infile)
    outdir = Path('outputs/tmp/' if args.debug else args.outpath)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Branch mapping
    name_mapping = RENAME_MAPPING
    rev_mapping = {v: k for k, v in name_mapping.items()}

    # Detect old or new tree name
    input_tree_name = None
    available_keys = set() 
    try:
        with ur.open(infile) as f_check:
            keys = [k.split(';')[0] for k in f_check.keys()] 
            
            if 'Events' in keys:
                input_tree_name = 'Events'
            elif 'mytree' in keys:
                input_tree_name = 'mytree'
            else:
                available = ", ".join(keys)
                raise ValueError(f"Could not find 'Events' or 'mytree' in {infile}. Found: {available}")
            
            tree_obj = f_check[input_tree_name]
            available_keys = {k.decode('utf-8') if isinstance(k, bytes) else k for k in tree_obj.keys()}
    except Exception as e:
        logger.error(f"Failed to inspect file: {e}")
        raise
    logger.info(f"Using input tree: '{input_tree_name}'")


    # Run inference
    scores = None
    if not args.just_rename:
        logger.info(f"Mode: Inference + Rename. Model: {args.model}")
        
        # Resolve features
        actual_features_to_read = []
        missing_features = []
        for feat in config['features']:
            found_name = resolve_branch_name(feat, available_keys, rev_mapping)
            if found_name:
                actual_features_to_read.append(found_name)
            else:
                # Try Formula Parsing
                formula = resolve_formula(feat, available_keys, rev_mapping)
                if formula:
                    logger.debug(f"Resolved formula: {feat} -> {formula}")
                    actual_features_to_read.append(formula)
                else:
                    missing_features.append(feat)
        if missing_features:
            raise RuntimeError(f"Could not find branches for features: {missing_features}.")

        # Resolve cuts
        b_mass_var = resolve_branch_name(config['b_mass_branch'], available_keys, rev_mapping)
        if not b_mass_var: b_mass_var = config['b_mass_branch'] # Fallback if not found (will error in uproot)

        ll_mass_var = resolve_branch_name(config['ll_mass_branch'], available_keys, rev_mapping)
        if not ll_mass_var: ll_mass_var = config['ll_mass_branch']

        # Resolve weights
        weights_var = None
        if args.isMC:
            weights_var = resolve_branch_name(config['weights_branch'], available_keys, rev_mapping)
            if not weights_var: weights_var = config['weights_branch']

        logger.info(f"Reading {len(actual_features_to_read)} features...")
        
        # Load data arrays
        X, _, _ = read_bdt_arrays(
            infile,
            input_tree_name,
            actual_features_to_read, 
            weights_var,  # Pass resolved weights name
            args.cut,
            (b_mass_var, ll_mass_var), # Pass resolved cut variables
            n_evts=10000 if args.debug else None
        )

        if X is None or X.size == 0:
            logger.warning("No events found passing cuts. Exiting.")
            return

        # Load model
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(str(args.model))
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Run inference
        logger.info(f"Running inference on {X.shape[0]} events...")
        start_t = time.perf_counter()
        scores = predict_in_chunks(model, X, batch_size=args.batch_size)
        logger.info(f"Inference complete in {time.perf_counter() - start_t:.2f}s")
        
        del X, model
        gc.collect()

    else:
        logger.info("Mode: Rename Only (Skipping Inference)")

    
    # Define which branches to pass through
    desired_branches = get_branches(argparse.Namespace(**config), ['common', 'mc' if args.isMC else 'data'])
    
    if name_mapping:
        desired_branches.extend(name_mapping.values())
        desired_branches = list(set(desired_branches)) 

    # Map branch names
    branches_to_read = []
    for b in desired_branches:
        found_name = resolve_branch_name(b, available_keys, rev_mapping)
        if found_name:
            branches_to_read.append(found_name)

    logger.info(f"Reading {len(branches_to_read)} output branches...")
    
    with ur.open(infile) as f_in:
        tree = f_in[input_tree_name]
        out_data = tree.arrays(
            branches_to_read, 
            cut=args.cut, 
            entry_stop=10000 if args.debug else None, 
            library='np'
        )

    # Apply final renaming
    final_data = {}
    for actual_name, arr in out_data.items():
        if actual_name in name_mapping:
            desired_name = name_mapping[actual_name]
        else:
            desired_name = actual_name
            
        final_data[desired_name] = arr

    # Add scores
    if scores is not None:
        final_data[config['score_branch']] = scores

    # Add trigger_OR
    if 'trigger_OR' not in final_data:
        n_rows = len(next(iter(final_data.values()))) if final_data else 0
        final_data['trigger_OR'] = np.ones(n_rows, dtype=np.uint8)

    # Write to file
    outfile = outdir / f'{infile.stem}{"_"+args.label if args.label else ""}{infile.suffix}'
    output_tree_name = args.out_tree if args.out_tree else input_tree_name
    
    logger.info(f"Saving to {outfile} (Tree: {output_tree_name})...")
    with ur.recreate(outfile, compression=ur.LZMA(1)) as f_out:
        f_out[output_tree_name] = final_data

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Lightweight BDT Inference & Renaming Tool")
    
    parser.add_argument('-i', '--infile', type=Path, required=True, help='Input ROOT file')
    parser.add_argument('-o', '--outpath', type=Path, default=Path('outputs/'), help='Output directory')
    parser.add_argument('-m', '--model', type=Path, help='XGBoost model file')
    parser.add_argument('-l', '--label', type=str, default='_wScores', help='Suffix for output filename')
    parser.add_argument('--out-tree', type=str, default='mytree', help='Output tree name (default: "mytree")')
    parser.add_argument('--cut', type=str, help='Preselection cut string')
    parser.add_argument('-mc', '--isMC', action='store_true', help='Process as MC')
    parser.add_argument('--just-rename', action='store_true', help='Skip inference, only rename branches')
    parser.add_argument('--debug', action='store_true', help='Run on small subset')
    parser.add_argument('--batch-size', type=int, default=100_000, help='Inference batch size')

    args = parser.parse_args()

    config = {
        'tree_name'       : 'Events', 
        'b_mass_branch'   : 'Bmass',
        'll_mass_branch'  : 'Mll',
        'score_branch'    : 'bdt_score',
        'features'        : [
            'Bprob', 'BsLxy', 'L2iso/L2pt', 'Bcos', 'Kiso/Kpt', 
            'LKdz', 'LKdr', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id'
        ],
        'weights_branch'  : 'total_weight',
        'output_branches' : {
            'common': ['event', 'Bmass', 'Mll'],
            'data': [],
            'mc': ['total_weight'],
        },
    }

    if args.just_rename and args.label == '_wScores':
        args.label = '_wRenames'
        
    if not args.just_rename and not args.model:
        parser.error("--model is required unless --just-rename is specified.")

    run_pipeline(args, config)


if __name__ == '__main__':
    main()