import os
from pathlib import Path
import time
import argparse
import yaml
import json
import gc
from array import array

import numpy as np
from sklearn.model_selection import train_test_split

from utils import (
    R3KLogger,
    read_bdt_arrays,
    save_bdt_arrays,
    load_external_model,
    edit_filename,
    get_branches,
)

rand_seed = 271996


def as_weights(weights, n):
    if weights is None:
        return np.ones(n, dtype=np.float64)
    return np.asarray(weights, dtype=np.float64)


def make_signal_validation_holdout(n_total, signal_mask, validation_frac, rng):
    """
    Randomly hold out validation_frac of the signal MC before k-fold splitting

    - Only events passing signal_mask are eligible for the validation holdout
    - The validation events are never used in any training fold
    """
    signal_idx = np.where(signal_mask)[0]
    if signal_idx.size < 2:
        raise ValueError(f"Need at least 2 signal MC events, got {signal_idx.size}")

    shuffled = signal_idx.copy()
    rng.shuffle(shuffled)

    n_val = int(round(validation_frac * signal_idx.size))
    n_val = max(1, n_val)
    n_val = min(n_val, signal_idx.size - 1)

    validation_idx = np.sort(shuffled[:n_val])
    train_idx = np.sort(shuffled[n_val:])

    validation_mask = np.zeros(n_total, dtype=bool)
    train_mask = np.zeros(n_total, dtype=bool)
    validation_mask[validation_idx] = True
    train_mask[train_idx] = True

    return train_mask, validation_mask, train_idx, validation_idx


def make_fold_ids_for_training_subset(n_total, training_subset_mask, n_folds, rng):
    """
    Assign fold IDs 0..n_folds-1 only to the training subset

    Events outside the training subset get fold ID -1. This avoids accidentally
    treating non-sideband data, non-signal MC, or the initial 10 percent MC
    validation holdout as part of the OOF training folds
    """
    fold_ids = np.full(n_total, -1, dtype=np.int32)
    subset_idx = np.where(training_subset_mask)[0]

    if subset_idx.size < n_folds:
        raise ValueError(
            f"Need at least n_folds={n_folds} events in the training subset, "
            f"got {subset_idx.size}"
        )

    shuffled = subset_idx.copy()
    rng.shuffle(shuffled)
    chunks = np.array_split(shuffled, n_folds)

    for fold, idxs in enumerate(chunks):
        fold_ids[idxs] = fold

    return fold_ids


def log_fold_content(name, fold_ids, training_subset_mask, lgr, n_folds):
    """Keeps track of statistics within each fold"""
    lgr.log(f"{name} k-fold composition", just_print=True)
    for fold in range(n_folds):
        subset = int(np.sum(training_subset_mask & (fold_ids == fold)))
        lgr.log(f"  fold {fold}: training_subset={subset}", just_print=True)

    outside = int(np.sum(fold_ids < 0))
    lgr.log(f"  outside k-fold training subset: {outside}", just_print=True)


def get_effective_n_trees(model):
    """ Determine how many trees each model actually used """
    best_iter = None
    try:
        best_iter = model.best_iteration
    except Exception:
        best_iter = None

    if best_iter is not None:
        return int(best_iter) + 1

    best_ntree_limit = None
    try:
        best_ntree_limit = model.best_ntree_limit
    except Exception:
        best_ntree_limit = None

    if best_ntree_limit is not None and int(best_ntree_limit) > 0:
        return int(best_ntree_limit)

    n_estimators = model.get_params().get("n_estimators", None)
    if n_estimators is None:
        return None
    return int(n_estimators)


def summarize_tree_counts(name, counts, lgr):
    """Keeps track of tree counts across different folds"""
    if len(counts) == 0:
        lgr.log(f"{name}: no tree counts recorded", just_print=True)
        return

    counts_arr = np.asarray(counts, dtype=float)
    lgr.log(
        (
            f"{name}: n={len(counts)}, "
            f"counts={list(map(int, counts))}, "
            f"mean={np.mean(counts_arr):.2f}, "
            f"median={np.median(counts_arr):.2f}, "
            f"min={np.min(counts_arr):.0f}, "
            f"max={np.max(counts_arr):.0f}"
        ),
        just_print=True,
    )


def model_predict_scores(model, X):
    """
    Score prediction for fitted XGBClassifier models
    - Used to score signal MC and sideband data 
    """
    if X is None or X.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)

    booster = model.get_booster()
    scores = booster.inplace_predict(X)
    scores = np.asarray(scores, dtype=np.float64)

    if scores.ndim > 1:
        scores = scores[:, -1]

    return scores.reshape(-1)


def predict_score_matrix(models, X):
    """
    Scores the same events with multiple models, storing the scores in a maxtrix
    - E.g.: For 10 models scoring N events, builds an N x 10 matrix

    Used for scoring the remaining samples with K-fold model ensemble average
    """
    if X is None or X.shape[0] == 0:
        return np.empty((0, len(models)), dtype=np.float64)

    cols = [model_predict_scores(model, X) for model in models]
    return np.column_stack(cols)


def append_extra_branches_to_root(root_path, tree_name, branch_arrays, lgr=None):
    """
    Append extra branches to an existing ROOT TTree in UPDATE mode

    If UPDATE on the target file fails, save a .npz file instead so the
    extra information is not lost.
    """
    try:
        import ROOT

        f = ROOT.TFile.Open(str(root_path), "UPDATE")
        if not f or f.IsZombie():
            raise RuntimeError(f"Could not open ROOT file for UPDATE: {root_path}")

        tree = f.Get(tree_name)
        if tree is None:
            f.Close()
            raise RuntimeError(f"Could not find tree '{tree_name}' in {root_path}")

        n_entries = tree.GetEntries()

        for name, arr in branch_arrays.items():
            if len(arr) != n_entries:
                f.Close()
                raise ValueError(
                    f"Branch {name} has length {len(arr)} but tree has {n_entries} entries"
                )

        existing = {br.GetName() for br in tree.GetListOfBranches()}
        for name in branch_arrays.keys():
            if name in existing:
                f.Close()
                raise RuntimeError(f"Branch '{name}' already exists in {root_path}")

        buffers = {}
        branches = {}

        for name, arr in branch_arrays.items():
            arr_np = np.asarray(arr)

            if np.issubdtype(arr_np.dtype, np.floating):
                buf = array("f", [0.0])
                br = tree.Branch(name, buf, f"{name}/F")
            elif np.issubdtype(arr_np.dtype, np.integer):
                buf = array("i", [0])
                br = tree.Branch(name, buf, f"{name}/I")
            else:
                f.Close()
                raise TypeError(f"Unsupported dtype for branch '{name}': {arr_np.dtype}")

            buffers[name] = buf
            branches[name] = br

        dtype_map = {name: np.asarray(arr).dtype for name, arr in branch_arrays.items()}

        for i in range(n_entries):
            tree.GetEntry(i)
            for name, arr in branch_arrays.items():
                val = arr[i]
                if np.issubdtype(dtype_map[name], np.floating):
                    buffers[name][0] = float(val)
                else:
                    buffers[name][0] = int(val)
                branches[name].Fill()

        tree.Write("", ROOT.TObject.kOverwrite)
        f.Close()

    except Exception as e:
        npz_file = Path(root_path).with_suffix(Path(root_path).suffix + ".extra_branches.npz")
        np.savez_compressed(npz_file, **branch_arrays)
        msg = (
            f"WARNING: could not append branches directly to {root_path}. "
            f"Saved .npz instead: {npz_file}. Reason: {e}"
        )
        if lgr is not None:
            lgr.log(msg, just_print=True)
        else:
            print(msg)


def save_manifest(
    output_dir,
    n_folds,
    mc_validation_frac,
    debug_n_evts,
    data_fold_ids,
    mc_fold_ids,
    mask_bkg,
    mask_sig_all,
    mc_train_signal_mask,
    mc_validation_mask,
    tree_counts,
):
    """
    Stores various metadata into json and npz files
    - Keeps track of statistics in different data groups
    - Useful for plotting stage involving fold composition
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_folds = list(range(n_folds))
    payload = {
        "workflow": "full_sideband_mc_signal_holdout_kfold",
        "training_design": "signal_holdout_then_paired_sideband_and_signal_oof_kfold",
        "n_folds": int(n_folds),
        "train_folds": [int(x) for x in train_folds],
        "mc_validation_frac": float(mc_validation_frac),
        "debug_n_evts": None if debug_n_evts is None else int(debug_n_evts),
        "effective_tree_counts": {
            str(k): (None if v is None else int(v)) for k, v in tree_counts.items()
        },
        "data_sideband_training_subset_total": int(np.sum(mask_bkg)),
        "mc_signal_total": int(np.sum(mask_sig_all)),
        "mc_signal_training_pool_after_holdout": int(np.sum(mc_train_signal_mask)),
        "mc_signal_validation_holdout": int(np.sum(mc_validation_mask)),
        "data_folds": {},
        "mc_folds": {},
    }

    for fold in train_folds:
        payload["data_folds"][str(fold)] = {
            "sideband_training_subset": int(np.sum(mask_bkg & (data_fold_ids == fold)))
        }
        payload["mc_folds"][str(fold)] = {
            "signal_training_subset": int(np.sum(mc_train_signal_mask & (mc_fold_ids == fold)))
        }

    with open(output_dir / "holdout_manifest.json", "w") as f:
        json.dump(payload, f, indent=2)

    np.savez_compressed(
        output_dir / "holdout_split_indices.npz",
        mc_validation_idx=np.where(mc_validation_mask)[0].astype(np.int64),
        mc_train_pool_idx=np.where(mc_train_signal_mask)[0].astype(np.int64),
        mc_signal_all_idx=np.where(mask_sig_all)[0].astype(np.int64),
        data_sideband_idx=np.where(mask_bkg)[0].astype(np.int64),
        mc_train_fold_ids=mc_fold_ids.astype(np.int32),
        data_fold_ids=data_fold_ids.astype(np.int32),
    )


def fit_one_fold_model(
    fold,
    X_mc,
    X_data,
    weights_mc,
    weights_data,
    sig_train_mask,
    bkg_train_mask,
    model_params,
    args,
):
    """Trains one fold model"""

    # Pick out signal and background training data
    X_sig_train = X_mc[sig_train_mask]
    X_bkg_train = X_data[bkg_train_mask]

    if X_sig_train.shape[0] == 0 or X_bkg_train.shape[0] == 0:
        raise ValueError(
            f"Fold {fold} has empty training class: "
            f"train_sig={X_sig_train.shape[0]}, train_bkg={X_bkg_train.shape[0]}"
        )

    # Signal is labeled 1, background is labeled 0
    y_sig_train = np.ones(X_sig_train.shape[0], dtype=np.float64)
    y_bkg_train = np.zeros(X_bkg_train.shape[0], dtype=np.float64)

    w_sig_train = weights_mc[sig_train_mask]
    w_bkg_train = weights_data[bkg_train_mask]

    # Combine signal MC and sideband data for training
    combined_X = np.concatenate((X_sig_train, X_bkg_train), axis=0)
    combined_y = np.concatenate((y_sig_train, y_bkg_train), axis=0)
    combined_w = np.concatenate((w_sig_train, w_bkg_train), axis=0)

    # Define early stoppage validation split
    split = train_test_split(
        combined_X,
        combined_y,
        combined_w,
        test_size=args.early_stop_validation_frac,
        random_state=rand_seed,
        stratify=combined_y,
    )

    X_train, X_val, y_train, y_val, weights_train, weights_val = split

    # Rescale signal weight to match total background weight
    # This ensures no class dominates a given model
    sum_w_sig_train = np.sum(weights_train[y_train == 1])
    sum_w_bkg_train = np.sum(weights_train[y_train == 0])
    train_sf = sum_w_bkg_train / sum_w_sig_train if sum_w_sig_train > 0 else 1.0
    weights_train[y_train == 1] *= train_sf

    sum_w_sig_val = np.sum(weights_val[y_val == 1])
    sum_w_bkg_val = np.sum(weights_val[y_val == 0])
    val_sf = sum_w_bkg_val / sum_w_sig_val if sum_w_sig_val > 0 else 1.0
    weights_val[y_val == 1] *= val_sf

    assert np.all(weights_train[y_train == 0] == 1.0)
    # Clip large MC weights so abnormally large weights don't dominate
    max_w_sig = np.percentile(weights_train[y_train == 1], 99) * 2
    weights_train[y_train == 1] = np.clip(weights_train[y_train == 1], 0, max_w_sig)
    weights_val[y_val == 1] = np.clip(weights_val[y_val == 1], 0, max_w_sig)

    eval_set = ((X_train, y_train), (X_val, y_val))

    # Load model template from config and perform fitting
    fold_model = load_external_model(model_params.template_file, debug=args.debug)
    fold_model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        sample_weight=weights_train,
        sample_weight_eval_set=[weights_train, weights_val],
        verbose=100 if args.verbose else 0,
    )

    return fold_model, X_sig_train.shape[0], X_bkg_train.shape[0]


def score_with_ensemble(models, X):
    """Perform scoring with K-fold model score averaging"""
    if X is None or X.shape[0] == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    score_matrix = predict_score_matrix(models, X)
    avg_scores = np.mean(score_matrix, axis=1).astype(np.float64)
    std_scores = np.std(score_matrix, axis=1).astype(np.float64)
    return avg_scores, std_scores

"""
The BDT workflow runs as follows:
K-Fold Evaluation (for signal MC and sideband data only)
1.) Split signal MC (lowq2 B-->Kee) into 10% validation, 90% training samples
2.) Use all sideband data for training (Bmass sideband, lowq2 Mll)
3.) Split the sideband data and 90% of signal MC into K-folds each
4.) Pair folds: paired fold i = sideband data fold i + signal MC fold i
5.) For each model i, train on all paired folds EXCEPT i
6.) Out-of-fold (OOF) score the paired fold i, and repeat for remaining paired folds
- NOTE: the held out 10% of signal MC gets scored by the model averages, as do the remaining samples
        discussed next

Scoring Remaining Samples (everthing that's NOT sideband data and signal MC)
1.) Use each K-fold model to evaluate the sample, giving K scores
2.) The final score for each remaining sample event is the average over the K model scores
"""
def bdt_inference(dataset_params, model_params, output_params, args):
    debug_n_evts = 100000 if args.debug else None
    n_folds = int(args.n_folds)
    mc_validation_frac = float(args.mc_validation_frac)

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if not (0.0 < mc_validation_frac < 1.0):
        raise ValueError("mc_validation_frac must be between 0 and 1")
    if not (0.0 < float(args.early_stop_validation_frac) < 1.0):
        raise ValueError("early_stop_validation_frac must be between 0 and 1")

    os.makedirs(output_params.output_dir, exist_ok=True)
    os.makedirs(output_params.output_dir / "models", exist_ok=True)
    base_filename = output_params.output_dir / "measurement.root"

    lgr = R3KLogger(
        output_params.output_dir / output_params.log_file,
        verbose=args.verbose,
        append=False,
    )

    with open(args.config, "r") as f:
        cfg_str = f"cfg_path: {args.config}\n{f.read()}"

    lgr.log(f"Configuration File:\n{cfg_str}\n")
    lgr.log(
        (
            "Using paired-fold workflow: "
            f"hold out {mc_validation_frac:.3f} of signal MC first, "
            f"then train {n_folds} OOF paired sideband/signal folds"
        ),
        just_print=True,
    )

    # ------------------------------------------------------------------
    # Load rare MC and data samples:
    # - X_mc, X_data: consist of BDT input features
    # - cutvars_mc, cutvars_data: used to make sig and bkg masks
    # - weights_mc: mc event weights
    # - weights_data: data weights are set to 1
    # ------------------------------------------------------------------
    X_mc, cutvars_mc, weights_mc = read_bdt_arrays(
        dataset_params.rare_file,
        dataset_params.tree_name,
        model_params.features,
        model_params.sample_weights,
        model_params.preselection,
        (dataset_params.b_mass_branch, dataset_params.ll_mass_branch),
        n_evts=debug_n_evts,
    )

    X_data, cutvars_data, weights_data = read_bdt_arrays(
        dataset_params.data_file,
        dataset_params.tree_name,
        model_params.features,
        None,
        model_params.preselection,
        (dataset_params.b_mass_branch, dataset_params.ll_mass_branch),
        n_evts=debug_n_evts,
    )

    # Create numeric weight arrays for data and mc
    weights_mc = as_weights(weights_mc, X_mc.shape[0])
    weights_data = as_weights(weights_data, X_data.shape[0])

    # ------------------------------------------------------------------
    # Define signal MC and sideband data training subsets
    # ------------------------------------------------------------------
    mask_sig_all = np.logical_and(
        cutvars_mc[dataset_params.ll_mass_branch] > 1.05,
        cutvars_mc[dataset_params.ll_mass_branch] < 2.45,
    )

    mask_bkg_lowq2 = np.logical_and(
        cutvars_data[dataset_params.ll_mass_branch] > 1.05,
        cutvars_data[dataset_params.ll_mass_branch] < 2.45,
    )

    mask_bkg_sideband = np.logical_or(
        np.logical_and(
            cutvars_data[dataset_params.b_mass_branch] > 4.8,
            cutvars_data[dataset_params.b_mass_branch] < 5.0,
        ),
        np.logical_and(
            cutvars_data[dataset_params.b_mass_branch] > 5.4,
            cutvars_data[dataset_params.b_mass_branch] < 5.6,
        ),
    )

    mask_bkg = np.logical_and(mask_bkg_lowq2, mask_bkg_sideband)

    if np.sum(mask_bkg) < n_folds:
        raise ValueError(
            f"Need at least n_folds={n_folds} sideband data events, got {int(np.sum(mask_bkg))}"
        )

    rng = np.random.default_rng(rand_seed)

    # First remove 10 percent of signal MC for validation plots
    # This holdout is not included in any OOF fold or training fold
    mc_train_signal_mask, mc_validation_mask, _, _ = make_signal_validation_holdout(
        X_mc.shape[0],
        mask_sig_all,
        mc_validation_frac,
        rng,
    )

    # Split the full sideband data and the remaining 90 percent signal MC
    # into k separately balanced folds, then pair fold i with fold i.
    data_fold_ids = make_fold_ids_for_training_subset(X_data.shape[0], mask_bkg, n_folds, rng)
    mc_fold_ids = make_fold_ids_for_training_subset(X_mc.shape[0], mc_train_signal_mask, n_folds, rng)

    lgr.log("Training subset counts", just_print=True)
    lgr.log(f"  sideband data used for OOF training: {int(np.sum(mask_bkg))}", just_print=True)
    lgr.log(f"  all low-q2 signal MC: {int(np.sum(mask_sig_all))}", just_print=True)
    lgr.log(f"  signal MC validation holdout: {int(np.sum(mc_validation_mask))}", just_print=True)
    lgr.log(f"  signal MC used for OOF training: {int(np.sum(mc_train_signal_mask))}", just_print=True)
    log_fold_content("data sideband", data_fold_ids, mask_bkg, lgr, n_folds)
    log_fold_content("signal MC training pool", mc_fold_ids, mc_train_signal_mask, lgr, n_folds)

    # Initialize score arrays for data, mc, and averages
    data_scores = np.full(X_data.shape[0], np.nan, dtype=np.float64)
    mc_scores = np.full(X_mc.shape[0], np.nan, dtype=np.float64)

    data_ensemble_std = np.full(X_data.shape[0], np.nan, dtype=np.float32)
    mc_ensemble_std = np.full(X_mc.shape[0], np.nan, dtype=np.float32)

    fold_models = {}
    tree_counts = {}

    lgr.log(f"Training {n_folds} paired OOF fold models", just_print=True)
    start = time.perf_counter()

    # ------------------------------------------------------------------
    # Train k models. For model i, hold out paired fold i from both classes.
    # Score that paired fold with the model that did not train on it.
    # ------------------------------------------------------------------
    for fold in range(n_folds):
        lgr.log(f"Starting paired OOF fold-model {fold}", just_print=True)

        # Pick out signal MC and sideband data from OOF samples
        sig_train_mask = mc_train_signal_mask & (mc_fold_ids != fold)
        bkg_train_mask = mask_bkg & (data_fold_ids != fold)

        # Find samples of held-out-fold to be scored this iteration
        data_test_idx = np.where(mask_bkg & (data_fold_ids == fold))[0]
        mc_test_idx = np.where(mc_train_signal_mask & (mc_fold_ids == fold))[0]

        # Training step
        fold_model, n_sig_train, n_bkg_train = fit_one_fold_model(
            fold=fold,
            X_mc=X_mc,
            X_data=X_data,
            weights_mc=weights_mc,
            weights_data=weights_data,
            sig_train_mask=sig_train_mask,
            bkg_train_mask=bkg_train_mask,
            model_params=model_params,
            args=args,
        )
        
        # Compute ntrees for this K-fold model (w/ early stoppage)
        n_trees_fold = get_effective_n_trees(fold_model)
        tree_counts[str(fold)] = None if n_trees_fold is None else int(n_trees_fold)

        # OOF score the paired fold: sideband-data fold i and signal-MC fold i.
        data_scores[data_test_idx] = model_predict_scores(fold_model, X_data[data_test_idx])
        mc_scores[mc_test_idx] = model_predict_scores(fold_model, X_mc[mc_test_idx])

        # Save model and store relevant information
        model_path = output_params.output_dir / "models" / f"model_fold_{fold}.json"
        fold_model.save_model(model_path)
        fold_models[fold] = fold_model

        lgr.log(
            (
                f"Finished paired OOF fold-model {fold}: "
                f"train_sig={n_sig_train}, train_bkg={n_bkg_train}, "
                f"oof_signal={mc_test_idx.size}, oof_sideband={data_test_idx.size}, "
                f"effective_n_trees={n_trees_fold}"
            ),
            just_print=True,
        )

        del sig_train_mask, bkg_train_mask, data_test_idx, mc_test_idx
        gc.collect()

    elapsed = round(time.perf_counter() - start)
    lgr.log(f"Elapsed Training/Inference Time for paired OOF models = {elapsed}s", just_print=True)
    summarize_tree_counts("Paired OOF fold tree counts", [c for c in tree_counts.values() if c is not None], lgr)

    all_models = [fold_models[f] for f in range(n_folds)]

    # ------------------------------------------------------------------
    # Score anything not covered by OOF with the average of all k models.
    #
    # Data:
    #   sideband data gets OOF scores.
    #   non-sideband data gets the ensemble average.
    #
    # Rare MC:
    #   90 percent signal training pool gets OOF scores.
    #   initial 10 percent signal validation holdout gets ensemble scores.
    #   non-signal MC gets ensemble scores.
    # ------------------------------------------------------------------
    data_remaining_idx = np.where(~mask_bkg)[0]  # Locate non-sideband-data samples
    # Score remaining (non-sideband) data with K-fold model averages
    if data_remaining_idx.size > 0:
        avg, std = score_with_ensemble(all_models, X_data[data_remaining_idx])
        data_scores[data_remaining_idx] = avg
        data_ensemble_std[data_remaining_idx] = std.astype(np.float32)
        del avg, std
        gc.collect()

    # Score held-out 10% of signal MC with K-fold model averages 
    # This is done for unbiased validation plot purposes
    mc_validation_idx = np.where(mc_validation_mask)[0]
    if mc_validation_idx.size > 0:
        avg, std = score_with_ensemble(all_models, X_mc[mc_validation_idx])
        mc_scores[mc_validation_idx] = avg
        mc_ensemble_std[mc_validation_idx] = std.astype(np.float32)
        del avg, std
        gc.collect()

    # Score remaining (non-signal) MC with K-fold model averages
    mc_remaining_idx = np.where(~mask_sig_all)[0]
    if mc_remaining_idx.size > 0:
        avg, std = score_with_ensemble(all_models, X_mc[mc_remaining_idx])
        mc_scores[mc_remaining_idx] = avg
        mc_ensemble_std[mc_remaining_idx] = std.astype(np.float32)
        del avg, std
        gc.collect()

    # Check to make sure scores are sensible
    if not np.all(np.isfinite(data_scores)):
        n_bad = int(np.sum(~np.isfinite(data_scores)))
        raise RuntimeError(f"data_scores still has {n_bad} non-finite entries")
    if not np.all(np.isfinite(mc_scores)):
        n_bad = int(np.sum(~np.isfinite(mc_scores)))
        raise RuntimeError(f"mc_scores still has {n_bad} non-finite entries")

    lgr.log(
        (
            f"Scored sideband data and 90 percent signal MC with OOF models. "
            f"Scored non-sideband data, non-signal MC, and the 10 percent signal validation holdout "
            f"with the average of all {len(all_models)} models."
        ),
        just_print=True,
    )

    # Save split information for plotting
    save_manifest(
        output_params.output_dir / "models",
        n_folds,
        mc_validation_frac,
        debug_n_evts,
        data_fold_ids,
        mc_fold_ids,
        mask_bkg,
        mask_sig_all,
        mc_train_signal_mask,
        mc_validation_mask,
        tree_counts,
    )

    # ------------------------------------------------------------------
    # Save nominal data and rare-MC outputs
    # ------------------------------------------------------------------
    output_filename = edit_filename(base_filename, suffix="data")
    output_branch_names = get_branches(output_params, ["common", "data"])
    save_bdt_arrays(
        dataset_params.data_file,
        dataset_params.tree_name,
        output_filename,
        dataset_params.tree_name,
        output_branch_names,
        output_params.score_branch,
        data_scores,
        None,
        preselection=model_params.preselection,
        n_evts=debug_n_evts,
    )

    data_branch_payload = {
        "bdt_fold_index": data_fold_ids.astype(np.int32),
        "bdt_is_sideband_training_subset": mask_bkg.astype(np.int32),
        "bdt_is_oof_scored": mask_bkg.astype(np.int32),
        "bdt_is_ensemble_scored": (~mask_bkg).astype(np.int32),
        "bdt_ensemble_score_std": data_ensemble_std.astype(np.float32),
    }
    append_extra_branches_to_root(output_filename, dataset_params.tree_name, data_branch_payload, lgr=lgr)
    lgr.log(f"Data Measurement File (sideband OOF + non-sideband ensemble): {output_filename}")

    output_filename = edit_filename(base_filename, suffix="rare")
    output_branch_names = get_branches(output_params, ["common", "mc"])
    save_bdt_arrays(
        dataset_params.rare_file,
        dataset_params.tree_name,
        output_filename,
        dataset_params.tree_name,
        output_branch_names,
        output_params.score_branch,
        mc_scores,
        None,
        preselection=model_params.preselection,
        n_evts=debug_n_evts,
    )

    mc_oof_mask = mc_train_signal_mask
    mc_ensemble_mask = ~mc_train_signal_mask
    mc_branch_payload = {
        "bdt_fold_index": mc_fold_ids.astype(np.int32),
        "bdt_is_signal_mc": mask_sig_all.astype(np.int32),
        "bdt_is_signal_training_pool": mc_train_signal_mask.astype(np.int32),
        "bdt_is_signal_validation_holdout": mc_validation_mask.astype(np.int32),
        "bdt_is_oof_scored": mc_oof_mask.astype(np.int32),
        "bdt_is_ensemble_scored": mc_ensemble_mask.astype(np.int32),
        "bdt_ensemble_score_std": mc_ensemble_std.astype(np.float32),
    }
    append_extra_branches_to_root(output_filename, dataset_params.tree_name, mc_branch_payload, lgr=lgr)
    lgr.log(
        (
            "Rare MC Measurement File "
            "with 90 percent signal OOF plus validation/non-signal ensemble: "
            f"{output_filename}"
        )
    )

    # ------------------------------------------------------------------
    # Score all explicitly listed remaining samples with all k models and
    # average scores, since these samples were not used in this training.
    # ------------------------------------------------------------------
    lgr.log(
        f"Scoring other configured samples with all {len(all_models)} k-fold models and averaging scores",
        just_print=True,
    )

    # Load remaining samples from config
    other_data_files = getattr(dataset_params, "other_data_files", None) or {}
    other_mc_files = getattr(dataset_params, "other_mc_files", None) or {}

    # Score remaining data samples (e.g. same-sign data) using model averages
    for data_name, data_file in other_data_files.items():
        X_data_extra, _, _ = read_bdt_arrays(
            data_file,
            dataset_params.tree_name,
            model_params.features,
            None,
            model_params.preselection,
            (dataset_params.b_mass_branch, dataset_params.ll_mass_branch),
            n_evts=debug_n_evts,
        )

        if X_data_extra is None or X_data_extra.shape[0] == 0:
            lgr.log(f"Data ({data_name}) has 0 events after preselection; skipping.", just_print=True)
            del X_data_extra
            gc.collect()
            continue

        score_matrix = predict_score_matrix(all_models, X_data_extra)
        avg_scores = np.mean(score_matrix, axis=1).astype(np.float64)
        std_scores = np.std(score_matrix, axis=1).astype(np.float64)

        output_filename = edit_filename(base_filename, suffix=data_name)
        output_branch_names = get_branches(output_params, ["common", "data"])
        save_bdt_arrays(
            data_file,
            dataset_params.tree_name,
            output_filename,
            dataset_params.tree_name,
            output_branch_names,
            output_params.score_branch,
            avg_scores,
            preselection=model_params.preselection,
            n_evts=debug_n_evts,
        )

        branch_payload = {
            f"bdt_score_fold{fold}": score_matrix[:, i].astype(np.float32)
            for i, fold in enumerate(range(n_folds))
        }
        branch_payload["bdt_score_std"] = std_scores.astype(np.float32)
        append_extra_branches_to_root(output_filename, dataset_params.tree_name, branch_payload, lgr=lgr)

        lgr.log(f"Data ({data_name}) Measurement File (avg over all k-fold models): {output_filename}")

        del X_data_extra, score_matrix, avg_scores, std_scores, branch_payload
        gc.collect()

    # Score remaining MC samples using model averages
    for mc_name, mc_file in other_mc_files.items():
        X_mc_extra, _, _ = read_bdt_arrays(
            mc_file,
            dataset_params.tree_name,
            model_params.features,
            model_params.sample_weights,
            model_params.preselection,
            (dataset_params.b_mass_branch, dataset_params.ll_mass_branch),
            n_evts=debug_n_evts,
        )

        if X_mc_extra is None or X_mc_extra.shape[0] == 0:
            lgr.log(f"MC ({mc_name}) has 0 events after preselection; skipping.", just_print=True)
            del X_mc_extra
            gc.collect()
            continue

        score_matrix = predict_score_matrix(all_models, X_mc_extra)
        avg_scores = np.mean(score_matrix, axis=1).astype(np.float64)
        std_scores = np.std(score_matrix, axis=1).astype(np.float64)

        output_filename = edit_filename(base_filename, suffix=mc_name)
        output_branch_names = get_branches(output_params, ["common", "mc"])
        save_bdt_arrays(
            mc_file,
            dataset_params.tree_name,
            output_filename,
            dataset_params.tree_name,
            output_branch_names,
            output_params.score_branch,
            avg_scores,
            preselection=model_params.preselection,
            n_evts=debug_n_evts,
        )

        branch_payload = {
            f"bdt_score_fold{fold}": score_matrix[:, i].astype(np.float32)
            for i, fold in enumerate(range(n_folds))
        }
        branch_payload["bdt_score_std"] = std_scores.astype(np.float32)
        append_extra_branches_to_root(output_filename, dataset_params.tree_name, branch_payload, lgr=lgr)

        lgr.log(f"MC ({mc_name}) Measurement File (avg over all k-fold models): {output_filename}")

        del X_mc_extra, score_matrix, avg_scores, std_scores, branch_payload
        gc.collect()


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg["datasets"])
    model_params = argparse.Namespace(**cfg["model"])
    output_params = argparse.Namespace(**cfg["output"])

    if args.debug:
        args.verbose = True
        output_params.output_dir = Path("outputs/tmp")
    else:
        output_params.output_dir = Path(output_params.output_dir)

    bdt_inference(dataset_params, model_params, output_params, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        required=True,
        help="BDT configuration file (.yml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="print parameters and training progress to stdout",
    )
    parser.add_argument(
        "-db",
        "--debug",
        dest="debug",
        action="store_true",
        help="debug mode: run in verbose mode with scaled-down model and datasets",
    )
    parser.add_argument(
        "--n_folds",
        dest="n_folds",
        type=int,
        default=10,
        help="Number of paired OOF folds. Default: 10.",
    )
    parser.add_argument(
        "--mc_validation_frac",
        dest="mc_validation_frac",
        type=float,
        default=0.10,
        help="Fraction of low-q2 signal MC held out before k-fold training. Default: 0.10.",
    )
    parser.add_argument(
        "--early_stop_validation_frac",
        dest="early_stop_validation_frac",
        type=float,
        default=0.05,
        help="Internal validation fraction used only for XGBoost early stopping. Default: 0.05.",
    )
    parser.add_argument(
        "--eval_fold",
        dest="eval_fold",
        default=None,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    main(args)