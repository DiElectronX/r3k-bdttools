# BDT Tools for Run 3 R(K) Analysis

This repository contains tools for training, evaluating, and applying Boosted Decision Trees (BDTs) for the R(K) analysis using XGBoost.

## 1\. Setting up Environment

You can set up a computing environment with the necessary Python libraries (XGBoost, Uproot, NumPy, Matplotlib, etc.) using one of the following methods:

**Option A: Conda (Recommended)**

```bash
conda env create -f environment.yml
conda activate r3k_bdt
```

**Option B: Pip**

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

-----

## 2\. Preparing Inputs

The BDT training and inference scripts require ROOT files. The tool is flexible regarding branch names.

### Method A: Use Compliant Files (Zero Prep)

If your input ROOT files already use the standard analysis branch names (e.g., `Bmass`, `Mll`, `Bprob`), you can skip directly to **Section 3**. The scripts will read them natively.

### Method B: Smart Auto-Mapping (Recommended)

You do **not** need to create new copies of your files just to rename branches. The training scripts now support "Smart Lookup".

1.  Define your features in `bdt_cfg.yml`.
2.  The scripts will automatically look for those names.
3.  If not found, they will check internal alias maps (e.g., mapping `Bmass` $\to$ `BToKEE_fit_mass`) automatically.

### Method C: Lightweight Rename (Optional)

If you prefer to physically rename branches in a file (e.g., for sharing or archival), use the `quick_inference.py` tool in rename-only mode. This creates a lightweight copy with the new names.

```bash
python quick_inference.py \
    -i <input_file.root> \
    -o <output_dir> \
    --just-rename \
    --label _wRenames
```

  * `--just-rename`: Skips BDT inference, only creates a new tree with renamed branches.
  * This uses the internal JSON mapping to standardize variable names.

-----

## 3\. BDT Training & Inference

The main driver script `bdt_inference.py` handles the full analysis pipeline:

1.  **K-Fold Training:** Trains independent models on Data Sidebands + Rare MC.
2.  **Inference:** Applies the models to Data and Signal MC (using K-Fold cross-validation indices to avoid bias).
3.  **Application:** Applies the models to external files (Control channels like J/psi).
4.  **Plotting:** Generates ROC curves, Score distributions, and Feature Importance plots.

### Running the Script

```bash
python bdt_inference.py -c bdt_cfg.yml [-v] [--debug] [-cm <path to pretrained .json model>] [-np]
```

  * `-c bdt_cfg.yml`: Path to your configuration file (see below).
  * `-v`: Verbose mode (prints training progress).
  * `--debug`: Runs on a small subset of events (100k) and outputs to `outputs/tmp/` for testing logic.
  * `-cm`: Run in cached model mode.
      * Given to a pre-trained `.json` model file/output dir as an argument, the script will skip the training loop and instead run inference directly on the additional files (described below).
  * `-np`: Run in no plot mode.

### Configuration: `bdt_cfg.yml`

This YAML file controls the entire pipeline.

```yaml
datasets:
  # --- Training & Measurement Files ---
  # These are used to train the K-Fold models.
  data_file: outputs/data_slimmed.root
  rare_file: outputs/rare_mc_slimmed.root
  
  # --- External Inference Files ---
  # Models will be applied to these files after training using random fold assignment.
  other_data_files: 
    same_sign: outputs/same_sign_data.root
  
  other_mc_files: 
    jpsi: outputs/jpsi_mc.root
    psi2s: outputs/psi2s_mc.root
    kstar_jpsi: outputs/bkg_kstar_jpsi.root

  # --- Tree Info ---
  tree_name: Events        # Name of the TTree in your files
  b_mass_branch: Bmass     # Branch used for sideband masking
  ll_mass_branch: Mll      # Branch used for q2 masking

model:
  template_file: model_template.py
  # List of branches to use as BDT input features. 
  # Math is allowed (e.g. "L2iso/L2pt") if branches exist in file/mapping.
  features: [Bprob, BsLxy, L2iso/L2pt, Bcos, Kiso/Kpt, LKdz, LKdr, Passymetry, Kip3d/Kip3dErr, L1id, L2id]
  
  # LaTeX labels for plotting feature importance
  feature_labels: 
    - "$prob({\\rm SV})$"
    - "$L_{xy}/\\sigma_{L_{xy}}$"
    - "..."
    
  sample_weights: total_weight  # MC weight branch name
  preselection: "Bmass > 4.5"   # Optional TTree cut string applied to all inputs

output:
  output_dir: outputs/run3_test/
  score_branch: bdt_score       # Name of the new BDT score branch in output files
  
  # Branches to copy from input -> output files
  output_branches: 
    common: [Bmass, Mll]        # Saved for ALL files
    data:                       # Saved only for Data files
    mc: [total_weight]          # Saved only for MC files
    
  log_file: log.txt
```

### Model Configuration: `model_template.py`

This python file must contain a `model` object (an instance of `XGBClassifier`). This allows you to define the hyperparameters and architecture in standard Python.

**Example `model_template.py`:**

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    # --- Physics / Complexity ---
    max_depth             = 6,            # Tree depth (keep < 7 for physics)
    learning_rate         = 0.03,         # Step size (lower = more robust, needs more trees)
    n_estimators          = 10000,        # Max trees (stops early if converged)
    early_stopping_rounds = 100,          # Stop if valid_loss doesn't improve for 100 rounds
    
    # --- Regularization (Anti-Overfitting) ---
    min_child_weight      = 200.,         # High value prevents splitting on single high-weight MC events
    gamma                 = 1.0,          # Min loss reduction to make a split
    reg_lambda            = 5.0,          # L2 regularization
    
    # --- Stability ---
    subsample             = 0.5,          # Use 50% of data per tree
    colsample_bytree      = 0.8,          # Use 80% of features per tree
    
    # --- Technical ---
    objective             = 'binary:logistic', # Required for prob outputs (0-1)
    eval_metric           = 'logloss',
    n_jobs                = 4,
    seed                  = 271996,
)
```

-----

## 4\. Plotting

The training script automatically generates plots in the `plots/` subdirectory of your output folder. However, if you want to regenerate plots later (e.g., changing aesthetics) without retraining, you can use the standalone plotting scripts.

The training script saves the raw plotting data into `.pkl` (pickle) files in the output directory.

**Generate Feature Importance Plot:**

```bash
python plotting_scripts/feature_importance_plotter.py \
    -c bdt_cfg.yml \
    -f <output_dir>/plots/feature_importance.pkl \
    -o <output_dir>/plots/new_importance.pdf
```

**Generate ROC Curve:**

```bash
python plotting_scripts/roc_plotter.py \
    -c bdt_cfg.yml \
    -f <output_dir>/plots/roc.pkl \
    -o <output_dir>/plots/new_roc.pdf
```

**Generate Score Distribution:**

```bash
python plotting_scripts/score_plotter.py \
    -c bdt_cfg.yml \
    -f <output_dir>/plots/scores.pkl \
    -o <output_dir>/plots/new_scores.pdf
```
