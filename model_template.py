from xgboost import XGBClassifier

model = XGBClassifier(
        # --- Core Physics Parameters ---
        max_depth             = 6,
        learning_rate         = 0.03,
        n_estimators          = 10000,
        early_stopping_rounds = 100,
        
        # --- Regularization ---
        min_child_weight      = 200.,
        gamma                 = 1.0,
        reg_lambda            = 5.0,
        
        # --- Randomness (Bagging) ---
        subsample             = 0.5,
        colsample_bytree      = 0.8,
        
        # --- Technical ---
        objective             = 'binary:logistic',
        eval_metric           = 'logloss',
        scale_pos_weight      = 1.,
        
        # --- Performance ---
        tree_method           = 'auto',
        n_jobs                = 4,
        num_parallel_tree     = 1,
        
        # --- Reproducibility ---
        seed                  = 271996,
        random_state          = 271996,
    )
