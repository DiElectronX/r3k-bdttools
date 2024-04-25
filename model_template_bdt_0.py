from xgboost import XGBClassifier

model = XGBClassifier(
    max_depth        = 9,
    n_estimators     = 100,
    learning_rate    = 0.35,
    min_child_weight = 1.4,
    gamma            = 0.5,
    subsample        = 0.85,
    colsample_bytree = 0.7,
    num_parallel_tree= 8,
    scale_pos_weight = 0.7,
    reg_lambda       = 3,
    objective        = 'binary:logitraw',
    eval_metric      = ['logloss'],
)
