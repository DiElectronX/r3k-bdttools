from xgboost import XGBClassifier

model = XGBClassifier(
    max_depth        = 15,
    n_estimators     = 100,
    learning_rate    = 0.4,
    min_child_weight = 1.4,
    gamma            = 0.5,
    subsample        = 1,
    colsample_bytree = 0.7,
    num_parallel_tree= 7,
    scale_pos_weight = 1.,
    reg_lambda       = 3,
    objective        = 'binary:logitraw',
    eval_metric      = ['logloss'],
)
