from scipy.stats import uniform, randint

# Define the parameter distributions for hyperparameter tuning
LIGHTGBM_PARAMS = {
    'n_estimators': randint(100, 500),  # Number of boosting rounds
    'learning_rate': uniform(0.01, 0.3),  # Learning rate
    'num_leaves': randint(20, 150),  # Maximum number of leaves
    'max_depth': randint(3, 15),  # Maximum depth of the tree
    'min_child_samples': randint(10, 100),  # Minimum number of samples in a leaf
    'subsample': uniform(0.5, 0.5),  # Sub
    'colsample_bytree': uniform(0.5, 0.5),  # Subsample ratio of columns when constructing each tree
    'reg_alpha': uniform(0, 1),  # L1 regularization term
    'reg_lambda': uniform(0, 1),  # L2 regularization term
    'boosting_type': ['gbdt', 'dart', 'goss']  # Type of boosting algorithm
}

RANDOM_SEARCH_PARAMS = {
    'n_iter': 100,  # Number of parameter settings that are sampled
    'cv': 5,  # Number of cross-validation folds
    'verbose': 1,  # Verbosity level
    'random_state': 42,  # Random state for reproducibility
    'n_jobs': -1,  # Use all available cores
    'scoring': 'accuracy'  # Evaluation metric for hyperparameter tuning

}