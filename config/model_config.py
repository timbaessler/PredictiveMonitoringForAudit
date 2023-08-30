param_dict = dict({
    "RandomForest": {
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 6, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]},

    "XGBoost": {"learning_rate": [0.01, 0.1],
                "min_child_weight":[1, 3],
                "subsample": [0.7, 1],
                "colsample_bytree": [0.7, 1],
                "max_depth": [3, 6],
                "n_estimators": [100]},
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf", "poly"],
        "degree": [2, 3, 4],
        "gamma": ["scale", "auto"],
        "coef0": [0, 1, 2]
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "learning_rate_init": [0.001, 0.01, 0.1]
    }
})