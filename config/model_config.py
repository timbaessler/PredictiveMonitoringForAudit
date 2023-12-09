import torch

param_dict = dict({
    "RandomForest": {
        "n_estimators": [50, 100],
        "max_depth": [3, 6, None],
        "min_samples_split": [2],
        "min_samples_leaf": [2],
        "max_features": ["sqrt"]},

    "XGBoost": {"learning_rate": [0.01, 0.1],
                "min_child_weight":[1, 3],
                "subsample": [0.7, 1],
                "colsample_bytree": [0.7, 1],
                "max_depth": [3, 6],
                "n_estimators": [100]},

    "TabNet":{
                 "optimizer_fn": [torch.optim.RMSprop],
                 "optimizer_params": [dict(lr=0.05)],
                 "n_a": [3, 8],
                 "n_steps": [3, 5],
                 "gamma": [0],
                 "n_shared": [1],
                 "lambda_sparse": [0.001]
             },
             "LogisticRegression" : {
    "solver": ["lbfgs", "liblinear"],
    "penalty":  ["l2"],
    "C": [0.001, 0.01, 0.1, 1]
    },
    "DecisionTree": {
        'max_depth': [None, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    },

    "SVM": {
            "C": [0.001, 0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "degree": [2, 3, 4],
            "gamma": ["scale", "auto"],
            "coef0": [0, 1, 2]
        },
    "MLP": {
        "hidden_layer_sizes": [(100,), (200, 100), (200,), (100, 50),  (100, 100, 100) ],
        "activation": ["tanh"],
        "solver": ["adam",],
        "alpha": [0.0001],
        "learning_rate": ["constant"],
        "learning_rate_init": [0.0001],
        #"verbose":[True],
        #"learning_rate": ["constant", "invscaling", "adaptive"]
    }
})