param_dict = dict({
    "RandomForest": {"max_depth": [6, 10],
                                    "n_estimators": [100, 500],
                                    "max_features":[0.7, 1]},
    "XGBoost": {"learning_rate": [0.01, 0.1],
                "min_child_weight":[1, 3],
                "subsample": [0.7, 1],
                "colsample_bytree": [0.7, 1],
                "max_depth": [3, 6],
                #"alpha": [1],
                #"lambda": [5],
                #"gamma": [5],
                "n_estimators": [100]}})