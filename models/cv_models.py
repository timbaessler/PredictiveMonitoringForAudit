from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

class CrossValidation:

    def __init__(self, classifier, gpu=False):
        self.classifier = classifier
        self.gpu = gpu

    def get_classifier(self):
        if self.classifier == "XGBoost":
            param_dict = dict()
            param_dict["learning_rate"] = [0.01, 0.1]
            param_dict["min_child_weight"] = ([1, 3])
            param_dict["max_depth"] = [6, 10]
            param_dict["alpha"] = [1]
            param_dict["lambda"] = [5]
            param_dict["gamma"] = [5]
            param_dict["colsample_bytree"] = [0.7]
            param_dict["n_estimators"] = [100, 500]

            if self.gpu:
                clf = GridSearchCV(XGBClassifier(random_state=42,
                                             eval_metric="auc",
                                             tree_method='gpu_hist'),
                           param_grid=param_dict,
                           scoring='roc_auc',
                           refit=True,
                           n_jobs=-1,
                           cv=3,
                           verbose=10)
            else:
                clf = GridSearchCV(XGBClassifier(random_state=42, eval_metric="auc"),
                           param_grid=param_dict,
                           scoring='roc_auc',
                           refit=True,
                           cv=3,
                           verbose=10)

        elif self.classifier == "RandomForest":

            param_grid = {"max_depth": [6, 10],
                  "n_estimators": [100, 500],
                  "max_features":[0.7, 1]}

            clf = GridSearchCV(RandomForestClassifier(random_state=42),
                       param_grid=param_grid,
                       refit=True,
                       cv=3,
                       verbose=10,
                       n_jobs=-1,
                       scoring='roc_auc')
        else:
            raise RuntimeError
        return clf