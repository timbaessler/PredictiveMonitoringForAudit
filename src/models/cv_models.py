from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import svm
from xgboost import XGBClassifier
from sys import platform
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
import torch
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from pytorch_tabnet.augmentations import ClassificationSMOTE


class TabNetTuner(TabNetClassifier):

    def fit(self, X_train, y_train, *args, **kwargs):
        self.n_d = self.n_a
        self.seed = 42
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                                           train_size=0.8,
                                                                           shuffle=True,
                                                                           random_state=42)

        return super().fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            virtual_batch_size=32,
            eval_metric=['auc'],
            max_epochs=10,
            patience=10,
            compute_importance=False,
        )



class CrossValidation:

    def __init__(self, classifier: str, param_dict: dict, tree_method='hist', cvs=5, random_state=42, n_jobs=0, gpu=False):
        self.classifier = classifier
        self.param_dict = param_dict
        self.random_state = random_state
        self.cvs = cvs
        self.tree_method = tree_method
        self.n_jobs=n_jobs
        self.gpu=gpu

    def get_classifier(self):
        if self.classifier == "XGBoost":
            if self.gpu:
                clf = GridSearchCV(XGBClassifier(random_state=self.random_state,
                                             eval_metric="auc",
                                             tree_method="gpu_hist",
                                            enable_categorical=True),
                                   param_grid=self.param_dict,
                                   scoring='roc_auc',
                                   refit=True,
                                   cv=self.cvs,
                                   verbose=10,
                                   n_jobs=self.n_jobs)
            else:
                clf = GridSearchCV(XGBClassifier(random_state=self.random_state,
                                                 eval_metric="auc",
                                                tree_method=self.tree_method,
                                                enable_categorical=True),
                                   param_grid=self.param_dict,
                                   scoring='roc_auc',
                                   refit=True,
                                   cv=self.cvs,
                                   verbose=10,
                      n_jobs=self.n_jobs)

        elif self.classifier == "RandomForest":
            clf = GridSearchCV(RandomForestClassifier(random_state=42),
                       param_grid=self.param_dict,
                       refit=True,
                       cv=self.cvs,
                       verbose=10,
                       scoring='roc_auc',
                               n_jobs=self.n_jobs)
        elif self.classifier == "MLP":
            clf = GridSearchCV(MLPClassifier(random_state=42),
                               param_grid=self.param_dict,
                               refit=True,
                               cv=self.cvs,
                               verbose=10,
                               scoring='roc_auc',
                               n_jobs=self.n_jobs)
        elif self.classifier == "SVM":
            clf = GridSearchCV(svm.SVC(random_state=42),
                               param_grid=self.param_dict,
                               refit=True,
                               cv=self.cvs,
                               verbose=10,
                               scoring='roc_auc',
                               n_jobs=self.n_jobs)
        elif self.classifier == "TabNet":
            clf = GridSearchCV(TabNetTuner(),
                               param_grid=self.param_dict,
                               refit=True,
                               cv=self.cvs,
                               verbose=10,
                               scoring='roc_auc',
                               n_jobs=self.n_jobs
            )
        elif self.classifier =="LogisticRegression":
            clf = GridSearchCV(LogisticRegression(random_state=42),
                               param_grid=self.param_dict,
                               refit=True,
                               cv=self.cvs,
                               verbose=10,
                               scoring='roc_auc',
                               n_jobs=self.n_jobs)
        elif self.classifier =="DecisionTree":
            clf = GridSearchCV(DecisionTreeClassifier(random_state=42),
                               param_grid=self.param_dict,
                               refit=True,
                               cv=self.cvs,
                               verbose=10,
                               scoring='roc_auc',
                               n_jobs=self.n_jobs)
        else:
            raise RuntimeError
        return clf
