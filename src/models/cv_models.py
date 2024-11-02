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
import optuna
from optuna import Trial, visualization
from optuna.pruners import MedianPruner
from optuna.samplers import BruteForceSampler
from sklearn.model_selection import KFold
import os
import pandas as pd
import numpy as np


class TabNetTuner2(TabNetClassifier):

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
            max_epochs=20,
            patience=0,
            compute_importance=False,
        )





class CrossValidation:

    def __init__(self, classifier: str, param_dict: dict, bpi_dict: dict, tree_method='hist', cvs=5, random_state=42, n_jobs=-1, gpu=False):
        self.classifier = classifier
        self.bpi_dict = bpi_dict

        self.param_dict = param_dict

        self.random_state = random_state
        self.cvs = cvs
        self.tree_method = tree_method
        self.n_jobs= n_jobs
        self.gpu=gpu
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def TabNetObjective(self,trial):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        X = pd.read_feather(os.path.join(self.bpi_dict["predict_path"], "X.feather"))
        y = pd.read_feather(os.path.join(self.bpi_dict["predict_path"], "y.feather"))
        X = np.array(X.values, dtype=np.float64)
        y = np.array(y.values.flatten(), dtype=np.int64)
        #mask_type = trial.suggest_categorical("mask_type", ["entmax"])
        n_da = trial.suggest_categorical("n_da", [3, 8])
        n_steps = trial.suggest_categorical("n_steps", [3, 5])
        #gamma = trial.suggest_categorical("gamma", [1.3])
        n_shared = trial.suggest_categorical("n_shared", [1, 3])
        #lambda_sparse = trial.suggest_categorical("lambda_sparse", [1e-6])
        patience = 10
        max_epochs=100
        tabnet_params = dict(n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=1.3,
                             lambda_sparse=1e-6, optimizer_fn=torch.optim.Adam,
                             optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                             mask_type="entmax", n_shared=n_shared,
                             scheduler_params=dict(mode="min",
                                                   patience=patience,#trial.suggest_int("patience",low=15,high=30),
                                                   #max_epochs=max_epochs, #trial.suggest_int('epochs', 1, 100),                                               min_lr=1e-5,
                                                   factor=0.5,),
                             scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                             verbose=0,
                             seed=42
                             ) #early stopping
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        CV_score_array = []
        for train_index, test_index in kf.split(X):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            clf = TabNetClassifier(**tabnet_params, device_name=DEVICE)
            clf.fit(X_train=X_train, y_train=y_train,
                    eval_set=[(X_valid, y_valid)],
                    max_epochs=50,
                    patience=patience,#trial.suggest_int("patience",low=15,high=30),
                    eval_metric=['auc'],
                    batch_size=512,
                    virtual_batch_size=32)
            CV_score_array.append(clf.best_cost)
        avg = np.mean(CV_score_array)
        return avg

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
            clf = optuna.create_study(direction="maximize",
                                                          study_name='TabNet optimization',
                                                          pruner=MedianPruner(),
                                                          sampler=BruteForceSampler(seed=42)
                                                          )
            clf.optimize(self.TabNetObjective,
                           #n_trials=2,
                           n_jobs=-1
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
