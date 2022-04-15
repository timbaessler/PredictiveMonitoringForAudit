import os
import sys
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings('ignore')
sys.path.append('..')
from src.models.cv_models import CrossValidation
from src.models.encoding import Aggregation
from src.models.bucketing import StateBucketing
from src.models.optim import ThreshholdOptimizer
from src.models.split import temporal_train_split
from config.data_config import bpic_2019_dict as bpi_dict
from config.model_config import param_dict
desired_width = 200
pd.set_option('display.width', desired_width)
pd.set_option("display.max_rows", 40)
pd.set_option("display.max_columns", 20)

if __name__ == "__main__":
    predict_path = bpi_dict["predict_path"]
    log = pd.read_feather(bpi_dict["labelled_log_path"])
    train_idx, test_idx = temporal_train_split(log, test_size=0.33)
    static_cat_cols = bpi_dict["static_cat_cols"]
    dynamic_cat_cols = bpi_dict["dynamic_cat_cols"]
    num_cols = bpi_dict["num_cols"]
    res = pd.DataFrame()
    for classifier in list(["RandomForest"]):
        for state in list(["Clear Invoice",
                           "Record Invoice Receipt",
                           "Record Goods Receipt",
                           "Create Purchase Order Item"
                           ]):
            fname = os.path.join(predict_path, classifier + state+ "_")
            if os.path.exists(fname + ".sav"):
                continue
            state_extractor = StateBucketing(state)
            log2 = state_extractor.fit_transform(log)
            onehot =  True
            agg_transformer = Aggregation(num_cols, static_cat_cols, dynamic_cat_cols, one_hot_static=onehot)
            X, y = agg_transformer.fit_transform(log2)
            del log2
            X_train = X[X.index.isin(train_idx)].values
            y_train = y[y.index.isin(train_idx)].values
            X_test = X[X.index.isin(test_idx)].values
            y_test = y[y.index.isin(test_idx)].values
            del X
            crossval = CrossValidation(classifier, param_dict[classifier])
            clf = crossval.get_classifier()
            clf.fit(X_train, y_train)
            pickle.dump(clf, open(fname + ".sav", 'wb'))
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            y_pred_proba_train = clf.predict_proba(X_train)[:, 1]
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            auc = np.round(auc, 5)
            np.save(fname + "y_pred_proba_train.npy", y_pred_proba_train)
            np.save(fname +"y_train.npy", y_train)
            np.save(fname + "y_pred_proba_test.npy", y_pred_proba)
            np.save(fname + "y_test.npy", y_test)
            opt = ThreshholdOptimizer(r=2, p=1)
            opt.fit(y_true=y_train, y_pred_proba=y_pred_proba_train)
            opt.plot_threshold()
            plt.savefig(fname+"_thresh.png", dpi=500)
            y_pred_proba_opt = opt.predict(y_pred_proba)

            results_df = pd.DataFrame(clf.cv_results_)
            results_df = results_df.sort_values(by=["rank_test_score"])
            results_df.to_csv(fname + 'cross_val_scores.csv', index=False, sep=";")
            res = pd.DataFrame({
                "Classifier": classifier,
                "State": state,
                "AUC": auc}, index=[0])
            res.to_csv(fname + 'results.csv', sep=";", index=False)

            res_all = pd.DataFrame()
            for f in glob.glob(os.path.join(predict_path, "*results.csv")):
                res_all = res_all.append(pd.read_csv(f, delimiter=";"))
            res_all.to_csv(os.path.join(predict_path, "r_all_bpic2019.csv"), sep=";", index=False)

