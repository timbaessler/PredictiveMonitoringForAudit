import os
import numpy as np
import sklearn.metrics as metrics
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.augmentations import ClassificationSMOTE
import pandas as pd
import glob
import sys
import os
import warnings
import matplotlib.pyplot as plt
import torch
import xgboost
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')
sys.path.append('..')
from src.models.optim import ThreshholdOptimizer
from src.models.model_loader import load_trained_xgboost
from src.preprocessing.data_loader import load_train_test
from src.models.cv_models import TabNetTuner
from config.data_config import bpic_2018_dict as bpi_dict
from config.model_config import param_dict
from src.models.cv_models import TabNetTuner, CrossValidation




DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import multiprocessing
cores = multiprocessing.cpu_count()
plt.rcParams.update({'font.size': 2})
desired_width = 200
pd.set_option('display.width', desired_width)
pd.set_option("display.max_rows", 40)
pd.set_option("display.max_columns", 20)


def ensemble_predict(X, X2, model1, model2):
    # Get predictions from both models
    model1_preds = model1.predict_proba(X)[:, 1]
    model2_preds = model2.predict(X2)

    # Average the predictions
    ensemble_preds = np.mean([model1_preds, model2_preds], axis=0)

    # Convert probabilities to class labels
    ensemble_labels = np.where(ensemble_preds > 0.5, 1, 0)

    return ensemble_preds, ensemble_labels



def plot_history(clf, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(clf.history["train_logloss"], label="train_logloss")
    ax1.plot(clf.history["val_logloss"], label="val_logloss")
    ax1.legend()
    ax2.plot(clf.history["train_auc"], label="train_auc")
    ax2.plot(clf.history["val_auc"], label="val_auc")
    ax2.legend()
    fig.savefig(fname+"history.png", dpi=500)




if __name__ == "__main__":
    print(DEVICE)
    print(cores)
    predict_path = bpi_dict["predict_path"]
    processed_path = bpi_dict["processed_path"]
    log = pd.read_feather(bpi_dict["labelled_log_path"])
    print(log)
    print(np.median(log.groupby("case:concept:name")["trace_length"].first()))
    print(log["activity"].nunique())
    sys.exit()
    log["case:year"] = log["case:year"].astype(int)
    train_idx = log[(log["case:year"]==2015)|(log["case:year"]==2016)
                    ]["case:concept:name"].unique().tolist()
    test_idx = log[(log["case:year"]==2017)]["case:concept:name"].unique().tolist()
    cv1 = log[(log["case:year"]==2015)]["case:concept:name"].unique().tolist()
    cv2 = log[(log["case:year"]==2016)]["case:concept:name"].unique().tolist()
    static_cat_cols = bpi_dict["static_cat_cols"]
    dynamic_cat_cols = bpi_dict["dynamic_cat_cols"]
    num_cols = bpi_dict["num_cols"]
    res_cv = pd.DataFrame({"p":[], "n":[], "n_steps":[],
                           "avg_train_auc":[], "max_train_auc":[], "avg_val_auc":[], "max_val_auc":[],
                           "tabnet_auc":[], "xgboost_auc":[], "ensemble_auc":[]})
    max_auc = 0.5
    cv_res = pd.DataFrame()
    for classifier in list([#"TabNet",
                            #"XGBoost", "RandomForest",
        "DecisionTree", "MLP", "LogisticRegression"
                           ]):
        for pr in range(15):
            aug = ClassificationSMOTE(p=0.1)
            # Filenames
            fname = os.path.join(predict_path, classifier +str(pr).zfill(2)+ "_")
            filename = os.path.join(processed_path,str(pr).zfill(2)+"_")


            # Train Test Split
            X_train, X_test, y_train, y_test = load_train_test(log, filename, num_cols, static_cat_cols,
                                                                                          dynamic_cat_cols,
                                                                                          pr, train_idx, test_idx,
                                                                                          read=False)

            FEATS = X_train.columns

            # CV Data
            X_train = np.array(X_train.values, dtype=np.float64)
            y_train = np.array(y_train.values.flatten(), dtype=np.int64)
            X_test = np.array(X_test.values, dtype=np.float64)
            y_test = np.array(y_test.values.flatten(), dtype=np.int64)

            crossval = CrossValidation(classifier=classifier,
                                       param_dict=param_dict[classifier],
                                       cvs=5,
                                       n_jobs=max(cores-1, 2)
                                       )

            clf = crossval.get_classifier()
            clf.fit(X_train, y_train)
            results_df = pd.DataFrame(clf.cv_results_)
            results_df = results_df.sort_values(by=["rank_test_score"])
            results_df.to_csv(fname + 'cross_val_scores.csv', index=False, sep=";")

            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            y_pred_proba_train = clf.predict_proba(X_train)[:, 1]
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            auc = np.round(auc, 5)
            print(classifier, auc)
            res = pd.DataFrame({
                "Classifier":classifier,
                "timediff": pr,
                "AUC": auc}, index=[0])
            res.to_csv(fname + 'results.csv', sep=";", index=False)
            res_all = pd.DataFrame()
            for f in glob.glob(os.path.join(predict_path, "*results.csv")):
                res_all = res_all.append(pd.read_csv(f, delimiter=";"))
            res_all.to_csv(os.path.join(predict_path, "r_all_bpic2018.csv"), sep=";", index=False)
            np.save(fname + "y_pred_proba_train.npy", y_pred_proba_train)
            np.save(fname +"y_train.npy", y_train)
            np.save(fname + "y_pred_proba_test.npy", y_pred_proba)
            np.save(fname + "y_test.npy", y_test)
            opt = ThreshholdOptimizer(r=1, p=2)
            opt.fit(y_true=y_train, y_pred_proba=y_pred_proba_train)
            opt.plot_threshold()
            plt.savefig(fname+"_thresh.png", dpi=500)
            y_pred_proba_opt = opt.predict(y_pred_proba)