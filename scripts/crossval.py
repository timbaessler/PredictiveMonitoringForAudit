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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
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

desired_width = 200
pd.set_option('display.width', desired_width)
pd.set_option("display.max_rows", 40)
pd.set_option("display.max_columns", 20)
plt.rcParams.update({'text.usetex':True,
                     'font.size':11,
                     'font.family':'serif'
                     })

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
    paired_palette = sns.color_palette("Paired")
    paired_cmap = ListedColormap(paired_palette.as_hex())
    predict_path = bpi_dict["predict_path"]
    processed_path = bpi_dict["processed_path"]
    log = pd.read_feather(bpi_dict["labelled_log_path"])
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
    cv_res = pd.DataFrame()

    for classifier in list(["TabNet","XGBoost","RandomForest", "DecisionTree", "MLP", "LogisticRegression"]):
        for pr in range(7, 8):
            aug = ClassificationSMOTE(p=0.1)
            # Filenames
            fname = os.path.join(predict_path, classifier +str(pr).zfill(2)+ "_")
            filename = os.path.join(processed_path,str(pr).zfill(2)+"_")

            # Train Test Split
            X_train, X_test, y_train, y_test = load_train_test(log, filename, num_cols, static_cat_cols,
                                                                                          dynamic_cat_cols,
                                                                                          pr, train_idx, test_idx,
                                                                                          read=True)

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
            # Training
            clf = crossval.get_classifier()
            clf.fit(X_train, y_train)
            results_df = pd.DataFrame(clf.cv_results_)
            results_df = results_df.sort_values(by=["rank_test_score"])
            results_df.to_csv(fname + 'cross_val_scores.csv', index=False, sep=";")

            y_pred_train = clf.predict(X_train)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            y_pred_proba_train = clf.predict_proba(X_train)[:, 1]
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            auc = np.round(auc, 5)
            res = pd.DataFrame({
                "Classifier":classifier,
                "timediff": pr,
                "AUC": auc}, index=[0])
            res.to_csv(fname + 'results.csv', sep=";", index=False)
            res_all = pd.DataFrame()
            for f in glob.glob(os.path.join(predict_path, "*results.csv")):
                res_all = res_all.append(pd.read_csv(f, delimiter=";"))
            res_all.to_csv(os.path.join(predict_path, "r_all_bpic2018.csv"), sep=";", index=False)

            # OOF / Test-Data
            y_pred = clf.predict(X_test)
            y_pred_proba_test = y_pred_proba.copy()

            # Threshold Optimization
            recall_list = list()
            recall_test_list = list()
            precision_list = list()
            precision_test_list = list()
            threshhold_list = list()
            x_s = list()

            y_pred_baseline_train = np.where(y_pred_proba_train >= 0.5, 1, 0)
            prec_baseline_train = metrics.precision_score(y_true=y_train, y_pred=y_pred_baseline_train)
            rec_baseline_train = metrics.recall_score(y_true=y_train, y_pred=y_pred_baseline_train)

            y_pred_baseline = np.where(y_pred_proba_test >= 0.5, 1, 0)
            prec_baseline = metrics.precision_score(y_true=y_test, y_pred=y_pred_baseline)
            rec_baseline = metrics.recall_score(y_true=y_test, y_pred=y_pred_baseline)
            for r, p in list([
                (10, 1), (9,1), (8, 1), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1),
                 (1, 2),  (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)]):
                x_s.append(str(r)+":"+str(p))
                opt = ThreshholdOptimizer(r=r, p=p)
                opt.fit(y_pred_proba=y_pred_proba_train, y_true=y_train)
                y_pred_test = opt.predict(y_pred_proba_test)
                if r == 1 and p == 3:
                    fig32 = opt.plot_threshold()
                    fig32.suptitle("Threshold Optimization (r=1, p=3) for BPIC2018 (-"+str(pr)+" bus. days)")
                    fig32.savefig(fname+'bpic2018_thresh.png', dpi=500)
                if r == 2 and p == 1:
                    fig13 = opt.plot_threshold()
                    fig13.suptitle("Threshold Optimization (r=2, p=1) for BPIC2018 (-"+str(pr)+" bus. days)")
                    fig13.savefig(fname +'bpic2018_thresh2.png', dpi=700)
                recall_list.append(opt.recall[opt.min_pos])
                recall_test_list.append(metrics.recall_score(y_pred=y_pred_test, y_true=y_test))
                precision_test_list.append(metrics.precision_score(y_pred=y_pred_test, y_true=y_test))
                precision_list.append(opt.precision[opt.min_pos])
                threshhold_list.append(opt.theta)

            fig0, (ax1) = plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True)
            ax1.plot(x_s, precision_list, label='precision (train)', linewidth=2, linestyle="dotted", color="black")
            ax1.plot(x_s, recall_list, label='recall (train)', linewidth=2, linestyle="dashed", color="black")
            ax1.plot(x_s, precision_test_list, label='precision (test)', linewidth=2, marker="o", markersize=8, color='#1F78B4')
            ax1.plot(x_s, recall_test_list, label='recall (test)', linewidth=2, marker="v",color='#E31A1C', markersize=8 )
            ax1.plot(x_s, threshhold_list, label=r'\(\theta\)', linewidth=2, color='#33A02C')
            ax1.axhline(y=rec_baseline, label='baseline recall (test)', linestyle='dashdot', color='#E31A1C')
            ax1.axhline(y=prec_baseline, label='baseline precision (test)', linestyle='--', color='#1F78B4')
            ax1.set_xlabel("r:p", fontsize=10)
            ax1.set_xlim(x_s[0], x_s[-1])
            ax1.legend()
            ax1.grid()
            fig0.tight_layout()
            fig0.savefig(fname +'r_p.png', dpi=700)
