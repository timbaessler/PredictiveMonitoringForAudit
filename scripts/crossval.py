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
from config.data_config import bpic_2018_dict as bpi_dict
from config.data_config import syn_dict as bpi_dict2
from config.model_config import param_dict
from src.models.cv_models import CrossValidation
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


def assert_temporal_split(train_log, test_log, train_idx, test_idx):
    # Ensure all events in the training set occur before any events in the testing set
    assert train_log['Timestamp'].max() < test_log['Timestamp'].min(), \
        "Temporal order violated: some training events occur after the start of the test period."

    # Ensure no training events overlap with the test period
    assert train_log['Timestamp'].max() < test_log['StartTime'].min(), \
        "Overlap detected: training events overlap with the test period."

    # Ensure all events for a given CaseID are either in the training set or the test set
    train_case_ids = set(train_log['CaseID'])
    test_case_ids = set(test_log['CaseID'])
    assert train_case_ids.isdisjoint(test_case_ids), \
        "CaseID integrity violated: some cases are split between training and test sets."

    # No need to check cross-validation coverage here, since we use 5-Fold CV on train_log

    print("All assertions passed. Temporal split and cross-validation are correct.")


if __name__ == "__main__":
    paired_palette = sns.color_palette("Paired")
    paired_cmap = ListedColormap(paired_palette.as_hex())
    for data_set in ["BPIC2018","SynLog"]:
        bpi_dict = bpi_dict2.copy() if data_set == "SynLog" else bpi_dict
        predict_path = bpi_dict["predict_path"]
        processed_path = bpi_dict["processed_path"]
        log = pd.read_feather(bpi_dict["labelled_log_path"])
        if data_set == "BPIC2018":
            log["case:year"] = log["case:year"].astype(int)
            train_idx = log[(log["case:year"]==2015)|(log["case:year"]==2016)
                            ]["case:concept:name"].unique().tolist()
            test_idx = log[(log["case:year"]==2017)]["case:concept:name"].unique().tolist()
            cv1 = log[(log["case:year"]==2015)]["case:concept:name"].unique().tolist()
            cv2 = log[(log["case:year"]==2016)]["case:concept:name"].unique().tolist()
            static_cat_cols = bpi_dict["static_cat_cols"]
            dynamic_cat_cols = bpi_dict["dynamic_cat_cols"]
            num_cols = bpi_dict["num_cols"]
            case_id_col="case:concept:name"
            activity_col="case:concept"
            timestamp_col="time:timestamp"
            resource_col="org:resource"
            deadline_col = "deadline"
        elif data_set == "SynLog":
            # Step 1: Ensure the Timestamp column is in datetime format
            log['Timestamp'] = pd.to_datetime(log['Timestamp'])
            # Step 2: Calculate the start time for each case
            log['StartTime'] = log.groupby('CaseID')['Timestamp'].transform('min')
            # Step 3: Sort the log by start time of each case
            log = log.sort_values(by='StartTime')
            # Step 4: Determine the cutoff date for the 80-20 split
            cutoff_index = int(len(log['CaseID'].unique()) * 0.8)
            cutoff_case_id = log['CaseID'].unique()[cutoff_index]
            cutoff_date = log[log['CaseID'] == cutoff_case_id]['StartTime'].min()
            # Step 5: Split the log into train and test sets based on the cutoff date
            train_log = log[log['StartTime'] < cutoff_date].copy()
            test_log = log[log['StartTime'] >= cutoff_date].copy()
            # Step 6: Remove any training events that overlap with the test period
            latest_train_time = test_log['StartTime'].min()
            train_log = train_log[train_log['Timestamp'] < latest_train_time]
            # Display the resulting split
            print(f"Training set: {train_log['CaseID'].nunique()} cases, {train_log.shape[0]} events")
            print(f"Test set: {test_log['CaseID'].nunique()} cases, {test_log.shape[0]} events")
            # Step 7: Retrieve unique CaseIDs for training and testing
            train_idx = train_log['CaseID'].unique()
            test_idx = test_log['CaseID'].unique()
            assert_temporal_split(train_log, test_log, train_idx, test_idx)
            case_id_col="CaseID"
            activity_col="Activity"
            timestamp_col="Timestamp"
            resource_col="Resource"
            deadline_col = "payment_run"
            log = log.rename(columns={"Late": "y"})
        static_cat_cols = bpi_dict["static_cat_cols"]
        dynamic_cat_cols = bpi_dict["dynamic_cat_cols"]
        num_cols = bpi_dict["num_cols"]
        res_cv = pd.DataFrame({"p":[], "n":[], "n_steps":[],
                               "avg_train_auc":[], "max_train_auc":[], "avg_val_auc":[], "max_val_auc":[],
                               "tabnet_auc":[], "xgboost_auc":[], "ensemble_auc":[]})
        cv_res = pd.DataFrame()

        for classifier in list(["TabNet","XGBoost","RandomForest",  "LogisticRegression"]):

            for pr in range(15):
                print(data_set, classifier, pr)
                # Filenames
                fname = os.path.join(predict_path, classifier +str(pr).zfill(2)+ "_")
                filename = os.path.join(processed_path,str(pr).zfill(2)+"_")
                # Train Test Split
                X_train, X_test, y_train, y_test = load_train_test(log,
                                                                   filename,
                                                                   num_cols,
                                                                   static_cat_cols,
                                                                   dynamic_cat_cols,
                                                                   pr,
                                                                   train_idx,
                                                                   test_idx,
                                                                   read=True,
                                                                   case_id_col=case_id_col,
                                                                   activity_col=activity_col,
                                                                   timestamp_col=timestamp_col,
                                                                   resource_col=resource_col,
                                                                   deadline_col=deadline_col
                                                                   )

                FEATS = X_train.columns

                # CV Data
                X_train = np.array(X_train.values, dtype=np.float64)
                y_train = np.array(y_train.values.flatten(), dtype=np.int64)
                X_test = np.array(X_test.values, dtype=np.float64)
                y_test = np.array(y_test.values.flatten(), dtype=np.int64)

                crossval = CrossValidation(classifier=classifier,
                                           bpi_dict = bpi_dict,
                                           param_dict=param_dict[classifier],
                                           cvs=5,
                                           n_jobs=max(cores-1, 2)
                                           )
                # Training
                clf = crossval.get_classifier()
                if classifier == "TabNet":
                    TabNet_params = clf.best_params
                    print("best params")
                    print(TabNet_params)
                    final_params = dict(n_d=TabNet_params['n_da'],
                                        n_a=TabNet_params['n_da'],
                                        n_steps=TabNet_params['n_steps'],
                                        #gamma=TabNet_params['gamma'],
                                        lambda_sparse=1e-6,
                                        optimizer_fn=torch.optim.Adam,
                                        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                                        mask_type="entmax", n_shared=TabNet_params['n_shared'],
                                        scheduler_params=dict(mode="min",
                                                              patience=15,
                                                              min_lr=1e-5,
                                                              factor=0.5,),
                                        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                                        verbose=1,
                                        seed=42
                                        )
                    results_df = clf.trials_dataframe()
                    results_df.to_csv(fname + 'cross_val_scores.csv', index=False, sep=";")
                    clf = TabNetClassifier(**final_params, device_name=DEVICE)
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.2)
                    clf.fit(X_train=X_train,
                            y_train=y_train,
                            eval_set=[(X_val, y_val),
                                      (X_test, y_test)],
                            patience=0,
                            max_epochs=100,
                            eval_metric=['auc'],
                            batch_size=512,
                            virtual_batch_size=32)

                else:
                    clf.fit(X_train, y_train)
                    results_df = pd.DataFrame(clf.cv_results_)
                    results_df = results_df.sort_values(by=["rank_test_score"])
                    results_df.to_csv(fname + 'cross_val_scores.csv', index=False, sep=";")

                y_pred_proba = clf.predict_proba(X_test)[:, 1]
                auc = metrics.roc_auc_score(y_test, y_pred_proba)
                auc = np.round(auc, 5)
                print(auc)
                res = pd.DataFrame({
                    "Classifier":classifier,
                    "timediff": pr,
                    "AUC": auc}, index=[0])
                res.to_csv(fname + 'results.csv', sep=";", index=False)
                dfs = list([])
                for f in glob.glob(os.path.join(predict_path, "*results.csv")):
                     dfs.append(pd.read_csv(f, delimiter=";"))
                res_all = pd.concat(dfs, ignore_index=True)
                res_all.to_csv(os.path.join(predict_path, "r_all.csv"), sep=";", index=False)

