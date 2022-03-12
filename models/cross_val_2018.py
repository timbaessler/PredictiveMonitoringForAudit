import os
import pickle
import glob
from cv_models import CrossValidation
from encoding import *
from optim import *


if __name__ == "__main__":
    processed_path = os.path.join('<labeled log path>')
    predict_path = os.path.join('<results path>')
    log = pd.read_feather(os.path.join(processed_path, 'bpic2018_labelled.feather'))
    log["case:year"] = log["case:year"].astype(int)
    train_idx = log[(log["case:year"]==2015)|(log["case:year"]==2016)
                    ]["case:concept:name"].unique().tolist()
    test_idx = log[(log["case:year"]==2017)]["case:concept:name"].unique().tolist()
    cv1 = log[(log["case:year"]==2015)]["case:concept:name"].unique().tolist()
    cv2 = log[(log["case:year"]==2016)]["case:concept:name"].unique().tolist()
    static_cat_cols = list(['case:young farmer',
                            'case:penalty_AJLP',
                            'case:small farmer',
                            'case:penalty_BGP',
                            'case:department',
                            'case:penalty_C16',
                            'case:penalty_BGK',
                            'case:penalty_AVUVP',
                            'case:penalty_CC',
                            'case:penalty_AVJLP',
                            'case:penalty_C9',
                            'case:cross_compliance',
                            'case:rejected',
                            'case:penalty_C4',
                            'case:penalty_AVGP',
                            'case:penalty_ABP',
                            'case:penalty_B6',
                            'case:penalty_B4',
                            'case:penalty_B5',
                            'case:penalty_AVBP',
                            'case:penalty_B2',
                            'case:selected_risk',
                            'case:penalty_B3',
                            'case:selected_manually',
                            'case:penalty_AGP',
                            'case:penalty_B16',
                            'case:penalty_GP1',
                            'case:basic payment',
                            'case:penalty_B5F',
                            'case:penalty_V5',
                            'case:payment_actual0',
                            'case:redistribution',
                            'case:penalty_JLP6',
                            'case:penalty_JLP7',
                            'case:penalty_JLP5',
                            'case:penalty_JLP2',
                            'case:penalty_JLP3',
                            'case:penalty_JLP1'])

    dynamic_cat_cols = list(['org:resource', 'concept:name', 'success',
                             'doctype', 'subprocess', 'note',
                             ])
    num_cols = list(['case:penalty_amount0', 'month', 'weekday', 'hour',
                     'time_since_first_event',
                     'time_since_last_event',
                     'case:amount_applied0',
                     'case:number_parcels',
                     'case:area'])

    res = pd.DataFrame()

    for classifier in list(["XGBoost", "RandomForest"]):
        for pr in range(32):
            fname = os.path.join(predict_path, classifier +str(pr).zfill(2)+ "_")
            if os.path.exists(fname + ".sav"):
                continue
            log2 = log[log.dayofyear<=365-pr]
            agg_transformer = Aggregation(num_cols,
                                          static_cat_cols,
                                          dynamic_cat_cols)
            Xpr, ypr = agg_transformer.transform(log2)
            X_train = Xpr[Xpr.index.isin(train_idx)].reset_index(drop=False)
            y_train = ypr[Xpr.index.isin(train_idx)].reset_index(drop=False)
            cv1_idx = X_train[X_train["case:concept:name"].isin(cv1)].index.tolist()
            cv2_idx = X_train[X_train["case:concept:name"].isin(cv2)].index.tolist()
            cvs = [(cv1_idx, cv2_idx), (cv2_idx, cv1_idx)]
            onehot = True
            agg_transformer = Aggregation(num_cols, static_cat_cols, dynamic_cat_cols, one_hot_static=onehot)
            X, y = agg_transformer.fit_transform(log2)
            del log2
            X_train = X[X.index.isin(train_idx)].values
            y_train = y[y.index.isin(train_idx)].values
            X_test = X[X.index.isin(test_idx)].values
            y_test = y[y.index.isin(test_idx)].values
            del X
            crossval = CrossValidation(classifier)
            clf = crossval.get_classifier()
            clf.fit(X_train, y_train)
            pickle.dump(clf, open(fname + ".sav", 'wb'))

            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            y_pred_proba_train = clf.predict_proba(X_train)[:, 1]
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            auc = np.round(auc, 5)

            opt = ThreshholdOptimizer(r=2, p=1)
            opt.fit(y_true=y_train, y_pred_proba=y_pred_proba_train)
            opt.plot_threshold()
            y_pred_proba_opt = opt.predict(y_pred_proba)
            results_df = pd.DataFrame(clf.cv_results_)
            results_df = results_df.sort_values(by=["rank_test_score"])
            results_df.to_csv(fname + 'cross_val_scores.csv', index=False, sep=";")

            res = pd.DataFrame({
                "Classifier": classifier,
                "timediff": pr,
                "AUC": auc}, index=[0])
            res.to_csv(fname + 'results.csv', sep=";", index=False)
            res_all = pd.DataFrame()
            for f in glob.glob(os.path.join(predict_path, "*results.csv")):
                res_all = res_all.append(pd.read_csv(f, delimiter=";"))
            res_all.to_csv(os.path.join(predict_path, "r_all_bpic2018.csv"), sep=";", index=False)

