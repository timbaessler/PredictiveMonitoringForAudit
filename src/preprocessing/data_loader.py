import os
import sys
import pandas as pd
from src.models.bucketing import TimeBucketing
from src.models.encoding import Aggregation
import pprint

def load_train_test(log, filename, num_cols, static_cat_cols, dynamic_cat_cols, pr, train_idx, test_idx,
                    read=False, tabnet_features=False,
                    case_id_col='case:concept:name', activity_col='concept:name',
                    timestamp_col='time:timestamp', resource_col='org:resource', deadline_col='deadline'):
    if os.path.exists(filename+"X_train.feather") and read:
        X = pd.read_feather(filename+"X.feather")
        y = pd.read_feather(filename+"y.feather")
        X_train = pd.read_feather(filename+"X_train.feather")
        X_test = pd.read_feather(filename+"X_test.feather")
        y_train = pd.read_feather(filename+"y_train.feather")
        y_test = pd.read_feather(filename+"y_test.feather")
        X, y = X.set_index(case_id_col), y.set_index(case_id_col)
    else:
        bucketer = TimeBucketing(offset=pr, deadline_col=deadline_col, timestamp_col=timestamp_col)
        log2 = bucketer.fit_transform(log)
        agg_transformer = Aggregation(case_id_col=case_id_col,
                                      activity_col=activity_col,
                                      timestamp_col=timestamp_col,
                                      num_cols=num_cols,
                                      static_cat_cols=static_cat_cols,
                                      dynamic_cat_cols=dynamic_cat_cols,
                                      one_hot_static=True)
        X, y, Xnum_cols = agg_transformer.fit_transform(log2)
        X, y = X.reset_index(drop=False), y.reset_index(drop=False)
        #X.to_feather(filename+"X.feather"), X.to_csv(filename+"X.csv", sep=";")
        #y.to_feather(filename+"y.feather"), y.to_csv(filename+"y.csv", sep=";")
        X, y = X.set_index(case_id_col), y.set_index(case_id_col)
        # X_train
        X_train = X[X.index.isin(train_idx)].reset_index(drop=False)
        #X_train[Xnum_cols] = (X_train[Xnum_cols] - X_train[Xnum_cols].mean()) / X_train[Xnum_cols].std()
        #X_train.to_feather(filename+"X_train.feather"), X_train.to_csv(filename+"X_train.csv", sep=';')
        # X_test
        X_test = X[X.index.isin(test_idx)].reset_index(drop=False)
        #X_test[Xnum_cols] = (X_test[Xnum_cols]- X_train[Xnum_cols].mean()) / X_train[Xnum_cols].std()
        #X_test.to_feather(filename+"X_test.feather"), X_test.to_csv(filename+"X_test.csv", sep=';')
        # y_train
        y_train = y[y.index.isin(train_idx)].astype(int).reset_index(drop=False)
        #y_train.to_feather(filename+"y_train.feather"), y_train.to_csv(filename+"y_train.csv", sep=';')
        # y_test
        y_test = y[y.index.isin(test_idx)].astype(int).reset_index(drop=False)
        #y_test.to_feather(filename+"y_test.feather"), y_test.to_csv(filename+"y_test.csv", sep=';')
        del log2
    X_train = X_train.drop(columns=case_id_col)
    X_test = X_test.drop(columns=case_id_col)
    y_train = y_train.drop(columns=case_id_col)
    y_test = y_test.drop(columns=case_id_col)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    if tabnet_features:
        cat_idxs = [X.columns.get_loc(col) for col in static_cat_cols]
        cat_dims = [X[col].nunique() for col in static_cat_cols]
        cat_emb_dim = [cat//2+1 for cat in cat_dims]
        grouped_features = list()
        list_list = list()
        Xcols = X_train.columns.tolist()
        org_resource = False
        concept_name = False
        for i in range(len(Xcols)):
            if i < len(num_cols)*5:
                list_list.append(i)
                if len(list_list) == 5:
                    grouped_features.append(list_list)
                    list_list = list()
            else:
                if Xcols[i].startswith(resource_col):  # Use variable for "org:resource"
                    org_resource = True
                    list_list.append(i)
                elif org_resource:
                    grouped_features.append(list_list)
                    list_list = list()
                    org_resource=False
                elif Xcols[i].startswith(activity_col):  # Use variable for "concept:name"
                    concept_name = True
                    list_list.append(i)
                elif concept_name:
                    grouped_features.append(list_list)
                    concept_name=False
        del X, y
        return X_train, X_test, y_train, y_test, cat_idxs, cat_dims, cat_emb_dim, grouped_features
    else:
        del X, y
        return X_train, X_test, y_train, y_test

