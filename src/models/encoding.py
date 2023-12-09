import pandas as pd
from sklearn.base import TransformerMixin
import numpy as np
import sys

class Aggregation(TransformerMixin):
    def __init__(self, num_cols,  static_cat_cols, dynamic_cat_cols, case_id_col, activity_col, timestamp_col, one_hot_static=False):

        self.num_cols = num_cols
        self.static_cat_cols = static_cat_cols
        self.dynamic_cat_cols = dynamic_cat_cols
        self.one_hot_static = one_hot_static
        self.caseid = case_id_col
        self.activity = activity_col
        self.timecol = timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.num_cols] = X[self.num_cols].fillna(0)
        X[self.static_cat_cols] = X[self.static_cat_cols].astype('category').apply(lambda x: x.cat.codes).astype('category')
        X_num = X.groupby(self.caseid)[self.num_cols].agg(
            [np.mean,np.max,np.min,np.sum,np.std]).fillna(0)
        self.X_num_cols = X_num.columns.tolist()
        X_cat_dyn = pd.concat([X[self.caseid],
                               pd.get_dummies(X[self.dynamic_cat_cols],
                                              dummy_na=True)], axis=1)
        X_cat_dyn = X_cat_dyn.groupby(self.caseid).agg(np.sum)
        self.X_cat_dyn_cols = X_cat_dyn.columns.tolist()
        if self.one_hot_static:
            X_cat_stat = pd.concat([X[self.caseid],
                                    pd.get_dummies(X[self.static_cat_cols],
                                                   dummy_na=True)], axis=1)
            X_cat_stat = X_cat_stat.groupby(self.caseid).first()
        else:
            X_cat_stat = X.groupby(self.caseid)[self.static_cat_cols].first().astype('category')
        self.X_cat_stat_cols = X_cat_stat.columns.tolist()
        y = X.groupby(self.caseid)['y'].first()
        X_num.columns = ['_'.join(col) if type(col) is tuple else col for col in X_num.columns.values]
        X_num_cols = X_num.columns
        X = X_num.join(X_cat_dyn).join(X_cat_stat)
        del X_cat_dyn, X_num
        X.columns = ['_'.join(col) if type(col) is tuple else col for col in X.columns.values]
        self.cat_indeces = list([i for i in range(len(X.columns)-len(self.static_cat_cols), len(X.columns))])
        self.X_cols = X.columns.tolist()
        return X, y, X_num_cols
