from sklearn.base import TransformerMixin
import pandas as pd


class StateBucketing(TransformerMixin):
    def __init__(self, state, case_id_col='case:concept:name',
                 activity_col='concept:name',
                 timestamp_col='time:timestamp'
                 ):
        self.state = state
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.timestamp_col=timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        state_dict = dict()
        for case_id, vals in X.groupby([self.case_id_col]):
            events = vals[self.activity_col].tolist()

            for i, event in enumerate(events):
                if event == self.state:
                    state_dict[case_id] = i+1
                    break
        X = X.merge(pd.DataFrame({self.case_id_col: state_dict.keys(),
                                  "state": state_dict.values()}),
                    on=[self.case_id_col], how="left").fillna(0)
        X = X[X.event_nr <= X.state]
        return X


class TimeBucketing(TransformerMixin):
    def __init__(self,
                 timediff,
                 end_of_year=False,
                 unit = None,
                 deadline = None,
                 deadline_col = None,
                 timestamp_col='time:timestamp'
                 ):
        self.timediff = timediff
        self.unit = unit
        self.deadline = deadline
        self.deadline_col = deadline_col
        self.end_of_year = end_of_year
        self.timestamp_col = timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.deadline is not None:
            X["deadline"] = pd.to_datetime(self.deadline) - pd.to_timedelta(self.timediff, unit=self.unit)
            return X[X.self.timestamp_col <= X.deadline].drop(columns=["deadline"])
        elif self.deadline_col is not None:
            X["deadline"] = X[self.deadline_col] - pd.to_timedelta(self.timediff, unit=self.unit)
            return X[X.self.timestamp_col <= X.deadline].drop(columns=["deadline"])
        elif self.end_of_year:
            X["deadline"] = X[self.timestamp_col].dt.dayofyear
            return X[X.deadline <= 365-self.timediff].drop(columns=["deadline"])
        else:
            raise RuntimeError('No deadline specified!')
