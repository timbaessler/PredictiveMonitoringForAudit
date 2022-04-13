import pandas as pd
from pm4py.algo.filtering.pandas.timestamp import timestamp_filter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter


def read_xes(file: str):
    df = xes_importer.apply(file)
    df = log_converter.apply(df, variant=log_converter.Variants.TO_DATA_FRAME)
    return df


def count_traces(df: pd.DataFrame, case_id_col='case:concept:name') -> int:
    return len(df[case_id_col].unique())


def get_activity_count(df: pd.DataFrame, event_name: str, case_id_col='case:concept:name',
                       activity_col="concept:name") -> pd.DataFrame:
    df = df.merge(df.groupby([case_id_col])[activity_col]
                  .value_counts().unstack(fill_value=0
                                          ).loc[:, event_name].reset_index()
                  .rename(columns={event_name: "Count " + event_name}),
                  on=[case_id_col], how="left")
    return df


def get_event_duration(log: pd.DataFrame, case_id_col='case:concept:name',
                       timestamp_col='time:timestamp'):
    log[timestamp_col] = log[timestamp_col].dt.tz_localize(None)
    log["duration"] = log.groupby(case_id_col)[timestamp_col].diff().dt.total_seconds().shift(-1)
    log["duration"] = (log.duration).astype(float) / 3600
    return log


def get_time_since_last_event(log: pd.DataFrame, case_id_col='case:concept:name',
                              timestamp_col='time:timestamp'):
    log[timestamp_col] = log[timestamp_col].dt.tz_localize(None)
    log["time_since_last_event"] = (log.groupby(case_id_col)[timestamp_col].diff()).dt.total_seconds().fillna(0)
    return log


def get_time_since_first_event(log: pd.DataFrame, case_id_col='case:concept:name',
                               timestamp_col='time:timestamp'):
    log[timestamp_col] = log[timestamp_col].dt.tz_localize(None)
    if not 'time_since_last_event' in log.columns.tolist():
        log = get_time_since_last_event(log)
    log["time_since_first_event"] =log.groupby(case_id_col)["time_since_last_event"].apply(lambda x: x.cumsum())
    return log


def get_cumulative_duration(log: pd.DataFrame, case_id_col='case:concept:name',
                            timestamp_col='time:timestamp'):
    dur = False
    if not "duration" in log.columns.tolist():
        log = get_event_duration(log, case_id_col=case_id_col, timestamp_col=timestamp_col)
    log["cumulative_duration"] = log.groupby(case_id_col)["duration"].apply(lambda x: x.cumsum())
    if dur:
        log = log.drop(columns=["duration"])
    return log


def get_total_duration(log: pd.DataFrame, case_id_col='case:concept:name',
                       timestamp_col='time:timestamp'):
    dur = False
    if not "duration" in log.columns.tolist():
        log = get_event_duration(log, case_id_col=case_id_col, timestamp_col=timestamp_col)
        dur = True
    log["total_duration"] = log.groupby(case_id_col)["duration"].transform('sum')
    if dur:
        log = log.drop(columns=["duration"])
    return log


def get_time_attributes(log: pd.DataFrame,  timestamp_col='time:timestamp'):
    log[timestamp_col] = log[timestamp_col].dt.tz_localize(None)
    log["year"] = log[timestamp_col].dt.year
    log["month"] = log[timestamp_col].dt.month
    log["weekday"] = log[timestamp_col].dt.weekday
    log["hour"] = log[timestamp_col].dt.hour
    log["dayofyear"] = log[timestamp_col].dt.dayofyear
    return log


def get_seq_length(log: pd.DataFrame, case_id_col='case:concept:name'):
    log = log.merge(log.groupby(case_id_col).size().reset_index().rename(columns={0: "trace_length"}),
                    on=[case_id_col], how="left")
    return log


def get_event_nr(log:pd.DataFrame, case_id_col='case:concept:name'):
    log["event_nr"] = 1
    log["event_nr"] = log.groupby([case_id_col])["event_nr"].cumsum()
    return log


def get_remaining_time(log: pd.DataFrame, case_id_col='case:concept:name',
                       timestamp_col='time:timestamp'):
    if not "duration" in log.columns.tolist():
        log = get_event_duration(log, case_id_col=case_id_col, timestamp_col=timestamp_col)
    log["remaining_time"] = (log[::-1].groupby(case_id_col)["duration"].cumsum().fillna(0)).astype(float)
    log["remaining_time"] = log["remaining_time"] / 24
    return log


def timefilter(df: pd.DataFrame, start, end, timestamp_col='time:timestamp') -> pd.DataFrame:
    df = timestamp_filter.filter_traces_intersecting(df, start, end)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)
    return df


def filter_by_min_activity(log: pd.DataFrame, activity: str, min: int) -> pd.DataFrame:
    log = get_activity_count(log, activity)
    log = log[log["Count " + activity] >= min]
    log = log.drop(columns=["Count " + activity])
    return log


def timefilter_start(log: pd.DataFrame, start, end=None, case_id_col='case:concept:name',
                     timestamp_col='time:timestamp') -> pd.DataFrame:
    log[timestamp_col] = log[timestamp_col].dt.tz_localize(None)
    log["start_of_case"] = log.groupby(case_id_col)[timestamp_col].transform("first")
    if end is None:
        log = log[log["start_of_case"]>=pd.to_datetime(start)]
    else:
        log = log[(log["start_of_case"]>=pd.to_datetime(start))&(log["start_of_case"]<=pd.to_datetime(end))]
    log = log.drop(columns=["start_of_case"])
    return log


def timefilter_end(log: pd.DataFrame, start, end, case_id_col='case:concept:name',
                   timestamp_col='time:timestamp') -> pd.DataFrame:
    log[timestamp_col] = log[timestamp_col].dt.tz_localize(None)
    log["end_of_case"] = log.groupby(case_id_col)[timestamp_col].transform("last")
    log = log[(log["end_of_case"]>=pd.to_datetime(start))
              &(log["end_of_case"]<=pd.to_datetime(end))]
    log = log.drop(columns=["end_of_case"])
    return log


