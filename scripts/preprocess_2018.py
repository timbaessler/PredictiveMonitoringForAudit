import os
import sys
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.append('..')
import gzip
import shutil
from src.preprocessing.utils import *
from config.data_config import bpic_2018_dict as bpi_dict


def check_late_payment(log, case_id_col="case:concept:name", event_col="concept:name"):
    pos_dict = dict()
    label_dict = dict()
    for case_id, vals in log.groupby([case_id_col]):
        events = vals[event_col].tolist()
        if 'begin payment' in events:
            if'abort payment' in events:
                pay_idx = events[::-1].index('begin payment')
                ab_idx = events[::-1].index('abort payment')
                if ab_idx < pay_idx:
                    label_dict[case_id] = 1
                    pos_dict[case_id] = len(events)
                else:
                    label_dict[case_id]=0
                    pay_idx = events[::-1].index('begin payment')
                    pos_dict[case_id] = len(events) - pay_idx
            else:
                label_dict[case_id]=0
                pay_idx = events[::-1].index('begin payment')
                pos_dict[case_id] = len(events) - pay_idx
        else:
            label_dict[case_id] = 1
            pos_dict[case_id] = len(events)
    log = log.merge(pd.DataFrame({case_id_col: label_dict.keys(),
                                  "y": label_dict.values()}),
                    on=[case_id_col], how="left")
    log = log.merge(pd.DataFrame({case_id_col: pos_dict.keys(),
                                  "pos": pos_dict.values()}),
                    on=[case_id_col], how="left")
    return log


if __name__ == "__main__":
    bpi_path = bpi_dict["bpi_path"]
    processed_path = bpi_dict["processed_path"]
    with gzip.open(os.path.join(bpi_path, "raw",  "BPI Challenge 2018.xes.gz"), "rb") as f_in, \
            open(os.path.join(bpi_path, "raw",  "BPI Challenge 2018.xes"), "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    if os.path.exists(os.path.join(bpi_path, "raw", "bpi.feather")):
        log = pd.read_feather(os.path.join(bpi_path, "raw", "bpi.feather"))
    else:
        log = read_xes(os.path.join(bpi_path, "raw",  "BPI Challenge 2018.xes"))
        log.to_feather(os.path.join(bpi_path, "raw", "bpi.feather"))
    log = log[log.doctype=="Payment application"]
    log = get_time_attributes(log)
    log["case:concept:name"] = log["case:concept:name"] + log["case:year"].astype(str)
    log["case:year"] = log["case:year"].astype(int)
    log = log[log.year == log["case:year"]]
    log = get_time_since_last_event(log)
    log = get_time_since_first_event(log)
    log = get_event_nr(log)
    log = get_event_duration(log)
    log = get_remaining_time(log)
    log = get_total_duration(log)
    log = check_late_payment(log)
    log = get_seq_length(log).reset_index(drop=True)
    log = log.merge(pd.DataFrame({
        "year" : [2015, 2016, 2017], "deadline": [
            pd.to_datetime("2015-12-23"),
            pd.to_datetime("2016-12-23"),
            pd.to_datetime("2017-12-22")]}), on=["year"], how="left")
    log.to_feather(os.path.join(processed_path, 'bpic2018_labelled.feather'))
