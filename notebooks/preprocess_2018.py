import os
import sys
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.append('..')
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
                    pos_dict[case_id] = len(events)

            else:
                label_dict[case_id]=0
                pay_idx = events[::-1].index('begin payment')
                pos_dict[case_id] = len(events)
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
    log = read_xes(os.path.join(bpi_path, "raw",  "BPI Challenge 2018.xes"))
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
    #log.to_feather(os.path.join(processed_path, 'bpic2018_labelled.feather'))
