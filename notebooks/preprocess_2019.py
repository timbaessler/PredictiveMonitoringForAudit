import os
import sys
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
sys.path.append('..')
from src.preprocessing.utils import *
from config.data_config import bpic_2019_dict as bpi_dict


def label_log(log, case_id_col="case:concept:name", activity_col="concept:name", timestamp_col="time:timestamp"):
    pos_dict = dict()
    label_dict = dict()
    violations_counter = 0
    case_counter = 0
    for case_id, vals in log.groupby([case_id_col]):
        case_counter +=1
        events = vals[activity_col].tolist()
        invoice_count = 0
        clear_count = 0
        clear_pos_1 = 0
        conformance = True
        for i, event in enumerate(events):
            if event == 'Record Invoice Receipt':
                invoice_count += 1
            if event == 'Clear Invoice':
                invoice_count -= 1
                clear_count +=1
                if clear_count == 1:
                    clear_pos_1 = i + 1
                if invoice_count < 0 and clear_count>0:
                    label_dict[case_id] = 1
                    pos_dict[case_id] = i + 1
                    conformance = False
                    violations_counter += 1
                    continue
            if i == len(events)-1 and conformance:
                label_dict[case_id] = 0
                pos_dict[case_id] = clear_pos_1
    print(str(violations_counter)+' violations '
          +str(round((violations_counter/case_counter)*100, 2))+" % of all cases")
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
    if os.path.exists(os.path.join(bpi_path, "raw", "bpi.feather")):
        log = pd.read_feather(os.path.join(bpi_path, "raw", "bpi.feather"))
    else:
        log = read_xes(os.path.join(bpi_path, "raw", 'BPI_Challenge_2019.xes'))
        log.to_feather(os.path.join(bpi_path, "raw", "bpi.feather"))
    log = timefilter_start(log, "2018-01-01 00:00:00")
    log = filter_by_min_activity(log, 'Clear Invoice', 1)
    log = log[log["case:Item Category"] == "3-way match, invoice before GR"]
    log = label_log(log)
    log = get_time_since_last_event(log)
    log = get_time_since_first_event(log)
    log = get_time_attributes(log)
    log = get_event_nr(log)
    log = get_event_duration(log)
    log = get_total_duration(log)
    log = get_remaining_time(log)
    log = get_seq_length(log)
    log.reset_index(drop=True).to_feather(os.path.join(processed_path, 'bpic_2019_labelled.feather'))