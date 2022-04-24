import sys
import warnings
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
sys.path.append('..')
from src.preprocessing.utils import get_remaining_time, get_event_nr
from src.models.bucketing import StateBucketing
from config.data_config import bpic_2019_dict as bpi_dict

desired_width = 200
pd.set_option('display.width', desired_width)
pd.set_option("display.max_rows", 40)
pd.set_option("display.max_columns", 20)


if __name__ == "__main__":
    res_path = os.path.join(bpi_dict["res_path"])
    log = pd.read_feather(bpi_dict["labelled_log_path"])
    log = get_event_nr(log)
    log = log[log.y==1]
    log = log[log.event_nr <= log.pos]
    log = get_remaining_time(log)
    state_list = ["Create Purchase Order Item",
                  "Record Goods Receipt",
                  "Record Invoice Receipt",
                  "Clear Invoice"]
    log_res = pd.DataFrame()
    for i, state in enumerate(state_list):
        bucket = StateBucketing(state)
        log2 = bucket.fit_transform(log)
        log2 = log2.groupby("case:concept:name").remaining_time.last().reset_index(drop=False)
        log2 = log2.rename(columns={"remaining_time": "remaining time in days"})
        log2["state"] = state
        log_res = log_res.append(log2)
    plt.figure(figsize=(16, 6))
    sns.set_theme(style="whitegrid")
    sns.boxplot(x="state", y="remaining time in days", data=log_res)
    plt.savefig(os.path.join(res_path, "rem_time.png"), dpi=500)