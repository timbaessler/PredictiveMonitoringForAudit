import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
sys.path.append('..')
from src.preprocessing.utils import get_remaining_bus_days, get_event_nr
from src.models.bucketing import StateBucketing
from config.data_config import bpic_2019_dict as bpi_dict
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})
desired_width = 200
pd.set_option('display.width', desired_width)
pd.set_option("display.max_rows", 40)
pd.set_option("display.max_columns", 20)


if __name__ == "__main__":
    res_path = os.path.join(bpi_dict["res_path"])
    log = pd.read_feather(bpi_dict["labelled_log_path"])
    print(log["case:concept:name"].unique())
    log = get_event_nr(log)
    log = log[log.y==1]
    log = log[log.event_nr <= log.pos]
    log = get_remaining_bus_days(log)
    state_list = ["Create Purchase Order Item",
                  "Record Goods Receipt",
                  "Record Invoice Receipt",
                  "Clear Invoice"]
    log_res = pd.DataFrame()
    for i, state in enumerate(state_list):
        print(i, state)
        bucket = StateBucketing(state)
        log2 = bucket.fit_transform(log)
        log2 = log2.groupby("case:concept:name")["remaining bus. hours / 24"].last().reset_index(drop=False)
        log2["state"] = state
        log_res = log_res.append(log2)
    print(log_res[log_res.state == "Clear Invoice"])
    print(log_res[(log_res.state == "Clear Invoice") & (log_res["remaining bus. hours / 24"]==0)])
    print(log_res.groupby("state")["remaining bus. hours / 24"].agg({np.median,
                                                                     np.mean,
                                                                     np.std}))
    print(stats.percentileofscore(log_res[log_res.state == "Clear Invoice"]["remaining bus. hours / 24"], 1))
    print(stats.percentileofscore(log_res[log_res.state == "Clear Invoice"]["remaining bus. hours / 24"], 1/24))
    print(stats.percentileofscore(log_res[log_res.state == "Clear Invoice"]["remaining bus. hours / 24"], 0))
    plt.figure(figsize=(16, 6))
    ax = sns.boxplot(x="state", y="remaining bus. hours / 24", data=log_res)
    plt.rcParams.update({
        "axes.axisbelow":True,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"]})
    ax.yaxis.grid(visible=True)
    ax.set_xlabel("state", fontsize=20)
    ax.set_ylabel("remaining business days", fontsize=20)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(res_path, "rem_time.png"), dpi=500)