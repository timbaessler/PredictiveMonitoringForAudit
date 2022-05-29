import sys
import warnings
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
sys.path.append('..')
from src.preprocessing.utils import get_remaining_time, get_event_nr
from src.models.bucketing import TimeBucketing
from config.data_config import bpic_2018_dict as bpi_dict
desired_width = 200
pd.set_option('display.width', desired_width)
pd.set_option("display.max_rows", 40)
pd.set_option("display.max_columns", 20)


if __name__ == "__main__":
    log = pd.read_feather(bpi_dict["labelled_log_path"])
    log = log[log.y == 0]
    log = log[log.event_nr <= log.pos]
    log["business days until deadline"] = np.busday_count([d.date() for d in log["deadline"]],
                                                 [d.date() for d in log["time:timestamp"]])
    log = log.groupby(["case:concept:name"]).agg({"business days until deadline": "last",
                                                 "year": "last"}).reset_index(drop=False)
    print(log.groupby("year")["business days until deadline"].value_counts())
    plt.figure(figsize=(16, 6))
    ax = sns.histplot(data=log, x="business days until deadline", hue="year", kde=False)
    plt.setp(ax.get_legend().get_texts(), fontsize='20')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='20')
    ax.tick_params(labelsize=14)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"]})
    ax.set_xlabel("business days until deadline", fontsize=20)
    ax.set_ylabel("count of cases", fontsize=20)
    plt.title("Last 'begin payment' of compliant cases", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(bpi_dict["res_path"], "last_begin_payment.png"), dpi=500)