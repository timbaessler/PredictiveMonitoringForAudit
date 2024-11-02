import sys
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('..')
from config.data_config import bpic_2018_dict as bpi_dict
from config.data_config import syn_dict as bpi_dict2

# Common plotting settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 11,
    'font.family': 'serif'
})


def plot_bpic2018_distribution(log):
    """Plot distribution for BPIC2018 dataset"""
    # Data preparation
    log = log[log.y == 0]
    log = log[log.event_nr <= log.pos]
    log["business_days_until_deadline"] = np.busday_count(
        [d.date() for d in log["deadline"]],
        [d.date() for d in log["time:timestamp"]]
    ) * -1

    log = log.groupby(["case:concept:name"]).agg({
        "business_days_until_deadline": "last",
        "year": "last"
    }).reset_index(drop=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(15, 6))

    sns.histplot(
        data=log,
        x="business_days_until_deadline",
        hue="year",
        kde=False,
        color='#2878B5',
        ax=ax
    )

    # Customize plot
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Business Days Until Deadline", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)
    ax.set_title("Distribution of Last Payment Activities\nBefore Deadline (BPIC2018)",
                 fontsize=16, pad=20)

    # Adjust legend
    if ax.get_legend():
        plt.setp(ax.get_legend().get_texts(), fontsize='12')
        plt.setp(ax.get_legend().get_title(), fontsize='12')

    plt.tight_layout()
    return fig

def plot_synlog_distribution(log):
    """Plot distribution for SynLog dataset with custom bins"""
    # Data preparation
    second_approvals = log[
        (log['Activity'] == 'Second Approval Complete') &
        (~log['Late'])
        ].copy()

    second_approvals['business_days_until_deadline'] = second_approvals.apply(
        lambda row: len(pd.date_range(row['Timestamp'], row['deadline'],
                                      freq='B')) - 1, axis=1
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(15, 6))

    # Create custom bins from 0 to maximum value, in steps of 2
    max_days = second_approvals['business_days_until_deadline'].max()
    bins = np.arange(0, max_days + 2, 1)  # +2 to include the last bin

    sns.histplot(
        data=second_approvals,
        x='business_days_until_deadline',
        bins=bins,
        color='#2878B5',
        ax=ax
    )

    # Customize plot
    ax.tick_params(labelsize=14)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}"))  # Integer formatting
    ax.set_xlabel("Business Days Until Deadline", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)
    ax.set_title("Distribution of Second Approvals\nBefore Deadline (SynLog)",
                 fontsize=16, pad=20)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    for data_set in ["BPIC2018", "SynLog"]:
        bpi_dict = bpi_dict2.copy() if data_set == "SynLog" else bpi_dict
        log = pd.read_feather(bpi_dict["labelled_log_path"])
        if data_set == "BPIC2018":
            plot_bpic2018_distribution(log)
            plt.savefig(os.path.join(bpi_dict["res_path"], "last_begin_payment.png"), dpi=500)
        elif data_set == "SynLog":
            plot_synlog_distribution(log)
            plt.savefig(os.path.join(bpi_dict["res_path"], "last_begin_payment_syn.png"), dpi=500)
        else:
            raise RuntimeError("no data set defined")