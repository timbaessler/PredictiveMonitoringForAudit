import os
import sys
sys.path.append('..')
from config.data_config import bpic_2018_dict as bpi_dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from autorank import autorank, create_report, plot_stats, latex_table


plt.rcParams.update({'text.usetex':True,
                     'font.size':11,
                     'font.family':'serif'
                     })



if __name__ == "__main__":
    predict_path = bpi_dict["predict_path"]
    fname = os.path.join(predict_path, "r_all_bpic2018.csv") # Results DF
    df = pd.read_csv(fname, delimiter=";")
    df = df[df.timediff<15]
    df = df.sort_values(by=["Classifier", "timediff"])
    sns.set_palette("Paired")
    df.to_excel(os.path.join(predict_path, "res_df_all.xlsx"), index=False)
    # Results Plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="timediff", y="AUC", hue="Classifier", marker='o', markersize=8, linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Business days until deadline")
    plt.ylabel("AUC")
    plt.title("AUC Trends Over Time by Classifier")
    plt.tight_layout()
    plt.savefig(os.path.join(predict_path,"results.png"), dpi=700)
    # Perform the Friedman Test
    classifiers = df.Classifier.unique().tolist()
    df = df.pivot(index="timediff", columns="Classifier", values="AUC")
    results = autorank(df)
    plot_stats(results)
    plt.show()