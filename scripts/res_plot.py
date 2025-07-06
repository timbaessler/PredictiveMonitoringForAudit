import os
import sys
sys.path.append('..')
from config.data_config import bpic_2018_dict as bpi_dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from autorank import autorank, create_report, plot_stats, latex_table


plt.rcParams.update({
    'text.usetex': False,  # Disable LaTeX to use system-installed fonts
    'font.size': 14,
    'font.family': 'Times New Roman'  # Ensure this font is installed on your system
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
    linestyles = {
        "LogisticRegression": (0, (1, 5)),  # dotted
        "RandomForest": (0, (5, 5)),        # dashed
        "TabNet": (0, (3, 1, 1, 1)),        # dash-dot
        "XGBoost": "solid"                 # solid
    }
    
    markers = {
        "LogisticRegression": "o",
        "RandomForest": "s",
        "TabNet": "^",
        "XGBoost": "D"
    }
    
    # Plot each classifier manually with distinct styles
    fig, ax = plt.subplots(figsize=(10, 5))
    for clf in df["Classifier"].unique():
        df_clf = df[df["Classifier"] == clf]
        sns.lineplot(
            data=df_clf, x="timediff", y="AUC", label=clf, ax=ax,
            linestyle=linestyles[clf],
            marker=markers[clf],
            linewidth=2,
            markersize=8
        )
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
