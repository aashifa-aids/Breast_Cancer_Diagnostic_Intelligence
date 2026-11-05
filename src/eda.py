from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def create_eda_outputs(frame, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    sns.countplot(data=frame, x="diagnosis", order=["B", "M"])
    plt.title("Diagnosis Class Distribution")
    plt.xlabel("B = Benign, M = Malignant")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close()

    for feature in ["mean radius", "mean texture", "mean area", "mean concavity"]:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=frame, x="diagnosis", y=feature, order=["B", "M"])
        plt.title(f"{feature.title()} by Diagnosis")
        plt.tight_layout()
        plt.savefig(output_dir / f"{feature.replace(' ', '_')}_boxplot.png", dpi=160)
        plt.close()

    plt.figure(figsize=(15, 12))
    sns.heatmap(frame.drop(columns="diagnosis").corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close()
