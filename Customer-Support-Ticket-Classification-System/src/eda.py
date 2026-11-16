"""Exploratory Data Analysis (Task 2). Saves charts to screenshots/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_clean, ROOT

OUT = ROOT / "screenshots"
OUT.mkdir(exist_ok=True)


def main():
    df = load_clean()
    print("Total tickets:", len(df))
    print("\nTickets per category:\n", df["category"].value_counts())
    print("\nTickets by priority:\n", df["priority"].value_counts())
    print("\nTickets by status:\n", df["status"].value_counts())
    print("\nCategory share (%):\n", (df["category"].value_counts(normalize=True) * 100).round(1))
    df["words"] = df["clean_description"].str.split().str.len()
    print("\nWords per description: mean=%.1f min=%d max=%d" % (df.words.mean(), df.words.min(), df.words.max()))
    print("\nPriority by category:\n", __import__("pandas").crosstab(df.category, df.priority))

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = df["category"].value_counts().index
    sns.countplot(data=df, y="category", order=order, color="#4C78A8", ax=ax)
    ax.set_title("Chart 1: Tickets per category"); ax.set_xlabel("Number of tickets"); ax.set_ylabel("")
    for c in ax.containers: ax.bar_label(c, padding=3)
    fig.tight_layout(); fig.savefig(OUT / "category_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.countplot(data=df, x="category", hue="priority", order=order,
                  hue_order=["High", "Medium", "Low", "Unknown"], ax=ax)
    ax.set_title("Chart 2: Priority distribution by category"); ax.set_xlabel(""); ax.set_ylabel("Number of tickets")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(); fig.savefig(OUT / "priority_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    df["status"].value_counts().plot.bar(color="#72B7B2", ax=ax, rot=0)
    ax.set_title("Chart 3: Tickets by status"); ax.set_ylabel("Number of tickets")
    fig.tight_layout(); fig.savefig(OUT / "status_distribution.png", dpi=150); plt.close(fig)
    print("\nCharts saved to", OUT)


if __name__ == "__main__":
    main()
