import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

comparison_file = Path(
    "reports/model_comparison.csv"
)

alpha_file = Path(
    "reports/alpha_tuning.csv"
)

output_dir = Path(
    "reports/figures"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD RESULTS
# ============================================================

comparison = pd.read_csv(
    comparison_file
)

alpha_results = pd.read_csv(
    alpha_file
)


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def save_bar_chart(
    data,
    x_column,
    y_column,
    title,
    y_label,
    filename
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        data[x_column],
        data[y_column]
    )

    plt.title(
        title,
        fontsize=16
    )

    plt.xlabel(
        "Model",
        fontsize=12
    )

    plt.ylabel(
        y_label,
        fontsize=12
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / filename,
        dpi=300
    )

    plt.close()


# ============================================================
# 4. PRECISION GRAPH
# ============================================================

save_bar_chart(
    comparison,
    "Model",
    "Precision@10",
    "Model Comparison - Precision@10",
    "Precision@10",
    "model_precision.png"
)


# ============================================================
# 5. RECALL GRAPH
# ============================================================

save_bar_chart(
    comparison,
    "Model",
    "Recall@10",
    "Model Comparison - Recall@10",
    "Recall@10",
    "model_recall.png"
)


# ============================================================
# 6. MAP GRAPH
# ============================================================

save_bar_chart(
    comparison,
    "Model",
    "MAP@10",
    "Model Comparison - MAP@10",
    "MAP@10",
    "model_map.png"
)


# ============================================================
# 7. NDCG GRAPH
# ============================================================

save_bar_chart(
    comparison,
    "Model",
    "NDCG@10",
    "Model Comparison - NDCG@10",
    "NDCG@10",
    "model_ndcg.png"
)


# ============================================================
# 8. ALPHA TUNING GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    alpha_results["Alpha"],
    alpha_results["Precision@10"],
    marker="o",
    label="Precision@10"
)

plt.plot(
    alpha_results["Alpha"],
    alpha_results["Recall@10"],
    marker="o",
    label="Recall@10"
)

plt.plot(
    alpha_results["Alpha"],
    alpha_results["MAP@10"],
    marker="o",
    label="MAP@10"
)

plt.plot(
    alpha_results["Alpha"],
    alpha_results["NDCG@10"],
    marker="o",
    label="NDCG@10"
)

plt.title(
    "Hybrid Model - Alpha Tuning",
    fontsize=16
)

plt.xlabel(
    "Alpha (Collaborative Weight)",
    fontsize=12
)

plt.ylabel(
    "Metric Score",
    fontsize=12
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    output_dir / "alpha_performance.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. FINISHED
# ============================================================

print(
    "\n========================================"
)

print(
    "       VISUALIZATION COMPLETE"
)

print(
    "========================================\n"
)

print(
    "Graphs saved in:"
)

print(
    "reports/figures/"
)

print(
    "\nGenerated files:"
)

for file in sorted(
    output_dir.glob("*.png")
):

    print(
        f"✓ {file.name}"
    )