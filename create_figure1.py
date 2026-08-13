"""Recreate Figure 1 from the strict-month metadata used for the manuscript."""

import matplotlib.pyplot as plt
import pandas as pd

from fold_isolated_pipeline import FIGURES, STRICT_DATA

META_PATH = STRICT_DATA / "meta_past_only.pkl"
OUTPUT_PATH = FIGURES / "Figure_1_Exploratory_Data_Analysis_v18_corrected.png"


def main() -> None:
    meta = pd.read_pickle(META_PATH)

    total_n = len(meta)
    risk_n = int(meta["Label"].sum())
    overall_rates = meta["Label"].value_counts(normalize=True).reindex([0, 1]) * 100

    by_test = meta.groupby("Test")["Label"].agg(["size", "mean"]).reindex(["A", "B"])
    by_test["rate"] = by_test["mean"] * 100

    by_year = meta.groupby("TestDate_year")["Label"].mean().mul(100).sort_index()

    tests_per_person = meta.groupby("PrimaryKey").size()
    repeat_record_rate = meta.duplicated("PrimaryKey", keep=False).mean() * 100
    test_types_per_person = meta.groupby("PrimaryKey")["Test"].nunique()
    both_test_rate = test_types_per_person.gt(1).mean() * 100

    repeat_counts = pd.Series(
        {
            "1": int(tests_per_person.eq(1).sum()),
            "2": int(tests_per_person.eq(2).sum()),
            "3": int(tests_per_person.eq(3).sum()),
            "4+": int(tests_per_person.ge(4).sum()),
        }
    )
    repeat_pct = repeat_counts / repeat_counts.sum() * 100

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    (ax1, ax2), (ax3, ax4) = axes

    for ax in axes.flat:
        ax.grid(True, color="#b0b0b0", alpha=0.35, linewidth=1)
        ax.set_axisbelow(True)

    # (a) Class imbalance
    bars = ax1.bar(
        ["Non-Risk (0)", "Risk (1)"],
        overall_rates.values,
        color=["#aeb7c8", "#c44e52"],
        width=0.6,
    )
    ax1.set_title("(a) Class Imbalance", pad=12)
    ax1.set_ylabel("Share (%)")
    ax1.set_ylim(0, 108)
    for bar, value in zip(bars, overall_rates.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=16,
        )
    ax1.text(
        0.98,
        24,
        f"n(Risk)={risk_n:,}\nN={total_n:,}",
        ha="center",
        va="center",
        fontsize=15,
        color="#333333",
    )

    # (b) Risk rate by test type
    bars = ax2.bar(
        ["A (new)", "B (renewal)"],
        by_test["rate"].values,
        color=["#4c72b0", "#c44e52"],
        width=0.6,
    )
    ax2.set_title("(b) Risk Rate by Test Type", pad=12)
    ax2.set_ylabel("Risk-Group Rate (%)")
    ax2.set_ylim(0, 5.4)
    for bar, test, value in zip(bars, by_test.index, by_test["rate"].values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value:.2f}%\n(n={int(by_test.loc[test, 'size']):,})",
            ha="center",
            va="bottom",
            fontsize=15,
            linespacing=0.95,
        )

    # (c) Risk rate by test year
    ax3.plot(
        by_year.index,
        by_year.values,
        color="#4c72b0",
        linewidth=3,
        marker="o",
        markersize=9,
    )
    ax3.set_title("(c) Risk Rate by Test Year", pad=12)
    ax3.set_xlabel("Test Year")
    ax3.set_ylabel("Risk-Group Rate (%)")
    ax3.set_xticks(by_year.index)
    ax3.set_ylim(2.2, 7.0)

    # (d) Repeated testing per individual
    bars = ax4.bar(
        repeat_counts.index,
        repeat_counts.values,
        color="#55a868",
        width=0.6,
    )
    ax4.set_title("(d) Repeated Testing per Individual", pad=12)
    ax4.set_xlabel("Tests per Individual")
    ax4.set_ylabel("Number of Individuals")
    ax4.set_ylim(0, 725_000)
    for bar, value in zip(bars, repeat_pct.values):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 9_000,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=15,
        )
    ax4.text(
        0.96,
        0.58,
        f"{repeat_record_rate:.1f}% of records from\n"
        "repeat testees\n"
        f"({both_test_rate:.1f}% of individuals took\n"
        "both A & B)",
        transform=ax4.transAxes,
        ha="right",
        va="center",
        fontsize=14,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#f5f5f5", "ec": "#aaaaaa"},
    )

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.955, wspace=0.28, hspace=0.40)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(OUTPUT_PATH)
    print(f"size={OUTPUT_PATH.stat().st_size:,} bytes")
    print(
        "verified values: "
        f"records={total_n:,}, risk={risk_n:,}, repeat_record_rate={repeat_record_rate:.6f}%, "
        f"both_test_rate={both_test_rate:.6f}%"
    )


if __name__ == "__main__":
    main()
