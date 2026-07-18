"""Create a 300-DPI adaptive-bin reliability figure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fold_isolated_pipeline import FIGURES, RESULTS


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = pd.read_csv(RESULTS / "nested_evaluation" / "stratified_voting_reliability.csv")
    temporal = pd.read_csv(RESULTS / "nested_evaluation" / "temporal_voting_reliability.csv")
    primary = primary.loc[primary["strategy"] == "adaptive"]
    temporal = temporal.loc[temporal["strategy"] == "adaptive"]
    summaries = {
        "primary": pd.read_csv(RESULTS / "nested_evaluation" / "stratified_model_summary.csv"),
        "temporal": pd.read_csv(RESULTS / "nested_evaluation" / "temporal_model_summary.csv"),
    }
    ece_primary = summaries["primary"].loc[
        summaries["primary"]["model"] == "voting", "ECE_adaptive_10_combined"
    ].iloc[0]
    ece_temporal = summaries["temporal"].loc[
        summaries["temporal"]["model"] == "voting", "ECE_adaptive_10_combined"
    ].iloc[0]
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=300)
    panels = [
        (axes[0], primary, "(a) Nested outer-CV OOF", ece_primary),
        (axes[1], temporal, "(b) 2021–2022 temporal holdout", ece_temporal),
    ]
    for ax, frame, title, ece in panels:
        upper = max(float(frame[["mean_pred", "observed_rate"]].to_numpy().max()) * 1.08, 0.04)
        ax.plot([0, upper], [0, upper], linestyle="--", color="gray", linewidth=1)
        ax.plot(frame["mean_pred"], frame["observed_rate"], marker="o", linewidth=1.6)
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed label rate")
        ax.set_title(f"{title}\nAdaptive ECE={ece:.4f}")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_4_Reliability_Adaptive10Bins_300dpi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
