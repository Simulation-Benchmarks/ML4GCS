from pathlib import Path
import os
import tempfile

matplotlib_cache_dir = Path(tempfile.gettempdir()) / "ml4gcs_matplotlib"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

fontsize = 28
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=fontsize)
plt.rcParams.update(
    {
        "text.latex.preamble": (
            r"\usepackage{bm}"
            r"\usepackage{amsmath}"
            r"\usepackage{mathrsfs}"
            r"\usepackage{amsfonts}"
        )
    }
)
matplotlib.rcParams["axes.linewidth"] = 1.5


def load_test_predictions(results_dir):
    """Load the test targets and predictions written by postprocess.py."""
    predictions_path = results_dir / "test_predictions.txt"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing {predictions_path}. Run postprocess.py first."
        )

    y_test, y_pred = np.loadtxt(predictions_path, dtype=float, ndmin=2).T
    return y_test, y_pred


def plot_r2(
    y_test,
    y_pred,
    output_stem,
    marker_color="black",
    marker_size=10,
    marker_alpha=0.5,
    line_color="gray",
    line_linestyle="--",
    line_linewidth=2,
):
    """Prediction against target, with the line y = x and R²."""
    discrepancy = y_pred - y_test
    r2 = 1 - np.sum(discrepancy**2) / np.sum((y_test - np.mean(y_test)) ** 2)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        y_test,
        y_pred,
        s=marker_size,
        color=marker_color,
        alpha=marker_alpha,
        edgecolors="none",
    )
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(
        lims,
        lims,
        color=line_color,
        linestyle=line_linestyle,
        linewidth=line_linewidth,
    )

    ax.text(
        0.05,
        0.95,
        rf"$R^2 = {r2:.4f}$",
        transform=ax.transAxes,
        verticalalignment="top",
    )
    ax.set_xlabel(r"Target distance [kg$\cdot$m]")
    ax.set_ylabel(r"Predicted distance [kg$\cdot$m]")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()

    for extension in ("png", "pdf"):
        output_path = output_stem.with_suffix(f".{extension}")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved {output_path}")

    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"

    y_test, y_pred = load_test_predictions(results_dir)
    plot_r2(y_test, y_pred, results_dir / "r2_test")


if __name__ == "__main__":
    main()
