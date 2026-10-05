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


def load_loss_history(results_dir):
    """Load the loss history written by main.py."""
    train_path = results_dir / "loss_train.txt"
    validation_path = results_dir / "loss_validation.txt"
    epochs_path = results_dir / "loss_epochs.txt"

    missing_paths = [
        path for path in (train_path, validation_path) if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing loss file(s): {missing}. Run main.py first.")

    loss_train = np.atleast_1d(np.loadtxt(train_path, dtype=float))
    loss_validation = np.atleast_1d(np.loadtxt(validation_path, dtype=float))
    if epochs_path.exists():
        x_values = np.atleast_1d(np.loadtxt(epochs_path, dtype=int))
        x_label = "Epoch"
    else:
        x_values = np.arange(1, loss_train.size + 1)
        x_label = "Evaluation"

    if not (loss_train.shape == loss_validation.shape == x_values.shape):
        raise ValueError(
            "Loss history files have inconsistent lengths: "
            f"train={loss_train.size}, validation={loss_validation.size}, "
            f"x_values={x_values.size}."
        )

    return x_values, x_label, loss_train, loss_validation


def plot_loss_history(
    x_values,
    x_label,
    loss_train,
    loss_validation,
    output_stem,
    train_linestyle="--",
    train_color="gray",
    train_linewidth=2,
    validation_linestyle="-",
    validation_color="black",
    validation_linewidth=2,
):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        x_values,
        loss_train,
        label="train",
        linestyle=train_linestyle,
        color=train_color,
        linewidth=train_linewidth,
    )
    ax.plot(
        x_values,
        loss_validation,
        label="validation",
        linestyle=validation_linestyle,
        color=validation_color,
        linewidth=validation_linewidth,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()

    for extension in ("png", "pdf"):
        output_path = output_stem.with_suffix(f".{extension}")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved {output_path}")

    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"

    train_linestyle = "--"
    train_color = "gray"
    train_linewidth = 2
    validation_linestyle = "-"
    validation_color = "black"
    validation_linewidth = 2

    x_values, x_label, loss_train, loss_validation = load_loss_history(results_dir)
    plot_loss_history(
        x_values,
        x_label,
        loss_train,
        loss_validation,
        results_dir / "loss_epochs",
        train_linestyle=train_linestyle,
        train_color=train_color,
        train_linewidth=train_linewidth,
        validation_linestyle=validation_linestyle,
        validation_color=validation_color,
        validation_linewidth=validation_linewidth,
    )


if __name__ == "__main__":
    main()
