import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


LOG_PATH = Path("log/run.log")


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log_msg(msg):
    line = f"[{timestamp()}] {msg}"

    # Print to terminal
    print(line)

    # Append to log file
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def plot_tool(csv_path, output_img_dir):
    log_msg(f"Plotting file: {csv_path}")

    df = pd.read_csv(csv_path)

    if not {"original_signal", "filtered_signal"}.issubset(df.columns):
        log_msg(f"ERROR: Missing required columns in {csv_path}")
        return

    original = df["original_signal"].values
    cuda_filtered = df["filtered_signal"].values

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(original, color="black")
    axes[0].set_title("Original Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    axes[1].plot(cuda_filtered, color="blue")
    axes[1].set_title("CUDA Butterworth Filter Output")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Sample Index")
    axes[1].grid(True)

    plt.tight_layout()

    out_path = output_img_dir / (csv_path.stem + ".png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    log_msg(f"Saved plot: {out_path}")


if __name__ == "__main__":
    output_dir = Path("data/output")
    plot_dir = Path("data/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    log_msg("Python plotting started")

    for csv_file in sorted(output_dir.glob("*.csv")):
        plot_tool(csv_file, plot_dir)

    log_msg("Python plotting finished")
