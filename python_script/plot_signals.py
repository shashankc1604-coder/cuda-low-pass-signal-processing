import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
RESULTS_DIR = "results"

SAMPLING_RATE_HZ = 2500  # for labeling only

os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_signal(raw_csv, filtered_csv, output_png):
    raw_df = pd.read_csv(raw_csv)
    filt_df = pd.read_csv(filtered_csv)

    raw_signal = raw_df["sensor"].values
    filtered_signal = filt_df["filtered_sensor"].values

    n = min(len(raw_signal), len(filtered_signal))
    time_axis = [i / SAMPLING_RATE_HZ for i in range(n)]

    # Create two subplots (vertical stack)
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # ---- Raw signal ----
    axs[0].plot(time_axis, raw_signal[:n], linewidth=1)
    axs[0].set_title("Raw Sensor Signal")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    # ---- Filtered signal ----
    axs[1].plot(time_axis, filtered_signal[:n], linewidth=1)
    axs[1].set_title("Filtered Sensor Signal (Low-Pass)")
    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def main():
    print("Starting signal plotting...")

    for file in os.listdir(INPUT_DIR):
        if not file.endswith(".csv"):
            continue

        raw_path = os.path.join(INPUT_DIR, file)
        filtered_path = os.path.join(OUTPUT_DIR, file)

        if not os.path.exists(filtered_path):
            print(f"Filtered file missing for {file}, skipping.")
            continue

        output_png = os.path.join(
            RESULTS_DIR,
            file.replace(".csv", ".png")
        )

        print(f"Plotting {file} → {output_png}")
        plot_signal(raw_path, filtered_path, output_png)

    print("Plotting complete.")


if __name__ == "__main__":
    main()
