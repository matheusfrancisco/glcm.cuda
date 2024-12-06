
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
def read_csv(file_path):
    functions = []
    times = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            functions.append(row['string'])
            times.append(float(row['time']))
    return functions, times



# Files and their respective labels
files = [
    ("./total_cpu.csv", "355 DICOM CPU C++"),
    ("./total_gpu_omp.csv", "355 DICOM CUDA+OMP C++"),
    ("./benchmark_times_dicom_python.csv", "355 DICOM Python ski"),
    ("./total_gpu_omp_cuda_only.csv", "355 DICOM/V Images CUDA C++"),
    ("./total_gpu_png_omp.csv", "355 Images-PNG CUDA+OMP C++"),
    ("./total_gpu_png.csv", "355 PNG CUDA C++"),
    ("./benchmark_times_png_python.csv", "PNG GLCM.cuda Python")
]

# Read data and calculate speedup relative to CPU
datas = []
cpu_time = None
for f, t in files:
    _, xs = read_csv(f)  # Read data (assuming a function exists for reading)
    time = xs[0]
    if "CPU" in t:
        cpu_time = time
    datas.append({"fn": t, "time": time})

df = pd.DataFrame(datas)
if cpu_time is None:
    raise ValueError("No CPU time found for baseline comparison.")

df["speedup"] = cpu_time / df["time"]

# Sort data by speedup for better visualization
df = df.sort_values(by="speedup", ascending=False)

# Plot speedup as a horizontal bar plot
plt.figure(figsize=(10, 6))
bars = plt.barh(df["fn"], df["speedup"], color="skyblue")

# Add annotations for speedup values
for bar, speedup in zip(bars, df["speedup"]):
    plt.text(
        bar.get_width(),
        bar.get_y() + bar.get_height() / 2,
        f"{speedup:.2f}x",
        va="center",
        ha="left"
    )

# Formatting
plt.xlabel("Speedup (Relative to CPU Time)")
plt.ylabel("Process")
plt.title("Speedup Comparison of Functions (CPU vs GPU)")
plt.tight_layout()  # Ensure the plot fits well
plt.savefig("speedup_comparison.png")
plt.show()
