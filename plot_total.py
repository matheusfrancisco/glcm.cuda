
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

def plot_execution_times(times_cpu, times_gpu, name):
    cpu = pd.DataFrame({"fn": ["total_cpu_dcm"], "time": [times_cpu[0]]})
    gpu = pd.DataFrame({"fn": ["total_gpu_dcm"], "time": [times_gpu[0]]})
    data = pd.concat([cpu, gpu]).reset_index(drop=True)
    plt.figure(figsize=(12, 7))
    plt.bar(data["fn"], data["time"], color=["skyblue", "salmon"])
    plt.xlabel('Process')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison of Functions (CPU vs GPU)')
    plt.savefig(name)
    plt.show()

if __name__ == "__main__":
    functions_cpu, times_cpu = read_csv("./total_cpu.csv")
    functions_gpu, times_gpu = read_csv("./total_gpu.csv")
    plot_execution_times(times_cpu, times_gpu, "total_time_comparison.png")
