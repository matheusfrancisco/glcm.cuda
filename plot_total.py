
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

def plot_execution_times(times_cpu,
                         times_gpu, 
                         python_impl,
                         omp,
                         cuda_png,
                         cuda_dicm,
                         name):
    cpu = pd.DataFrame({"fn": ["355 DICOM CPU"], "time": [times_cpu[0]]})
    gpu_omp = pd.DataFrame({"fn": ["355 DICOM CUDA+OMP"], "time": [times_gpu[0]]})
    omp= pd.DataFrame({"fn": ["355-Images-PNG with OMP+CUDA"], "time": [omp[0]]})
    python = pd.DataFrame({"fn": ["355 DICOM python Cuda"], "time": [python_impl[0]]})
    cuda_png= pd.DataFrame({"fn": ["355 Images-PNG Cuda"], "time": [cuda_png[0]]})
    cuda_dicm= pd.DataFrame({"fn": ["355 DICOM/V Images Cuda"], "time": [cuda_dicm[0]]})
    data = pd.concat([cpu, gpu_omp,cuda_png , omp, cuda_dicm, python]).reset_index(drop=True)
    plt.figure(figsize=(12, 7))
    plt.bar(data["fn"], data["time"], color=["skyblue", "salmon"])
    plt.xlabel('Process')
    plt.xticks(rotation=90)  # Rotate the x-axis labels to vertical
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison of Functions (CPU vs GPU)')
    plt.tight_layout()  # Ensure everything fits well
    plt.savefig(name)
    plt.show()

if __name__ == "__main__":
    functions_cpu, times_cpu = read_csv("./total_cpu.csv")
    functions_gpu, times_gpu = read_csv("./total_gpu_omp.csv")
    _, python_impl= read_csv("./benchmark_times.csv")
    _, omp_cuda_png= read_csv("./total_gpu_png_omp.csv")
    _, cuda_png= read_csv("./total_gpu_png.csv")
    _, cuda_dicm= read_csv("./total_gpu_omp_cuda_only.csv")
    plot_execution_times(times_cpu, times_gpu, python_impl, 
                         omp_cuda_png, cuda_png, cuda_dicm, "total_time_comparison.png")
