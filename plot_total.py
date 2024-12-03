
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


files = [("./total_gpu_png_omp.csv", "355-Images-PNG with OMP+CUDA C++"),
         ("./total_gpu_png.csv", "355 PNG CUDA C++"),
         ("./benchmark_times_png_python.csv", "PNG glcm.cuda python"),
         ("./total_gpu_png.csv", "355 Images-PNG Cuda C++" ),

         ("./total_cpu.csv", "355 DICOM CPU C++"),
         ("./total_gpu_omp.csv", "355 DICOM CUDA+OMP C++"),
         ("./benchmark_times_dicom_python.csv", "355 DICOM python ski"),
         ("./total_gpu_omp_cuda_only.csv",  "355 DICOM/V Images Cuda C++")]

datas = []
for (f, t) in files:
    _, xs = read_csv(f) 
    x = pd.DataFrame({"fn": [t], "time": [xs[0]]})
    datas.append(x)



data = pd.concat(datas).reset_index(drop=True)
plt.figure(figsize=(12, 7))
plt.bar(data["fn"], data["time"], color=["skyblue", "salmon"])
plt.xlabel('Process')
plt.xticks(rotation=90)  # Rotate the x-axis labels to vertical
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison of Functions (CPU vs GPU)')
plt.tight_layout()  # Ensure everything fits well
plt.savefig("total_time_comparison.png")
plt.show()
