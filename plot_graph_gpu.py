import csv
import os
import matplotlib.pyplot as plt

def read_csv(file_path):
    functions = []
    times = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            functions.append(row['string'])
            times.append(float(row['time']))
    return functions, times

def plot_execution_times(functions, times, name):
    plt.figure(figsize=(10, 6))
    plt.bar(functions, times, color='skyblue')
    plt.xlabel('Function')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of Functions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{name}.png")

def get_all_files_in_folder(folder_path):
    files_res = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            o = {"filename": f, "path": os.path.join(root, f), "name": f.split(".")[0]}
            files_res.append(o)
    return files_res

if __name__ == "__main__":
    files = get_all_files_in_folder('data/csv_result')
    for f in files:
        print(f" ploting {f.get("path")}")
        image, times = read_csv(f.get("path"))
        plot_execution_times(image, times, f.get("name"))
