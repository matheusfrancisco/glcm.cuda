import csv
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

def plot_execution_times(functions, times):
    plt.figure(figsize=(10, 6))
    plt.bar(functions, times, color='skyblue')
    plt.xlabel('Function')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of Functions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plot.png")

if __name__ == "__main__":
    # Replace 'timing_results.csv' with the path to your CSV file
    csv_file_path = 'data/csv_result/dcm_result0.csv'
    functions, times = read_csv(csv_file_path)
    plot_execution_times(functions, times)
