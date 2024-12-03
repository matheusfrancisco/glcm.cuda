import numpy as np
import cupy as cp
import csv
import cv2
from glcm_cupy import GLCM
import uuid
from PIL import Image
import os
from skimage.feature import graycomatrix
from pydicom import dcmread
import time
import os
import fnmatch



def list_images_recursive(dir_path, extensions=("*.png")):
    image_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if fnmatch.fnmatch(file, extensions):
                image_files.append(os.path.join(root, file))
    return image_files


def save_glcm_to_csv(glcm, output_dir, filename="output.csv"):
    # Check if output_dir is valid
    if not output_dir:
        raise ValueError("output_dir cannot be empty")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Full path for the output file
    file_path = os.path.join(output_dir, filename)

    # Save GLCM to a CSV file
    np.savetxt(file_path, glcm, delimiter=",")
    print(f"GLCM saved to {file_path}")

# Example usage
gclms = []
images = list_images_recursive("../data/")
for img in images:
    gclms.append(cp.asarray(cv2.imread(img)))


t = time.time()
glcm_result = []
for i, glcm in enumerate(gclms[:10]):
    glcm_result.append(GLCM(bin_from=256, bin_to=16, max_threads=512, verbose=False).run(glcm))
    print(f"Processed {i + 1} images")


elapsed_time = time.time() - t 

csv_path = os.path.join("../", "benchmark_times_png_python.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["string", "time"])
    writer.writerow(["python_png_cuda", elapsed_time])

print(f"Benchmark times saved to {csv_path}")


