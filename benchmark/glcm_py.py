import numpy as np
import csv
import uuid
import os
from skimage.feature import graycomatrix
from pydicom import dcmread
import time

import os
import fnmatch

def list_images_recursive(dir_path, extensions=("*.dcm")):
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
    import numpy as np
    np.savetxt(file_path, glcm, delimiter=",")
    print(f"GLCM saved to {file_path}")

# Example usage
images = list_images_recursive("../dataset/")
print(f"Found {len(images)} DICOM images")

distances = [1]  # Distance of 1 pixel
angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]

start = time.time()
glcms = []
for img in images:
    print(img)
# Compute GLCM for all angles
    dcm = dcmread(img)
    glcms.append(graycomatrix(
        dcm.pixel_array.astype(np.uint8),
        distances=distances,
        angles=angles,
        symmetric=True,
        normed=True
    ))
    
elapsed_time = time.time() - start

for glcm in glcms:
    name = str(uuid.uuid4())
    for angle in range(glcm.shape[3]):
        print(f"Saving GLCM to {name}.csv")
        save_glcm_to_csv(glcm[:, :, 0, angle], "./output", f"{name}.csv")

# Save elapsed time to CSV
csv_path = os.path.join("../", "benchmark_times.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["string", "time"])
    writer.writerow(["python_glcm", elapsed_time])

print(f"Benchmark times saved to {csv_path}")
