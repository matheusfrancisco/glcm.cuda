import os
import shutil

def copy_file_multiple_times(source_file, destination_folder, copies):
    os.makedirs(destination_folder, exist_ok=True)

    for i in range(1, copies + 1):
        destination_file = os.path.join(destination_folder, f"copy_{i}.png")
        try:
            shutil.copy(source_file, destination_file)
            print(f"Copied to: {destination_file}")
        except FileNotFoundError:
            print(f"Error: Source file '{source_file}' not found.")
            return
        except Exception as e:
            print(f"An error occurred: {e}")
            return

if __name__ == "__main__":
    source_file = "./data/images_png/sample1024.png"  
    destination_folder = "./data/images_png/" 
    copies = 300

    copy_file_multiple_times(source_file, destination_folder, copies)
