"""
Data ingestion script for CBIS-DDSM breast cancer image dataset.
Downloads the dataset, finds the CSV, and sorts images into cancer/no_cancer folders.
"""
import kagglehub
import pandas as pd
import os
import shutil

def download_and_prepare_dataset():
    # Step 1: Download the dataset
    path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
    print("Path to dataset files:", path)

    # Step 2: Find CSV file
    csv_path = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                print("Found CSV file:", csv_path)

    # Step 3: Read the CSV (optional)
    if csv_path:
        df = pd.read_csv(csv_path)
        print("First 5 rows of the dataset:")
        print(df.head())
    else:
        print("No CSV file found in the dataset.")
        df = None

    # Step 4: Prepare folders
    has_cancer_dir = os.path.join("sorted_ddsm_images", "has_cancer")
    no_cancer_dir = os.path.join("sorted_ddsm_images", "no_cancer")
    os.makedirs(has_cancer_dir, exist_ok=True)
    os.makedirs(no_cancer_dir, exist_ok=True)

    # Step 5: Sort and save images
    image_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    has_count = 0
    no_count = 0

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(image_extensions):
                original_path = os.path.join(root, file)
                first_char = os.path.basename(file)[0]

                if first_char == "1":
                    target_path = os.path.join(has_cancer_dir, file)
                    has_count += 1
                elif first_char == "2":
                    target_path = os.path.join(no_cancer_dir, file)
                    no_count += 1
                else:
                    continue  # skip files that don't match 1 or 2

                if not os.path.exists(target_path):
                    shutil.copy2(original_path, target_path)

    print(f"Saved {has_count} cancer-positive images to: {os.path.abspath(has_cancer_dir)}")
    print(f"Saved {no_count} cancer-negative images to: {os.path.abspath(no_cancer_dir)}")
    return path, csv_path, has_cancer_dir, no_cancer_dir, df

if __name__ == "__main__":
    download_and_prepare_dataset()
