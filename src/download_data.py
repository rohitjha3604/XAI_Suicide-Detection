"""
Download the Suicide Watch dataset from Kaggle using kagglehub.
Copies the CSV file to the data/ directory.
"""

import os
import shutil
import glob
import kagglehub

def download_dataset():
    """
    Downloads the suicide-watch dataset from Kaggle and copies it to data/ folder.
    """
    # Create data directory if it doesn't exist
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading dataset from Kaggle...")
    
    try:
        # Download dataset using kagglehub
        path = kagglehub.dataset_download("nikhileswarkomati/suicide-watch")
        print(f"Dataset downloaded to: {path}")
        
        # Find CSV files in the downloaded directory
        csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in downloaded dataset.")
        
        # Copy the first CSV file found to data directory
        source_csv = csv_files[0]
        dest_csv = os.path.join(data_dir, "suicide_watch.csv")
        
        shutil.copy(source_csv, dest_csv)
        print(f"CSV copied to: {dest_csv}")
        
        # Verify and print column names
        import pandas as pd
        df = pd.read_csv(dest_csv)
        print(f"\nDataset shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
        print(f"\nFirst few rows:\n{df.head()}")
        
        return dest_csv
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

if __name__ == "__main__":
    download_dataset()
