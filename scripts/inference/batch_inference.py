import subprocess
from pathlib import Path
import os
import sys

# Define the list of datasets to process
datasets = [
    # "Path to your dataset",
    # "Path to your dataset",
    # "Path to your dataset",
    # "Path to your dataset",
    # "Path to your dataset",
    # "Path to your dataset",
    # "Path to your dataset",
    # "Path to your dataset",
    # "Path to your dataset",
    # 'test_line.json',
    # "Path to your dataset",
    "Path to your dataset",
    # "Path to your dataset"
]

# Path to the inference script
script_path = "Path to your inference script"

if not os.path.exists(script_path):
    print(f"Error: Inference script not found at {script_path}")
    sys.exit(1)

for dataset_str in datasets:
    dataset_path = Path(dataset_str)
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Warning: Dataset {dataset_path} not found. Skipping.")
        continue
        
    parent_dir = dataset_path.parent
    stem = dataset_path.stem
    
    # Define output directories in the parent folder of the json file
    # We use specific directories for each dataset to avoid overwriting results
    # if multiple datasets are in the same folder.
    output_dir = parent_dir / f"{stem}_our8B_full_new4_infer"
    resized_dir = parent_dir / f"{stem}_resized-input"
    
    print(f"\n{'='*50}")
    print(f"Processing dataset: {dataset_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Resized Inputs Directory: {resized_dir}")
    print(f"{'='*50}\n")
    
    cmd = [
        "python3",
        script_path,
        "--dataset_path", str(dataset_path),
        "--output_dir", str(output_dir),
        "--resized_dir", str(resized_dir)
    ]
    
    try:
        # Run the inference script
        subprocess.run(cmd, check=True)
        print(f"\nSuccessfully processed {dataset_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nError processing {dataset_path}: {e}")
        # Option: continue to next dataset or stop?
        # We continue to try processing others.
    except Exception as e:
        print(f"\nUnexpected error for {dataset_path}: {e}")

print("\nBatch inference completed.")
