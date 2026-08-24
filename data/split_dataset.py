import json
import os
import argparse
from pathlib import Path

def process_dataset(source_dir, target_files):
    
    for filename in target_files:
        filepath = os.path.join(source_dir, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {filename}: File not found.")
            continue
            
        tool_name = os.path.splitext(filename)[0]
        # Create output directory inside the source directory
        output_dir = os.path.join(source_dir, tool_name)
        
        print(f"Processing {filename} -> {output_dir}...")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        try:
            print(f"Loading {filepath}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Loaded {len(data)} items. Saving individual files...")
            
            count = 0
            for item in data:
                if 'images' in item and len(item['images']) > 0:
                    image_path = item['images'][0]
                    file_name = os.path.basename(image_path)
                    
                    # Construct output filename: replace extension with .json
                    name_base, _ = os.path.splitext(file_name)
                    output_filename = f"{name_base}.json"
                    
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save the entire item including 'messages' and 'images'
                    # The user requested:
                    # { 
                    #     "messages": [...], 
                    #     "images": [str(png_path)] 
                    # }
                    # The 'item' from the source json already has this structure.
                    
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        json.dump(item, out_f, ensure_ascii=False)
                    
                    count += 1
                    if count % 1000 == 0:
                        print(f"Processed {count} items for {tool_name}...")
                        
            print(f"Finished {tool_name}: {count} files created.")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "Beagle_Plus" / "train_json",
    )
    parser.add_argument(
        "--datasets",
        default="chartblocks,fusion_clean,graphiq,plotly_export,echarts",
    )
    args = parser.parse_args()
    target_files = [
        f"{name.strip()}.json"
        for name in args.datasets.split(",")
        if name.strip()
    ]
    process_dataset(str(args.source_dir.resolve()), target_files)


if __name__ == "__main__":
    main()
