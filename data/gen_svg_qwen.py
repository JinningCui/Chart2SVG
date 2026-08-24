import sys
import os
import re
import json
import struct
import cairosvg
import multiprocessing
import argparse
from pathlib import Path
from tqdm import tqdm
from functools import partial

# Add project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
semantic_module_dir = current_file.parent
if str(semantic_module_dir) not in sys.path:
    sys.path.insert(0, str(semantic_module_dir))

from semantic_tokens import svg2syntactic, syntactic2svg


def _get_png_size(png_path):
    try:
        with open(png_path, 'rb') as f:
            sig = f.read(8)
            if sig != b'\x89PNG\r\n\x1a\n':
                return None, None
            length_bytes = f.read(4)
            chunk_type = f.read(4)
            if chunk_type != b'IHDR':
                return None, None
            ihdr_data = f.read(13)
            width, height = struct.unpack('>II', ihdr_data[:8])
            return width, height
    except Exception:
        return None, None

def process_single_chart(args):
    folder_name, charts_dir, tool_name = args
    folder_path = charts_dir / folder_name
    if not folder_path.is_dir():
        return None
    
    # Check if PNG exists
    png_path = folder_path / f"{folder_name}.png"
    if not png_path.exists():
        return None
    orig_w, orig_h = _get_png_size(png_path)
    
    # Identify the input text file
    input_file = folder_path / "svg.txt"
    if not input_file.exists():
        return None
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove XML declaration if present (lxml.fromstring limitation with unicode)
        content = re.sub(r'<\?xml.*?\?>', '', content).strip()
        
        # Remove bitmap images (handling both href and xlink:href)
        # Removes <image ... /> or <image ...>...</image>
        # Split into two regexes for better performance and safety
        content = re.sub(r'<image[^>]*?/>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'<image[^>]*?>.*?</image>', '', content, flags=re.DOTALL | re.IGNORECASE)

        # Filter out large files to avoid token length errors
        if len(content) > 80000:
            return None
        
        # 1. svg2syntactic
        _, svg_desc = svg2syntactic(content)
        
        # 2. syntactic2svg
        svg_code = syntactic2svg(svg_desc)
        if folder_name == "1":
            print(svg_code)
        
        # 3. Save as train_svg.txt
        train_svg_path = folder_path / "train_svg.txt"
        with open(train_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_desc)
        
        # 4. Render to png, forcing 512x512 as required by the model
        try:
            cairosvg.svg2png(
                bytestring=svg_code.encode('utf-8'),
                write_to=str(png_path),
                output_width=512,
                output_height=512,
            )

        except Exception as e:
            # print(f"Error rendering PNG for {folder_name}: {e}")
            pass
        
        # Return data for JSON in the expected format
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a world-class SVG Expert and Data Visualization Engineer. Your primary objective is to interpret rasterized chart images and reconstruct them into high-quality, semantically correct SVG code."
                },
                {
                    "role": "user",
                    "content": "<image>Convert this image to SVG code."
                },
                {
                    "role": "assistant",
                    "content": svg_desc
                }
            ],
            "images": [str(png_path)]
        }

    except Exception as e:
        print(f"Error processing {folder_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_beagle_dataset(base_dir: Path, datasets, workers: int):
    target_dirs = [base_dir / name / "charts" for name in datasets]
    output_dir = base_dir / "train_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if directories exist
    valid_dirs = []
    for d in target_dirs:
        if d.exists():
            valid_dirs.append(d)
        else:
            print(f"Warning: Directory not found: {d}")
    
    for charts_dir in valid_dirs:
        dataset_name = charts_dir.parent.name
        tool_name = "graphiq" if dataset_name == "graphiq_clean" else dataset_name

        print(f"Processing directory: {charts_dir} for tool: {tool_name}")
        
        # Prepare arguments for multiprocessing
        tasks = []
        for folder_name in sorted(os.listdir(charts_dir)):
            tasks.append((folder_name, charts_dir, tool_name))
            
        tool_data = []
        with multiprocessing.Pool(processes=workers) as pool:
            # Use imap_unordered for better responsiveness with tqdm
            # Limit to 10 for testing
            for result in tqdm(pool.imap_unordered(process_single_chart, tasks), total=len(tasks)):
                if result:
                    tool_data.append(result)
        
        # Save JSON
        json_output_path = output_dir / f"{tool_name}.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(tool_data, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(tool_data)} records to {json_output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=current_file.parent / "data" / "Beagle_Plus",
    )
    parser.add_argument(
        "--datasets",
        default="chartblocks,fusion_clean,graphiq_clean,plotly_export,echarts",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, int(os.environ.get("GEN_SVG_WORKERS", "8"))),
    )
    args = parser.parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    process_beagle_dataset(args.base_dir.resolve(), datasets, args.workers)


if __name__ == "__main__":
    main()
