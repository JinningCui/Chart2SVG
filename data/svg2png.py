import os
import cairosvg
import shutil

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Beagle")
ALL_DATASET_NAMES = ["chartblocks", "fusion_clean", "graphiq_clean", "plotly_export", "echarts"]
SELECTED_DATASET_NAMES = [
    name.strip()
    for name in os.environ.get("SVG2PNG_DATASETS", ",".join(ALL_DATASET_NAMES)).split(",")
    if name.strip()
]
SOURCE_DIRS = [os.path.join(BASE_DIR, name, "charts") for name in SELECTED_DATASET_NAMES]
SELECTED_IDS = {
    name.strip() for name in os.environ.get("SVG2PNG_IDS", "").split(",") if name.strip()
}
SELECTED_IDS_FILE = os.environ.get("SVG2PNG_IDS_FILE", "").strip()
if SELECTED_IDS_FILE:
    with open(SELECTED_IDS_FILE, "r", encoding="utf-8") as selected_handle:
        SELECTED_IDS.update(line.strip() for line in selected_handle if line.strip())
SHARD_COUNT = int(os.environ.get("SVG2PNG_SHARD_COUNT", "1"))
SHARD_INDEX = int(os.environ.get("SVG2PNG_SHARD_INDEX", "0"))


def stable_hash(name):
    value = 0
    for char in name:
        value = ((value * 31) + ord(char)) & 0xFFFFFFFF
    return value

def iter_svg_files(source_dir):
    for name in sorted(os.listdir(source_dir)):
        folder = os.path.join(source_dir, name)
        if not os.path.isdir(folder):
            continue
        if SELECTED_IDS and name not in SELECTED_IDS:
            continue
        if SHARD_COUNT > 1 and stable_hash(name) % SHARD_COUNT != SHARD_INDEX:
            continue
        svg_path = os.path.join(folder, "svg.txt")
        if not os.path.exists(svg_path):
            continue
        yield name, svg_path

def render_svg_to_png(svg_path, output_path):
    with open(svg_path, "r", encoding="utf-8") as f:
        svg = f.read()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=output_path)

def main():
    total = 0
    failed = 0
    all_tasks = []
    for source_dir in SOURCE_DIRS:
        if not os.path.isdir(source_dir):
            print(f"跳过，不存在: {source_dir}")
            continue
        tasks = list(iter_svg_files(source_dir))
        all_tasks.append((source_dir, tasks))

    total_tasks = sum(len(tasks) for _, tasks in all_tasks)
    processed = 0

    for source_dir, tasks in all_tasks:
        for name, svg_path in tasks:
            folder_output_path = os.path.join(source_dir, name, f"{name}.png")
            try:
                render_svg_to_png(svg_path, folder_output_path)
                total += 1
            except Exception as e:
                failed += 1
                print(f"失败: {svg_path} -> {folder_output_path} | {e}")
            processed += 1
            if total_tasks > 0:
                percent = (processed / total_tasks) * 100
                print(f"\r进度: {percent:.1f}% ({processed}/{total_tasks}) | 成功: {total} 失败: {failed}", end="", flush=True)
    if total_tasks > 0:
        print()
    print(f"完成: {total} 张, 失败: {failed} 张")


def copy_all_files(src_dir, dst_dir):
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"源目录不存在: {src_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        target_root = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        os.makedirs(target_root, exist_ok=True)
        for name in files:
            src_path = os.path.join(root, name)
            dst_path = os.path.join(target_root, name)
            shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    main()
    
    # SRC_DIR = "/disk/CJN/Chart2SVG_dataset/Echarts_cleaned/charts/images"
    # DST_DIR = "/disk/CJN/Chart2SVG_dataset/Echarts_cleaned/images"
    # os.makedirs(os.path.dirname(DST_DIR), exist_ok=True)
    # copy_all_files(SRC_DIR, DST_DIR)
    # print(f"复制完成: {SRC_DIR} -> {DST_DIR}")
