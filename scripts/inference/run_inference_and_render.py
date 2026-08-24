import sys
import os
import json
import re
import torch
from datetime import datetime
from lxml import etree
from pathlib import Path
from tqdm import tqdm
import cairosvg
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import argparse

# Add project root to sys.path to allow importing Chart2SVG modules
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from swift.llm import InferArguments, InferRequest
from swift.llm.infer import SwiftInfer
from Chart2SVG.data.semantic_tokens import syntactic2svg

TARGET_SIZE = 512

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and render SVG")
    parser.add_argument('--checkpoint_path', type=str, default='Path to your checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the input dataset JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save rendered SVGs/PNGs')
    parser.add_argument('--resized_dir', type=str, required=True, help='Directory to save resized images')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to process (for debugging)')
    parser.add_argument('--model_type', type=str, default='qwen3_vl', help='Model type (e.g. qwen3_vl, qwen2-vl). Default is qwen3_vl.')
    parser.add_argument('--infer_backend', type=str, default='pt', choices=['pt', 'vllm', 'lmdeploy'])
    parser.add_argument('--max_batch_size', type=int, default=16)
    parser.add_argument('--concurrent', action='store_true')
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument('--stream', action='store_true')
    stream_group.add_argument('--no_stream', action='store_true')
    parser.add_argument('--ddp_backend', type=str, default=None)
    parser.add_argument('--render_workers', type=int, default=None)
    return parser.parse_args()

def normalize_image(image_path, target_size=TARGET_SIZE):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        w, h = img.size
        scale = target_size / max(w, h)
        w_new = int(w * scale)
        h_new = int(h * scale)
        img_resized = img.resize((w_new, h_new), Image.Resampling.LANCZOS)
        pad_x = (target_size - w_new) / 2
        pad_y = (target_size - h_new) / 2
        new_img = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        new_img.paste(img_resized, (int(pad_x), int(pad_y)))
        return new_img, scale, pad_x, pad_y, w, h

_PATH_TOKEN_RE = re.compile(r'([a-zA-Z])|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

def _expand_arc_flags(tokens):
    new_tokens = []
    i = 0
    n = len(tokens)
    current_cmd = None
    arg_idx = 0 
    
    while i < n:
        t = tokens[i]
        if t[0].isalpha() and t.upper() not in ['E', 'e']:
            current_cmd = t.upper()
            new_tokens.append(t)
            arg_idx = 0
            i += 1
            continue
            
        if current_cmd == 'A':
            cycle = arg_idx % 7
            if cycle in (3, 4):
                if t == "0" or t == "1":
                    new_tokens.append(t)
                    arg_idx += 1
                    i += 1
                else:
                    first = t[0]
                    remainder = t[1:]
                    if first in ('0', '1'):
                        new_tokens.append(first)
                        arg_idx += 1
                        if remainder:
                            tokens[i] = remainder
                        else:
                            i += 1
                    else:
                        new_tokens.append(t)
                        arg_idx += 1
                        i += 1
            else:
                new_tokens.append(t)
                arg_idx += 1
                i += 1
        else:
            new_tokens.append(t)
            i += 1
            
    return new_tokens

def untransform_path_d(d, scale):
    if not d or len(d) < 2:
        return d

    # Pre-process
    d = d.replace('NaN', '0').replace('nan', '0').replace('NAN', '0').replace('null', '0')

    tokens = _PATH_TOKEN_RE.findall(d)
    
    raw_tokens = []
    for t in tokens:
        if t[0]: raw_tokens.append(t[0])
        else: raw_tokens.append(t[1])
        
    flat_tokens_str = _expand_arc_flags(raw_tokens)
    
    flat_tokens = []
    for t in flat_tokens_str:
        if t[0].isalpha() and t.upper() not in ['E', 'e']:
             flat_tokens.append(t)
        else:
             try:
                 flat_tokens.append(float(t))
             except ValueError:
                 flat_tokens.append(0.0)
            
    res = []
    n_tokens = len(flat_tokens)
    i = 0
    current_cmd = None
    
    args_count = {
        'M': 2, 'm': 2, 'L': 2, 'l': 2, 'H': 1, 'h': 1, 'V': 1, 'v': 1,
        'C': 6, 'c': 6, 'S': 4, 's': 4, 'Q': 4, 'q': 4, 'T': 2, 't': 2,
        'A': 7, 'a': 7, 'Z': 0, 'z': 0
    }
    
    while i < n_tokens:
        token = flat_tokens[i]
        if isinstance(token, str):
            current_cmd = token
            res.append(current_cmd)
            i += 1
        
        if current_cmd is None:
            i += 1
            continue
            
        cmd_upper = current_cmd.upper()
        n_args = args_count.get(cmd_upper, 0)
        
        if n_args == 0:
            i += 1
            continue
            
        start_arg_idx = i
        end_arg_idx = i
        while end_arg_idx < n_tokens and not isinstance(flat_tokens[end_arg_idx], str):
            end_arg_idx += 1
            
        available_args = end_arg_idx - start_arg_idx
        
        # Process arguments in chunks of n_args
        for chunk_start in range(0, available_args, n_args):
            if chunk_start + n_args > available_args:
                break
                
            args = flat_tokens[start_arg_idx + chunk_start : start_arg_idx + chunk_start + n_args]
            new_args = []
            
            # Helper to unscale
            def unscale(val):
                return round(val / scale, 2)
            
            if cmd_upper == 'A':
                # rx ry x-axis-rotation large-arc-flag sweep-flag x y
                new_args.append(unscale(args[0])) # rx
                new_args.append(unscale(args[1])) # ry
                new_args.append(args[2])          # rot
                new_args.append(args[3])          # large
                new_args.append(args[4])          # sweep
                new_args.append(unscale(args[5])) # x
                new_args.append(unscale(args[6])) # y
            elif cmd_upper in ['H', 'h', 'V', 'v']:
                new_args.append(unscale(args[0]))
            else:
                new_args = [unscale(x) for x in args]
                
            res.extend(map(str, new_args))
            
        i = end_arg_idx

    return " ".join(res)

def _unscale_transform_str(transform_str, scale):
    if not transform_str:
        return transform_str
    
    pattern = re.compile(r'([a-zA-Z]+)\s*\(([^)]*)\)')
    
    def repl(match):
        cmd = match.group(1).lower()
        args_str = match.group(2)
        raw_args = re.split(r'[,\s]+', args_str.strip())
        args = []
        for x in raw_args:
            if not x: continue
            clean_x = x.lower().replace('deg', '').replace('px', '').replace('null', '0')
            try:
                args.append(float(clean_x))
            except ValueError:
                args.append(0.0)
        
        if not args:
            return match.group(0)

        new_args = []
        
        if cmd == 'translate':
            new_args.append(round(args[0] / scale, 2))
            if len(args) > 1:
                new_args.append(round(args[1] / scale, 2))
                
        elif cmd == 'matrix':
            if len(args) == 6:
                new_args = [
                    args[0], args[1], args[2], args[3],
                    round(args[4] / scale, 2), # e (tx)
                    round(args[5] / scale, 2)  # f (ty)
                ]
            else:
                return match.group(0)
                
        elif cmd == 'rotate':
            new_args.append(args[0])
            if len(args) > 1:
                new_args.append(round(args[1] / scale, 2))
            if len(args) > 2:
                new_args.append(round(args[2] / scale, 2))
                
        else:
            new_args = [round(x, 2) for x in args] # keep others? or unscale? 
            # Scale usually means scaling the object, if we are restoring coordinate system,
            # we don't change object local scale.
            
        args_joined = ", ".join(map(str, new_args))
        return f"{match.group(1)}({args_joined})"

    return pattern.sub(repl, transform_str)

def _unscale_style_str(style_str, scale):
    if not style_str:
        return style_str
    
    parts = [p.strip() for p in style_str.split(';') if p.strip()]
    new_parts = []
    
    for part in parts:
        if ':' not in part:
            new_parts.append(part)
            continue
            
        key, val = [x.strip() for x in part.split(':', 1)]
        key_lower = key.lower()
        
        if key_lower in ('font-size', 'stroke-width'):
            try:
                val_clean = val.lower().replace('px', '').strip()
                val_float = float(val_clean)
                new_val = round(val_float / scale, 2)
                new_parts.append(f"{key}: {new_val}px")
            except ValueError:
                new_parts.append(part)
        else:
            new_parts.append(part)
            
    return "; ".join(new_parts)

def extract_svg_code(text):
    # Try to find SVG block
    pattern = r'(<svg[\s\S]*?</svg>)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Strip markdown code blocks if no clear svg tag found but markdown is present
    text = re.sub(r'```(?:xml|svg)?\n', '', text)
    text = re.sub(r'```\s*$', '', text)
    return text.strip()

def denormalize_svg(svg_code, meta):
    if not svg_code:
        return svg_code
        
    # 1. Remove invisible spaces.
    svg_code = svg_code.replace('\xa0', ' ').replace('\u200b', '')
    svg_code = re.sub(r'\s+', ' ', svg_code)
    
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    try:
        root = etree.fromstring(svg_code.encode('utf-8'), parser)
    except Exception:
        return svg_code
        
    # 2. Remove stale dimensions and viewport attributes.
    for attr in ['width', 'height', 'viewBox', 'preserveAspectRatio']:
        if attr in root.attrib:
            del root.attrib[attr]

    # 3. Remove an unnecessary 512x512 background that can cause misalignment.
    # Delete a direct child path that fills the entire 512x512 SVG canvas.
    for child in list(root):
        if child.tag == 'path':
            d_attr = child.get('d', '')
            if '512' in d_attr and 'H 0' in d_attr:
                root.remove(child)
                break

    # 4. Calculate and apply the correct viewBox.
    if meta and 'pad_x' in meta and 'pad_y' in meta:
        orig_w = float(meta['orig_w'])
        orig_h = float(meta['orig_h'])
        pad_x = float(meta['pad_x'])
        pad_y = float(meta['pad_y'])
        
        # Normalization scales the image into a 512x512 canvas.
        # The content area is 512 - 2*pad_x wide and 512 - 2*pad_y high.
        valid_w = 512.0 - 2 * pad_x
        valid_h = 512.0 - 2 * pad_y
        
        # The model offsets content with <g transform="translate(pad_x, pad_y)">.
        # Therefore, the visible content starts at pad_x and pad_y.
        # Frame that content area with the viewBox.
        root.set('viewBox', f"{pad_x:.2f} {pad_y:.2f} {valid_w:.2f} {valid_h:.2f}")
        
        # Restore the output file to the original physical dimensions.
        # The renderer scales the valid_w x valid_h viewBox to fill the
        # orig_w x orig_h canvas.
        root.set('width', str(int(orig_w)))
        root.set('height', str(int(orig_h)))
    else:
        root.set('viewBox', "0 0 512 512")
        root.set('width', '512')
        root.set('height', '512')

    return etree.tostring(root, encoding='unicode', pretty_print=False)

_RENDER_META_MAP = None

def _init_render_worker(meta_map):
    global _RENDER_META_MAP
    _RENDER_META_MAP = meta_map

def _render_single(task):
    # item, index, output_dir = task
    item, index, output_dir, candidate_index = task
    meta_map = _RENDER_META_MAP or {}
    output_dir = Path(output_dir)
    response = item['response']
    images = item.get('images', [])
    base_name = f"sample_{index}"
    try:
        semantic_svg = response
        try:
            standard_svg = syntactic2svg(semantic_svg)
        except Exception as e:
            print(f"Error in syntactic2svg for {base_name}: {e}")
            standard_svg = ""
        
        svg_code = standard_svg
        # svg_code= response
        
        img_key = None
        if images:
            img_info = images[0]
            if isinstance(img_info, str):
                img_key = img_info
            elif isinstance(img_info, dict) and 'path' in img_info:
                img_key = img_info['path']

        if img_key:
            base_name = Path(img_key).stem

        if candidate_index is not None:
            base_name = f"{base_name}{candidate_index}"
        
        svg_path = output_dir / f"{base_name}.svg"
        png_path = output_dir / f"{base_name}.png"
        
        meta = None
        if img_key and img_key in meta_map:
            meta = meta_map[img_key]
        
        final_svg = svg_code
        if meta:
            final_svg = denormalize_svg(svg_code, meta)

        json_path = output_dir / f"{base_name}.json"
        current_entry = {
            "timestamp": datetime.now().isoformat(),
            "semantic_svg": response,
            "standard_svg": svg_code,
            "converted_svg": final_svg
        }
        
        existing_data = []
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
            except Exception as e:
                print(f"Warning: Could not read existing JSON {json_path}: {e}")
        
        existing_data.append(current_entry)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        with open(svg_path, 'w') as f:
            f.write(final_svg)
        cairosvg.svg2png(bytestring=final_svg.encode('utf-8'), write_to=str(png_path))
        
    except Exception as e:
        return {'index': index, 'name': base_name, 'error': str(e)}
    return None


def run_inference(args):
    dataset_path = args.dataset_path
    checkpoint_path = args.checkpoint_path
    output_dir = Path(args.output_dir)
    resized_dir = Path(args.resized_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    resized_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]
        
    print(f"Found {len(data)} samples.")

    # Filter out samples whose SVG candidates have already been generated.
    filtered_data = []
    for idx, item in enumerate(data):
        # Derive the sample basename using the same logic as _render_single.
        images = item.get('images', [])
        if images and Path(images[0]).exists():
            base_name = Path(images[0]).stem
        else:
            base_name = f"sample_{idx}"
            
        # request_config.n = 4 generates four candidates per sample.
        # The fourth output indicates that the sample completed successfully.
        check_file = output_dir / f"{base_name}4.svg" 
        
        if check_file.exists():
            continue  # Skip completed samples.
        filtered_data.append(item)
        
    print(f"Filtered dataset: {len(filtered_data)} / {len(data)} samples need inference.")
    data = filtered_data
    # =======================================
    
    if args.limit:
        print(f"Limiting to first {args.limit} samples for debugging.")
        data = data[:args.limit]

    # 1.1 Normalize Images and record meta
    print("Normalizing images to 512x512...")
    meta_map = {}
    
    for item in tqdm(data, desc="Resizing"):
        images = item.get('images', [])
        new_images = []
        for img_path in images:
            src = Path(img_path)
            if not src.exists():
                new_images.append(img_path)
                continue
            new_filename = f"{src.stem}_normalized{src.suffix}"
            dst = resized_dir / new_filename
            if not dst.exists():
                try:
                    norm_img, scale, pad_x, pad_y, w0, h0 = normalize_image(src)
                    norm_img.save(dst)
                    meta_map[str(dst)] = {'orig_w': w0, 'orig_h': h0, 'scale': scale, 'pad_x': pad_x, 'pad_y': pad_y}
                except Exception:
                    new_images.append(img_path)
                    continue
            else:
                if str(dst) not in meta_map:
                    try:
                        with Image.open(src) as im:
                            im = im.convert('RGB')
                            w0, h0 = im.size
                        s = TARGET_SIZE / max(w0, h0)
                        w_new = int(w0 * s)
                        h_new = int(h0 * s)
                        meta_map[str(dst)] = {
                            'orig_w': w0,
                            'orig_h': h0,
                            'scale': s,
                            'pad_x': (TARGET_SIZE - w_new) / 2,
                            'pad_y': (TARGET_SIZE - h_new) / 2
                        }
                    except Exception:
                        pass
            new_images.append(str(dst))
        item['images'] = new_images
        
    # Update dataset json
    updated_dataset_path = Path(dataset_path).with_name(f"{Path(dataset_path).stem}_processed.json")
    with open(updated_dataset_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved resized dataset to {updated_dataset_path}")

    # 2. Configure SwiftInfer
    ckpt_path_obj = Path(checkpoint_path)
    if ckpt_path_obj.is_dir() and not (ckpt_path_obj / 'adapter_config.json').exists():
        candidate_ckpts = [d for d in ckpt_path_obj.iterdir() if d.is_dir() and d.name.startswith('checkpoint')]
        if candidate_ckpts:
            def _step(x):
                try:
                    return int(x.name.split('-')[-1])
                except Exception:
                    return 0
            latest = sorted(candidate_ckpts, key=_step)[-1]
            checkpoint_path = str(latest)
            print(f"Detected run directory, using latest checkpoint: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    is_lora = (Path(checkpoint_path) / 'adapter_config.json').exists()
    
    adapters = []
    base_model = checkpoint_path
    
    if is_lora:
        with open(Path(checkpoint_path) / 'adapter_config.json', 'r') as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get('base_model_name_or_path')
        adapters = [checkpoint_path]
        print(f"Detected LoRA checkpoint. Base model: {base_model}")
    
    if args.stream:
        stream = True
    elif args.no_stream:
        stream = False
    else:
        stream = not args.concurrent
    infer_args = InferArguments(
        model=base_model,
        adapters=adapters,
        val_dataset=[str(updated_dataset_path)],
        max_new_tokens=8192,
        temperature=0.5,
        repetition_penalty=1.1,
        stream=False,
        max_batch_size=args.max_batch_size,
        model_type=args.model_type,
        infer_backend=args.infer_backend,
        ddp_backend=args.ddp_backend
    )
    
    # 3. Run Inference
    print("Running inference with SwiftInfer...")
    inferer = SwiftInfer(infer_args)
    request_config = infer_args.get_request_config()
    if request_config is not None:
        request_config.n = 4
    val_dataset = inferer._prepare_val_dataset()
    if infer_args.rank >= 0 and infer_args.global_world_size > 1:
        val_dataset = val_dataset.shard(infer_args.global_world_size, infer_args.rank, contiguous=True)
    val_dataset = list(val_dataset)
    labels_list = []
    for data_item in val_dataset:
        if infer_args.task_type == 'causal_lm':
            labels = InferRequest.remove_response(data_item['messages'])
        else:
            labels = data_item.pop('label', None)
        labels_list.append(labels)

    # 4. Process Results
    print("Rendering SVGs...")
    failed_samples = []
    render_workers = args.render_workers
    if render_workers is None:
        render_workers = os.cpu_count() or 1
    render_workers = max(1, int(render_workers))
    global _RENDER_META_MAP
    _RENDER_META_MAP = meta_map
    total = len(val_dataset)
    batch_size = max(1, int(args.max_batch_size))
    if render_workers == 1 or total <= 1:
        with tqdm(total=total, desc="Rendering") as pbar:
            for start in range(0, total, batch_size):
                batch = val_dataset[start:start + batch_size]
                batch_labels = labels_list[start:start + batch_size]
                resp_list = inferer.infer(
                    batch, request_config, template=inferer.template, use_tqdm=False, **inferer.infer_kwargs)
                for offset, (data_item, resp, labels) in enumerate(zip(batch, resp_list, batch_labels)):
                    # response = resp.choices[0].message.content
                    # data_item['messages'].append({'role': 'assistant', 'content': response})
                    # item = {'response': response, 'labels': labels, 'logprobs': resp.choices[0].logprobs, **data_item}
                    # error = _render_single((item, start + offset, str(output_dir)))
                    # if error:
                    #     failed_samples.append(error)
                    for choice_index, choice in enumerate(resp.choices, start=1):
                        response = choice.message.content
                        item_messages = list(data_item['messages'])
                        item_messages.append({'role': 'assistant', 'content': response})
                        item = {'response': response, 'labels': labels, 'logprobs': choice.logprobs, **data_item}
                        item['messages'] = item_messages
                        error = _render_single((item, start + offset, str(output_dir), choice_index))
                        if error:
                            failed_samples.append(error)
                    pbar.update(1)
    else:
        futures = []
        with ProcessPoolExecutor(
            max_workers=render_workers,
            initializer=_init_render_worker,
            initargs=(meta_map,)
        ) as executor:
            for start in range(0, total, batch_size):
                batch = val_dataset[start:start + batch_size]
                batch_labels = labels_list[start:start + batch_size]
                resp_list = inferer.infer(
                    batch, request_config, template=inferer.template, use_tqdm=False, **inferer.infer_kwargs)
                for offset, (data_item, resp, labels) in enumerate(zip(batch, resp_list, batch_labels)):
                    # response = resp.choices[0].message.content
                    # data_item['messages'].append({'role': 'assistant', 'content': response})
                    # item = {'response': response, 'labels': labels, 'logprobs': resp.choices[0].logprobs, **data_item}
                    # futures.append(executor.submit(_render_single, (item, start + offset, str(output_dir))))
                    for choice_index, choice in enumerate(resp.choices, start=1):
                        response = choice.message.content
                        item_messages = list(data_item['messages'])
                        item_messages.append({'role': 'assistant', 'content': response})
                        item = {'response': response, 'labels': labels, 'logprobs': choice.logprobs, **data_item}
                        item['messages'] = item_messages
                        futures.append(executor.submit(
                            _render_single, (item, start + offset, str(output_dir), choice_index)))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering"):
                error = future.result()
                if error:
                    failed_samples.append(error)

    print(f"\nProcessing completed. Total: {total}, Failed: {len(failed_samples)}")
    if failed_samples:
        print("Failures:")
        for fail in failed_samples:
            print(f"  - Sample {fail['index']} ({fail['name']}): {fail['error']}")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
