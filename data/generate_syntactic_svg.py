import os
import json
import tempfile
import shutil
import re
import cairosvg
from PIL import Image
from lxml import etree, html
from lxml.cssselect import CSSSelector
import cssutils
import logging
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

TARGET_SIZE = 512
RENDER_PREVIEW_PNG = False
WORKER_COUNT = int(os.environ.get("BEAGLE_WORKERS", multiprocessing.cpu_count() + 2))
'''
生成 Beagle 数据集的 JSON 文件
简单清洗数据：移除SVG中所有嵌入的位图
'''

# 关闭 cssutils 的烦人日志
cssutils.log.setLevel(logging.CRITICAL)

# --- 1. 定义常见图表库的默认样式 ---
# 这些是 Highcharts, D3, Plotly 等常见的默认风格
DEFAULT_CHART_CSS = """
    /* 坐标轴与网格 */
    .axis path, .axis line, .tick line, .domain {
        fill: none;
        stroke: #333333;
        stroke-width: 1px;
        shape-rendering: crispEdges;
    }
    .grid-line, .grid line, .axis-grid line {
        stroke: #e6e6e6;
        stroke-width: 1px;
        stroke-dasharray: 2, 2;
    }
    /* 文本 */
    text, .axis text, .tick text, .legend text {
        font-family: sans-serif;
        font-size: 10px;
        fill: #333333;
    }
    /* Highcharts 特有 */
    .highcharts-axis-line, .highcharts-tick, .highcharts-grid-line {
        stroke: #ccd6eb;
        fill: none;
    }
    .highcharts-grid-line {
        stroke-dasharray: 2, 2;
    }
    .highcharts-axis-title, .highcharts-title, .highcharts-subtitle {
        fill: #333333;
        font-family: sans-serif;
    }
    /* D3/通用背景 */
    .background, .plot-background {
        fill: #ffffff;
    }
    /* Graphiq specific */
    .gfx-hl-axis .tick line {
        stroke-dasharray: 2, 2;
        stroke: #e6e6e6;
    }
"""

def remove_identity_transforms(svg_str):
    # 移除 transform="matrix(1,0,0,1,0,0)" 以及带小数点的等价形式
    pattern_decimals = re.compile(r'transform="matrix\(1\.0*,\s*0\.0*,\s*0\.0*,\s*1\.0*,\s*0\.0*,\s*0\.0*\)"')
    pattern_integers = re.compile(r'transform="matrix\(\s*1\s*,\s*0\s*,\s*0\s*,\s*1\s*,\s*0\s*,\s*0\s*\)"')
    svg_str = pattern_decimals.sub('', svg_str)
    svg_str = pattern_integers.sub('', svg_str)
    return svg_str

def remove_vector_fonts(root):
    """
    移除 Matplotlib/Seaborn 生成的矢量字体定义及其引用。
    大幅减少 Token (减少约 70%)。
    """
    # 1. 找到所有字体定义的 Path ID (通常包含 DejaVuSans)
    font_def_ids = set()
    # 查找 defs 中的 path
    for path in root.xpath('//*[local-name()="defs"]/*[local-name()="path"]'):
        pid = path.get('id')
        if pid and ('DejaVu' in pid or 'Bitstream' in pid):
            font_def_ids.add(pid)
            path.getparent().remove(path)

    # 2. 移除引用这些字体的 use 标签
    for use in root.xpath('//*[local-name()="use"]'):
        href = use.get('{http://www.w3.org/1999/xlink}href') or use.get('href')
        if href and href.startswith('#'):
            target_id = href[1:]
            if target_id in font_def_ids:
                parent = use.getparent()
                if parent is not None:
                    parent.remove(use)

def remove_accessibility_and_redundant(root):
    """
    清洗无障碍描述 (Accessibility) 和冗余的默认属性。
    减少无关的 Token。
    """
    # 1. 移除 Accessibility 标签: <desc>, <title>
    for tag_name in ['desc', 'title', 'accessibility']:
        # 使用 local-name() 忽略命名空间
        for element in root.xpath(f'//*[local-name()="{tag_name}"]'):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)

    # 2. 遍历所有元素，清理属性
    for element in root.iter():
        attrib = element.attrib
        keys_to_remove = []
        
        for k in list(attrib.keys()):
            # 2.1 移除 aria-* 和 role
            if k.startswith('aria-') or k == 'role':
                keys_to_remove.append(k)
                continue
            
            # 2.2 移除冗余的默认值
            val = attrib[k].strip()
            
            # opacity="1"
            if k in ('opacity', 'fill-opacity', 'stroke-opacity') and val in ('1', '1.0'):
                keys_to_remove.append(k)
            
            # stroke-dasharray="none"
            if k == 'stroke-dasharray' and val == 'none':
                keys_to_remove.append(k)
                
            # stroke-width="0"
            if k == 'stroke-width' and val in ('0', '0.0', '0px'):
                keys_to_remove.append(k)
                # 如果线宽为0，说明没有边框，可以把 stroke 定义也去掉，恢复默认(none)
                if 'stroke' in attrib:
                    keys_to_remove.append('stroke')

        # 执行删除
        for k in set(keys_to_remove):
            if k in attrib:
                del attrib[k]

def remove_markers_from_line_charts(root):
    """
    If a Highcharts series is a line chart (has a visible 'highcharts-graph' line),
    remove its corresponding 'highcharts-markers' group to avoid cluttering the line chart
    with scatter points.
    """
    line_series_indices = set()
    
    # 1. Identify line series
    # Look for groups with 'highcharts-series' class
    for series_group in root.xpath('//*[contains(@class, "highcharts-series")]'):
        classes = series_group.get('class', '').split()
        
        # Check if this group contains a visible graph line
        # Note: We assume inline_css_styles has already moved style="stroke-width:..." to attribute
        graph_lines = series_group.xpath('.//*[contains(@class, "highcharts-graph")]')
        is_line_chart = False
        for line in graph_lines:
            sw = line.get('stroke-width')
            # Check style just in case it wasn't fully inlined or is in style attribute
            if not sw and 'style' in line.attrib:
                style = line.get('style')
                # Simple check for stroke-width in style
                sw_match = re.search(r'stroke-width\s*:\s*([\d\.]+)', style)
                if sw_match:
                    sw = sw_match.group(1)
            
            if sw:
                try:
                    if float(sw) > 0:
                        is_line_chart = True
                        break
                except ValueError:
                    pass
        
        if is_line_chart:
            # Extract series index, e.g., 'highcharts-series-0'
            for cls in classes:
                if cls.startswith('highcharts-series-') and cls[len('highcharts-series-'):].isdigit():
                    line_series_indices.add(cls)

    # 2. Remove marker groups for identified line series
    if line_series_indices:
        for marker_group in root.xpath('//*[contains(@class, "highcharts-markers")]'):
            classes = marker_group.get('class', '').split()
            # If this marker group belongs to a line series, remove it
            if any(idx in classes for idx in line_series_indices):
                parent = marker_group.getparent()
                if parent is not None:
                    parent.remove(marker_group)

def convert_highcharts_paths_to_circles(root):
    """
    Detect Highcharts 'almost circle' paths and convert them to <circle> elements.
    Highcharts represents points as: M x y A r r 0 1 1 x' y' Z where (x,y) ~= (x',y').
    This fixes the issue where rounding coordinates merges start/end points, causing the arc to vanish.
    """
    # Regex to capture M x y A r r 0 1 1 x2 y2 Z
    # We allow some tolerance in whitespace and number format
    pattern = re.compile(
        r'^\s*M\s+([\d\.\-]+)\s+([\d\.\-]+)\s+'
        r'A\s+([\d\.\-]+)\s+([\d\.\-]+)\s+0\s+1\s+1\s+([\d\.\-]+)\s+([\d\.\-]+)\s*Z\s*$',
        re.IGNORECASE
    )

    for elem in root.xpath('//*[local-name()="path"]'):
        d = elem.get('d')
        if not d:
            continue
        
        match = pattern.match(d)
        if match:
            # print(f"DEBUG: Matched path {d}")
            x1, y1, rx, ry, x2, y2 = map(float, match.groups())
            
            # Check if it's a circle (rx == ry) and closed loop (start ~= end)
            if abs(rx - ry) < 0.01 and abs(x1 - x2) < 0.1 and abs(y1 - y2) < 0.1:
                # print(f"DEBUG: Converting to circle: {x1}, {y1}, {rx}")
                # Highcharts starts at the top of the circle (12 o'clock) for sweep=1
                # cx = x1, cy = y1 + rx
                cx = x1
                cy = y1 + rx
                
                # Create new circle element
                circle = etree.Element('circle')
                for k, v in elem.attrib.items():
                    if k != 'd':
                        circle.set(k, v)
                
                circle.set('cx', str(cx))
                circle.set('cy', str(cy))
                circle.set('r', str(rx))
                
                # Replace path with circle
                parent = elem.getparent()
                if parent is not None:
                    parent.replace(elem, circle)


def inline_css_styles(svg_content, default_css=DEFAULT_CHART_CSS, file_path="<unknown>"):
    """
    将 CSS 类样式转换为内联 style 属性，并注入默认图表样式。
    """
    if not svg_content or not svg_content.strip():
        return svg_content

    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    try:
        if isinstance(svg_content, str):
            svg_content_bytes = svg_content.encode('utf-8')
        else:
            svg_content_bytes = svg_content
        root = etree.fromstring(svg_content_bytes, parser)
    except Exception as e:
        print(f"[WARN] Parsing failed for {file_path}: {e}")
        return svg_content

    # 1. 初始化 CSS 解析器
    # 如果 SVG 内部本身就有 <style>，也把它提取出来
    existing_styles = []
    for style_tag in root.xpath('//*[local-name()="style"]'):
        if style_tag.text:
            existing_styles.append(style_tag.text)
    
    # 合并：默认样式 + 原有样式 (原有样式优先)
    combined_css = default_css + "\n" + "\n".join(existing_styles)
    
    try:
        sheet = cssutils.parseString(combined_css)
    except Exception:
        return svg_content

    # 2. 遍历 CSS 规则并应用到元素（使用 lxml.cssselect 支持复杂选择器）
    for rule in sheet:
        if rule.type != rule.STYLE_RULE:
            continue
        selectors = [s.strip() for s in rule.selectorText.split(',') if s.strip()]
        style_decl = rule.style
        for sel in selectors:
            try:
                xpath_selector = CSSSelector(sel)
                matched = xpath_selector(root)
            except Exception:
                # Fallback or ignore invalid selectors
                continue

            for elem in matched:
                old_style_str = elem.get('style', '')
                style_dict = {}
                if old_style_str:
                    for item in old_style_str.split(';'):
                        if ':' in item:
                            k, v = item.split(':', 1)
                            style_dict[k.strip().lower()] = v.strip()
                for prop in style_decl:
                    if prop.name not in style_dict:
                        style_dict[prop.name] = prop.value
                new_style_str = "; ".join([f"{k}: {v}" for k, v in style_dict.items()])
                elem.set('style', new_style_str)

    # 3. 去除原始 <style> 标签，避免重复样式影响
    for style_tag in root.xpath('//*[local-name()="style"]'):
        parent = style_tag.getparent()
        if parent is not None:
            parent.remove(style_tag)

    return etree.tostring(root, encoding='unicode')


def _strip_image_elements(svg_text: str) -> str:
    pattern_self_closing = re.compile(r'<image[^>]*/>', re.IGNORECASE | re.DOTALL)
    svg_text = pattern_self_closing.sub('', svg_text)
    pattern_pair = re.compile(r'<image[^>]*>.*?</image>', re.IGNORECASE | re.DOTALL)
    svg_text = pattern_pair.sub('', svg_text)
    return svg_text


def _strip_branding_blocks(svg_text: str) -> str:
    # Remove elements with specific branding classes
    # 1. Regex based on class names
    for tag in ['text', 'g', 'a']:
        # Match class="...branding..." or class="...highcharts-credits..."
        # Also match FusionCharts/Raphael branding classes like "raphael-group-1-messageGroup" or "creditgroup"
        pattern = re.compile(
            rf'<{tag}[^>]*class="[^"]*(?:highcharts-credits|branding|creditgroup|messageGroup)[^"]*"[^>]*>.*?</{tag}>',
            re.IGNORECASE | re.DOTALL
        )
        svg_text = pattern.sub('', svg_text)
    
    # 2. Regex based on text content (for cases without classes or with generic classes)
    # Remove elements containing specific branding text
    branding_texts = [
        "FusionCharts XT Trial",
        "Loading chart. Please wait",
        "Created with Highcharts",
        "www.highcharts.com"
    ]
    
    for b_text in branding_texts:
        # Match any tag containing the branding text
        # This is a bit aggressive, but safer for cleaning
        # We try to match the innermost container first
        pattern_text = re.compile(
            rf'<[^>]+>[^<]*{re.escape(b_text)}[^<]*</[^>]+>',
            re.IGNORECASE
        )
        svg_text = pattern_text.sub('', svg_text)

    # Also remove generic <a> tags that link to external sites if they look like branding
    return svg_text


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
        return new_img, scale, pad_x, pad_y

def _scale_transform_str(transform_str, scale):
    if not transform_str:
        return transform_str
    
    # Pattern to match commands like translate(10, 20) or matrix(...)
    pattern = re.compile(r'([a-zA-Z]+)\s*\(([^)]*)\)')
    
    def repl(match):
        cmd = match.group(1).lower()
        args_str = match.group(2)
        raw_args = re.split(r'[,\s]+', args_str.strip())
        args = []
        for x in raw_args:
            if not x:
                continue
            clean_x = x.lower().replace('deg', '').replace('px', '').replace('null', '0')
            try:
                args.append(float(clean_x))
            except ValueError:
                args.append(0.0)
        
        if not args:
            return match.group(0)

        new_args = []
        
        if cmd == 'translate':
            new_args.append(round(args[0] * scale, 1))
            if len(args) > 1:
                new_args.append(round(args[1] * scale, 1))
                
        elif cmd == 'matrix':
            if len(args) == 6:
                new_args = [
                    args[0], args[1], args[2], args[3], # a, b, c, d
                    round(args[4] * scale, 1),          # e (tx)
                    round(args[5] * scale, 1)           # f (ty)
                ]
            else:
                return match.group(0) # Invalid matrix
                
        elif cmd == 'rotate':
            new_args.append(args[0]) # angle
            if len(args) > 1:
                new_args.append(round(args[1] * scale, 1))
            if len(args) > 2:
                new_args.append(round(args[2] * scale, 1))

        elif cmd == 'scale':
            # Scale factors are dimensionless. Rounding them to one decimal can
            # visibly resize the whole chart (for example 0.6583 -> 0.7).
            new_args = [round(x, 8) for x in args]
                
        else:
            new_args = [round(x, 1) if isinstance(x, (int, float)) else x for x in args]
            
        args_joined = ", ".join(map(str, new_args))
        return f"{match.group(1)}({args_joined})"

    return pattern.sub(repl, transform_str)


def _scale_style_str(style_str, scale):
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
                val_clean = val.lower().strip()
                factor = 1.0
                if val_clean.endswith('px'):
                    val_clean = val_clean[:-2]
                elif val_clean.endswith('rem'):
                    val_clean = val_clean[:-3]
                    factor = 16.0
                elif val_clean.endswith('em'):
                    val_clean = val_clean[:-2]
                    factor = 16.0
                
                val_float = float(val_clean)
                new_val = round(val_float * factor * scale, 1)
                new_parts.append(f"{key}: {new_val}px")
            except ValueError:
                new_parts.append(part)
        else:
            new_parts.append(part)
            
    return "; ".join(new_parts)


def _get_svg_dimensions(root):
    def parse_unit(val):
        if not val: return None
        val = val.lower().strip()
        factor = 1.0
        if val.endswith('px'):
            val = val[:-2]
        elif val.endswith('pt'):
            val = val[:-2]
            factor = 1.3333
        elif val.endswith('em') or val.endswith('rem'):
             val = val[:-2] if val.endswith('em') else val[:-3]
             factor = 16.0
        elif val.endswith('%'):
            return None
        try:
            return float(val) * factor
        except ValueError:
            return None

    if 'viewBox' in root.attrib:
        try:
            vb = root.attrib['viewBox']
            parts = [float(x) for x in re.split(r'[,\s]+', vb.strip()) if x]
            if len(parts) == 4:
                return parts[2], parts[3]
        except Exception:
            pass

    w = parse_unit(root.get('width'))
    h = parse_unit(root.get('height'))

    if w and h:
        return w, h
            
    return None, None



def calculate_scale_from_svg(svg_path, target_size=TARGET_SIZE):
    """
    当 PNG 不存在或无法读取时，从 SVG 文件本身计算缩放比例。
    """
    try:
        with open(svg_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单过滤 HTML/DOCTYPE 头部，避免解析报错
        if content.strip().lower().startswith('<!doctype html'):
             return 1.0, 0.0, 0.0

        parser = etree.XMLParser(recover=True)
        if isinstance(content, str):
            content = content.encode('utf-8')
        root = etree.fromstring(content, parser)
        
        w, h = _get_svg_dimensions(root)
        if w and h and max(w, h) > 0:
            scale = target_size / max(w, h)
            w_new = w * scale
            h_new = h * scale
            pad_x = (target_size - w_new) / 2
            pad_y = (target_size - h_new) / 2
            return scale, pad_x, pad_y
            
    except Exception as e:
        # print(f"[WARN] Failed to calculate scale from SVG {svg_path}: {e}")
        pass
        
    return 1.0, 0.0, 0.0


def normalize_svg(svg_content, scale, pad_x, pad_y, file_path="<unknown>"):
    if not svg_content or not svg_content.strip():
        return svg_content

    parser = etree.XMLParser(remove_blank_text=True, remove_comments=True, recover=True)
    try:
        if isinstance(svg_content, str):
            svg_content_bytes = svg_content.encode('utf-8')
        else:
            svg_content_bytes = svg_content
        root = etree.fromstring(svg_content_bytes, parser)
    except etree.XMLSyntaxError as e:
        print(f"[WARN] SVG parsing failed during normalization for {file_path}: {e}")
        return svg_content

    # Remove invalid placeholder values that CairoSVG cannot parse as numbers
    # or paint properties (common in a small number of Graphiq exports).
    invalid_values = {'null', 'undefined', 'nan'}
    for elem in root.iter():
        for attr, value in list(elem.attrib.items()):
            if value.strip().lower() in invalid_values:
                del elem.attrib[attr]
        if 'style' in elem.attrib:
            style = re.sub(
                r'(?:^|;)\s*[-\w]+\s*:\s*(?:null|undefined|nan)\s*(?=;|$)',
                '',
                elem.attrib['style'],
                flags=re.IGNORECASE,
            ).strip(' ;')
            if style:
                elem.attrib['style'] = style
            else:
                del elem.attrib['style']

    # 调用移除矢量字体的函数
    remove_vector_fonts(root)

    # 移除无障碍描述和冗余属性
    remove_accessibility_and_redundant(root)
    
    # 对于折线图，移除冗余的散点标记
    remove_markers_from_line_charts(root)

    # 转换 Highcharts 路径为圆 (修复精度丢失问题)
    convert_highcharts_paths_to_circles(root)

    svg_ns = 'http://www.w3.org/2000/svg'
    etree.strip_elements(root, f'{{{svg_ns}}}image', with_tail=False)
    etree.strip_elements(root, 'image', with_tail=False)

    branding_nodes = root.xpath(".//*[contains(@class, 'branding') or contains(@class, 'highcharts-credits') or contains(@class, 'gfx-mouse-target')]")
    for node in branding_nodes:
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)

    refs = set()
    for elem in root.iter():
        for val in elem.values():
            if val and isinstance(val, str) and '#' in val:
                ids = re.findall(r'#([\w.-]+)', val)
                refs.update(ids)
    
    for elem in root.iter():
        if 'id' in elem.attrib:
            if elem.attrib['id'] not in refs:
                del elem.attrib['id']

    def transform_val(val, is_dim=False):
        try:
            v_str = val.lower().strip()
            factor = 1.0
            if v_str.endswith('px'):
                v_str = v_str[:-2]
            elif v_str.endswith('rem'):
                v_str = v_str[:-3]
                factor = 16.0
            elif v_str.endswith('em'):
                v_str = v_str[:-2]
                factor = 16.0

            v = float(v_str)
            return round(v * factor * scale, 1)
        except ValueError:
            return val

    attr_map = {
        'x': 'x', 'cx': 'x', 'x1': 'x', 'x2': 'x',
        'y': 'y', 'cy': 'y', 'y1': 'y', 'y2': 'y',
        'width': 'dim', 'height': 'dim', 'r': 'dim', 'rx': 'dim', 'ry': 'dim',
        'font-size': 'dim', 'stroke-width': 'dim'
    }

    for elem in root.iter():
        for attr, type_ in attr_map.items():
            if attr in elem.attrib:
                val = elem.attrib[attr]
                new_val = transform_val(val)
                elem.attrib[attr] = str(new_val)
        
        if 'd' in elem.attrib:
            elem.attrib['d'] = transform_path_d(elem.attrib['d'], scale, 0, 0)
            
        if 'points' in elem.attrib:
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', elem.attrib['points'])
            new_nums = []
            for i in range(0, len(nums), 2):
                if i+1 < len(nums):
                    # Padding is applied once by the outer content group.
                    x = float(nums[i]) * scale
                    y = float(nums[i+1]) * scale
                    new_nums.append(f"{round(x,1)},{round(y,1)}")
            elem.attrib['points'] = " ".join(new_nums)
            
        if 'transform' in elem.attrib:
            elem.attrib['transform'] = _scale_transform_str(elem.attrib['transform'], scale)
            
        if 'style' in elem.attrib:
            elem.attrib['style'] = _scale_style_str(elem.attrib['style'], scale)

        if elem != root and 'viewBox' in elem.attrib:
            try:
                parts = re.split(r'[,\s]+', elem.attrib['viewBox'].strip())
                parts = [float(x) for x in parts if x and x.lower() != 'null']
                if len(parts) == 4:
                    new_parts = [round(x * scale, 1) for x in parts]
                    elem.attrib['viewBox'] = " ".join(map(str, new_parts))
            except ValueError:
                pass

    vx, vy = 0.0, 0.0

    if 'viewBox' in root.attrib: # Fixed: Check root attrib instead of loop variable 'elem'
        raw_vb = root.attrib['viewBox'].strip()
        parts = [float(x) for x in re.split(r'[,\s]+', raw_vb) if x and x.lower() != 'null']
        if len(parts) == 4:
            vx, vy = parts[0], parts[1]

    keys_to_remove = ['width', 'height', 'viewBox']
    for k in keys_to_remove:
        if k in root.attrib:
            del root.attrib[k]
    
    # 强制设置 512x512 画布
    root.attrib['viewBox'] = f"0 0 {TARGET_SIZE} {TARGET_SIZE}"
    root.attrib['width'] = str(TARGET_SIZE)
    root.attrib['height'] = str(TARGET_SIZE)

    # 添加白色背景
    bg_rect = etree.Element('rect')
    bg_rect.attrib['width'] = str(TARGET_SIZE)
    bg_rect.attrib['height'] = str(TARGET_SIZE)
    bg_rect.attrib['x'] = "0"
    bg_rect.attrib['y'] = "0"
    bg_rect.attrib['fill'] = "white"
    
    # 将背景插入到最前面
    root.insert(0, bg_rect)

    # 坐标已经缩放，但上面的属性/路径处理没有加入 pad_x/pad_y。
    # 用一个统一平移组将内容居中；背景和 defs 保留在根节点下。
    if abs(pad_x) > 1e-6 or abs(pad_y) > 1e-6:
        content_group = etree.Element('g')
        content_group.attrib['transform'] = f"translate({round(pad_x, 1)} {round(pad_y, 1)})"
        movable_children = [
            child for child in list(root)
            if child is not bg_rect and etree.QName(child).localname != 'defs'
        ]
        for child in movable_children:
            root.remove(child)
            content_group.append(child)
        root.append(content_group)
    
    svg_out = etree.tostring(root, encoding='unicode', pretty_print=False)

    if 'xmlns=' not in svg_out:
        svg_out = svg_out.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)

    if 'xlink:' in svg_out and 'xmlns:xlink' not in svg_out:
        svg_out = svg_out.replace('<svg', '<svg xmlns:xlink="http://www.w3.org/1999/xlink"', 1)

    svg_out = remove_identity_transforms(svg_out)
    # Do not globally truncate decimals here. Dimensionless transform values
    # such as matrix(0.0858, ...) would become matrix(0.0, ...), making entire
    # layers disappear. Coordinate attributes are already rounded explicitly.

    return svg_out

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

def transform_path_d(d, scale, pad_x, pad_y):
    if not d or len(d) < 2:
        return d

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
        num_sets = available_args // n_args
        if num_sets > 0:
            chunk_len = num_sets * n_args
            args_chunk = flat_tokens[start_arg_idx : start_arg_idx + chunk_len]
            args_chunk = [x if isinstance(x, (int, float)) else 0.0 for x in args_chunk]
            
            is_rel = (current_cmd == current_cmd.lower())
            processed_chunk = []
            
            if cmd_upper in ('M', 'L', 'T'):
                if is_rel:
                    for j in range(0, chunk_len, 2):
                        processed_chunk.append(str(round(args_chunk[j] * scale, 1)))
                        processed_chunk.append(str(round(args_chunk[j+1] * scale, 1)))
                else:
                    for j in range(0, chunk_len, 2):
                        processed_chunk.append(str(round(args_chunk[j] * scale + pad_x, 1)))
                        processed_chunk.append(str(round(args_chunk[j+1] * scale + pad_y, 1)))
                        
            elif cmd_upper == 'H':
                if is_rel:
                    for x in args_chunk:
                        processed_chunk.append(str(round(x * scale, 1)))
                else:
                    for x in args_chunk:
                        processed_chunk.append(str(round(x * scale + pad_x, 1)))
                        
            elif cmd_upper == 'V':
                if is_rel:
                    for y in args_chunk:
                        processed_chunk.append(str(round(y * scale, 1)))
                else:
                    for y in args_chunk:
                        processed_chunk.append(str(round(y * scale + pad_y, 1)))
                        
            elif cmd_upper in ('C', 'S', 'Q'):
                for j in range(0, chunk_len, 2):
                    x = args_chunk[j]
                    y = args_chunk[j+1]
                    if is_rel:
                        processed_chunk.append(str(round(x * scale, 1)))
                        processed_chunk.append(str(round(y * scale, 1)))
                    else:
                        processed_chunk.append(str(round(x * scale + pad_x, 1)))
                        processed_chunk.append(str(round(y * scale + pad_y, 1)))
            
            elif cmd_upper == 'A':
                for j in range(0, chunk_len, 7):
                    rx = args_chunk[j]
                    ry = args_chunk[j+1]
                    rot = args_chunk[j+2]
                    large = args_chunk[j+3]
                    sweep = args_chunk[j+4]
                    x = args_chunk[j+5]
                    y = args_chunk[j+6]
                    processed_chunk.append(str(round(rx * scale, 1)))
                    processed_chunk.append(str(round(ry * scale, 1)))
                    processed_chunk.append(str(rot))
                    processed_chunk.append(str(int(float(large))))
                    processed_chunk.append(str(int(float(sweep))))
                    if is_rel:
                        processed_chunk.append(str(round(x * scale, 1)))
                        processed_chunk.append(str(round(y * scale, 1)))
                    else:
                        processed_chunk.append(str(round(x * scale + pad_x, 1)))
                        processed_chunk.append(str(round(y * scale + pad_y, 1)))
            
            res.extend(processed_chunk)
            i += chunk_len
        else:
            i = end_arg_idx
            
    return " ".join(res)

def process_single_chart(args):
    """
    处理单个图表的任务函数
    """
    root_dir, file_name, charts_root, images_root = args
    
    # 用户要求：只考虑从原始 svg.txt 生成 cleaned_svg.txt
    # 之前添加的 new_svg.txt 逻辑已移除
    
    svg_txt_path = os.path.join(charts_root, file_name, 'svg.txt')
    # svg_txt_path = os.path.join(charts_root, f"{file_name}.txt")
    if not os.path.isfile(svg_txt_path):
        return f"[WARN] svg.txt missing: {svg_txt_path}"
    
    # Check for HTML content disguised as SVG
    try:
        with open(svg_txt_path, 'r', encoding='utf-8') as sf:
            first_chars = sf.read(500).strip().lower()
            if (first_chars.startswith('<!doctype html') or 
                '<html' in first_chars or 
                'parsererror' in first_chars or
                'xml parsing error' in first_chars):
                 return f"[WARN] Skipping invalid/HTML file disguised as SVG: {svg_txt_path}"
    except Exception:
        pass

    png_path = os.path.join(images_root, f"{file_name}.png")
    scale, pad_x, pad_y = 1.0, 0.0, 0.0
    
    if os.path.isfile(png_path):
        try:
            _, scale, pad_x, pad_y = normalize_image(png_path, TARGET_SIZE)
        except Exception as e:
            # print(f"[WARN] Image normalize failed {png_path}: {e}; fallback to SVG parsing")
            scale, pad_x, pad_y = calculate_scale_from_svg(svg_txt_path, TARGET_SIZE)
    else:
        # print(f"[WARN] image missing: {png_path}; fallback to SVG parsing")
        scale, pad_x, pad_y = calculate_scale_from_svg(svg_txt_path, TARGET_SIZE)
        
    try:
        with open(svg_txt_path, 'r', encoding='utf-8') as sf:
            svg_code = sf.read()
        
        if not svg_code or not svg_code.strip():
            return None 

        cleaned = inline_css_styles(svg_code, file_path=svg_txt_path)
        cleaned = _strip_branding_blocks(cleaned)
        cleaned = _strip_image_elements(cleaned)
        cleaned = normalize_svg(cleaned, scale, pad_x, pad_y, file_path=svg_txt_path)
    except Exception as e:
        return f"[ERR] Failed to normalize SVG {svg_txt_path}: {e}"
            
    cleaned_out = os.path.join(charts_root, file_name, 'cleaned_svg.txt')
    try:
        with open(cleaned_out, 'w', encoding='utf-8') as wf:
            wf.write(cleaned)
    except Exception as e:
        return f"[ERR] Failed writing {cleaned_out}: {e}"

    if RENDER_PREVIEW_PNG:
        rendered_png_out = os.path.join(charts_root, file_name, f"{file_name}.png")
        try:
            if not cleaned or not cleaned.strip():
                 return f"[WARN] Cleaned SVG is empty for {file_name}"

            # Provide a base URL pointing to the current svg.txt to avoid directory fetch issues
            # Older versions of cairosvg do not support url_fetcher; use 'url' for base resolution
            try:
                cairosvg.svg2png(bytestring=cleaned.encode('utf-8'), write_to=rendered_png_out, url=svg_txt_path)
            except TypeError:
                # Fallback for environments where signature differs
                cairosvg.svg2png(bytestring=cleaned.encode('utf-8'), write_to=rendered_png_out)
        except Exception as e:
            return f"[ERR] Failed to render SVG to PNG {rendered_png_out}: {e}"
    
    return None # Success

def write_cleaned_svgs_for_roots(base_dir: str):
    all_dataset_names = ['fusion_clean', 'plotly_export', 'graphiq_clean', 'chartblocks', 'echarts']
    selected_dataset_names = [
        name.strip() for name in os.environ.get(
            'BEAGLE_DATASETS', ','.join(all_dataset_names)
        ).split(',') if name.strip()
    ]
    selected_ids = {
        name.strip() for name in os.environ.get('BEAGLE_IDS', '').split(',') if name.strip()
    }
    force = os.environ.get('BEAGLE_FORCE', '').lower() in {'1', 'true', 'yes'}
    roots = [os.path.join(base_dir, name) for name in selected_dataset_names]
    
    all_tasks = []
    
    print("Collecting tasks...")
    for root_dir in roots:
        urls_path = os.path.join(root_dir, 'urls.txt')
        charts_root = os.path.join(root_dir, 'charts')
        images_root = os.path.join(root_dir, 'images')
        
        if os.path.isfile(urls_path):
            try:
                with open(urls_path, 'r', encoding='utf-8') as f:
                    file_names = [
                        line.strip().split()[0]
                        for line in f
                        if line.strip()
                    ]
            except Exception as e:
                print(f"[ERR] Failed reading {urls_path}: {e}")
                continue
        else:
            # Crawled datasets such as ECharts already use the canonical
            # charts/<id>/svg.txt + images/<id>.png layout and do not need a
            # redundant urls.txt manifest. Only process complete SVG/PNG pairs.
            file_names = sorted(
                name for name in os.listdir(charts_root)
                if os.path.isfile(os.path.join(charts_root, name, 'svg.txt'))
                and os.path.isfile(os.path.join(images_root, f'{name}.png'))
            ) if os.path.isdir(charts_root) and os.path.isdir(images_root) else []
            print(
                f"[INFO] urls.txt not found; discovered {len(file_names)} "
                f"complete chart/image pairs in {root_dir}"
            )

        for file_name in file_names:

            if selected_ids and file_name not in selected_ids:
                continue

            cleaned_out = os.path.join(charts_root, file_name, 'cleaned_svg.txt')
            if os.path.isfile(cleaned_out) and not force:
                continue
            
            # Prepare args for multiprocessing
            all_tasks.append((root_dir, file_name, charts_root, images_root))

    print(f"Total tasks found: {len(all_tasks)}")
    print(f"Starting processing with {WORKER_COUNT} processes...")
    
    if WORKER_COUNT == 1:
        results = [process_single_chart(task) for task in tqdm(all_tasks)]
    else:
        with Pool(processes=WORKER_COUNT) as pool:
            # Use tqdm to show progress bar
            results = list(tqdm(pool.imap(process_single_chart, all_tasks), total=len(all_tasks)))
        
    # Print errors after progress bar finishes
    errors = [res for res in results if res is not None]
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for err in errors[:20]: # Show first 20 errors
            print(err)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors.")
    else:
        print("\nAll tasks completed successfully.")

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Beagle')
    write_cleaned_svgs_for_roots(base_dir)
