# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2024, XiMing Xing.
# License: MIT License
# Description: SVG Semantic Tokens encoder and decoder
# Fixed version for Chart2SVG task (V21 - Hierarchy Separation Fix)

import re
from typing import Union, List, Tuple
from lxml import etree
from xml.sax.saxutils import escape, quoteattr

"""Define SVG Mappers and Identifiers"""

PathCMDMapper = {
    '[m]': '[<|moveto_rel|>]', '[M]': '[<|moveto_abs|>]',
    '[l]': '[<|lineto_rel|>]', '[L]': '[<|lineto_abs|>]',
    '[h]': '[<|horizontal_lineto_rel|>]', '[H]': '[<|horizontal_lineto_abs|>]',
    '[v]': '[<|vertical_lineto_rel|>]', '[V]': '[<|vertical_lineto_abs|>]',
    '[c]': '[<|curveto_rel|>]', '[C]': '[<|curveto_abs|>]',
    '[s]': '[<|smooth_curveto_rel|>]', '[S]': '[<|smooth_curveto_abs|>]',
    '[q]': '[<|quadratic_bezier_curve_rel|>]', '[Q]': '[<|quadratic_bezier_curve_abs|>]',
    '[t]': '[<|smooth_quadratic_bezier_curveto_rel|>]', '[T]': '[<|smooth_quadratic_bezier_curveto_abs|>]',
    '[a]': '[<|elliptical_arc_rel|>]', '[A]': '[<|elliptical_arc_abs|>]',
    '[z]': '[<|close_path|>]', '[Z]': '[<|close_path|>]',
}

PathCMDIdentifier = {
    'moveto_rel': 'm', 'moveto_abs': 'M',
    'lineto_rel': 'l', 'lineto_abs': 'L',
    'horizontal_lineto_rel': 'h', 'horizontal_lineto_abs': 'H',
    'vertical_lineto_rel': 'v', 'vertical_lineto_abs': 'V',
    'curveto_rel': 'c', 'curveto_abs': 'C',
    'smooth_curveto_rel': 's', 'smooth_curveto_abs': 'S',
    'quadratic_bezier_curve_rel': 'q', 'quadratic_bezier_curve_abs': 'Q',
    'smooth_quadratic_bezier_curveto_rel': 't', 'smooth_quadratic_bezier_curveto_abs': 'T',
    'elliptical_arc_rel': 'a', 'elliptical_arc_abs': 'A',
    'close_path': 'z',
}

AttribMapper = {
    'id=': '[<|id=|>]', 'd=': '[<|d=|>]',
    'fill=': '[<|fill=|>]', 'stroke-width=': '[<|stroke-width=|>]',
    'stroke-linecap=': '[<|stroke-linecap=|>]', 'stroke=': '[<|stroke=|>]',
    'opacity=': '[<|opacity=|>]', 'fill-opacity=': '[<|fill-opacity=|>]',
    'stroke-opacity=': '[<|stroke-opacity=|>]', 'stroke-dasharray=': '[<|stroke-dasharray=|>]',
    'transform=': '[<|transform=|>]', 'gradientTransform=': '[<|gradientTransform=|>]',
    'offset=': '[<|offset=|>]', 'width=': '[<|width=|>]', 'height=': '[<|height=|>]',
    'cx=': '[<|cx=|>]', 'cy=': '[<|cy=|>]', 'rx=': '[<|rx=|>]', 'ry=': '[<|ry=|>]',
    'r=': '[<|r=|>]', 'points=': '[<|points=|>]',
    'x1=': '[<|x1=|>]', 'y1=': '[<|y1=|>]', 'x2=': '[<|x2=|>]', 'y2=': '[<|y2=|>]',
    'x=': '[<|x=|>]', 'y=': '[<|y=|>]', 'dx=': '[<|dx=|>]', 'dy=': '[<|dy=|>]',
    'fr=': '[<|fr=|>]', 'fx=': '[<|fx=|>]', 'fy=': '[<|fy=|>]', 'href=': '[<|href=|>]',
    'rotate=': '[<|rotate=|>]', 'font-size=': '[<|font-size=|>]',
    'font-style=': '[<|font-style=|>]', 'font-family=': '[<|font-family=|>]',
    'text-anchor=': '[<|text-anchor=|>]', 'text-content=': '[<|text-content=|>]',
    'preserveAspectRatio=': '[<|preserveAspectRatio=|>]', 'viewBox=': '[<|viewBox=|>]',
    'class=': '[<|class=|>]', 'clip-path=': '[<|clip-path=|>]',
    'stop-color=': '[<|stop-color=|>]', 'stop-opacity=': '[<|stop-opacity=|>]',
    'style=': '[<|style=|>]', 'font-weight=': '[<|font-weight=|>]',
    'visibility=': '[<|visibility=|>]', 'display=': '[<|display=|>]'
}

SVGToken = {'start': '[<|START_OF_SVG|>]', 'end': '[<|END_OF_SVG|>]'}

ContainerMapper = {
    '<svg>': SVGToken['start'], '</svg>': SVGToken['end'],
    '<g>': '[<|START_OF_GROUP|>]', '</g>': '[<|END_OF_GROUP|>]',
    '<clipPath>': '[<|clipPath|>]', '</clipPath>': '[<|/clipPath|>]',
    '<defs>': '[<|defs|>]', '</defs>': '[<|/defs|>]',
    '<text>': '[<|text|>]', '</text>': '[<|/text|>]',
    '<tspan>': '[<|tspan|>]', '</tspan>': '[<|/tspan|>]',
}

ContainerTagIdentifiers = {
    'svg_start': 'START_OF_SVG', 'svg_end': 'END_OF_SVG',
    'g_start': 'START_OF_GROUP', 'g_end': 'END_OF_GROUP',
    'clipPath': 'clipPath', 'clipPath_end': '/clipPath',
    'defs': 'defs', 'defs_end': '/defs',
}

PathMapper = {'<path>': '[<|path|>]', '</path>': '[<|/path|>]'}
PathIdentifier = 'path'

GradientsMapper = {
    '<linearGradient>': '[<|linearGradient|>]', '</linearGradient>': '[<|/linearGradient|>]',
    '<radialGradient>': '[<|radialGradient|>]', '</radialGradient>': '[<|/radialGradient|>]',
    '<stop>': '[<|stop|>]', '</stop>': '[<|/stop|>]',
}
GradientIdentifier = {'linear_gradient': 'linearGradient', 'radial_gradient': 'radialGradient', 'stop': 'stop'}

ShapeMapper = {
    '<circle>': '[<|circle|>]', '</circle>': '[<|/circle|>]',
    '<rect>': '[<|rect|>]', '</rect>': '[<|/rect|>]',
    '<ellipse>': '[<|ellipse|>]', '</ellipse>': '[<|/ellipse|>]',
    '<polygon>': '[<|polygon|>]', '</polygon>': '[<|/polygon|>]',
    '<line>': '[<|line|>]', '</line>': '[<|/line|>]',
    '<polyline>': '[<|polyline|>]', '</polyline>': '[<|/polyline|>]',
    '<text>': '[<|text|>]', '</text>': '[<|/text|>]',
    '<tspan>': '[<|tspan|>]', '</tspan>': '[<|/tspan|>]',
}
ShapeIdentifier = {
    'circle': 'circle', 'rect': 'rect', 'ellipse': 'ellipse', 'polygon': 'polygon',
    'line': 'line', 'polyline': 'polyline', 'text': 'text', 'tspan': 'tspan',
}

def remove_square_brackets(s: str) -> str: return s.replace('[', '').replace(']', '')
def is_path_closed(path_d: str) -> bool: return path_d.strip().endswith(('Z', 'z'))

# ==============================================================================
# ENCODER (V21)
# ==============================================================================
def svg2syntactic(
        svg_string: str, include_gradient_tag: bool = False, path_only: bool = False,
        include_group: bool = True, group_attr_inherit: bool = True,
        ignore_tags: List[str] = [], ignore_attrs: List[str] = ['gradientUnits']
) -> Tuple[str, str]:
    
    parser = etree.XMLParser(remove_blank_text=True)
    try:
        if isinstance(svg_string, str): tree = etree.fromstring(svg_string.encode('utf-8'), parser)
        else: tree = etree.fromstring(svg_string, parser)
    except Exception: return "", ""

    struct_ret = ""
    shape_tags = ['circle', 'rect', 'ellipse', 'polygon', 'line', 'polyline', 'use']
    gradient_tags = ['linearGradient', 'radialGradient', 'stop']
    ALLOWED_TAGS = set(shape_tags + gradient_tags + ['path', 'text', 'tspan', 'svg', 'g', 'clipPath', 'defs', 'use'])

    basic_shape_attrs = [
        'fill', 'stroke-width', 'stroke', 'opacity', 'transform',
        'cx', 'cy', 'r', 'rx', 'ry', 'width', 'height', 'points',
        'x1', 'y1', 'x2', 'y2', 'x', 'y', 'dx', 'dy', 'rotate', 'font-size',
        'textLength', 'font-style', 'font-family',
        'fill-opacity', 'stroke-opacity', 'stroke-dasharray',
        'text-anchor', 'preserveAspectRatio', 'viewBox', 'class', 'clip-path',
        'href', 'id', 'style', 'visibility', 'display', 'dominant-baseline'
    ]

    NON_INHERITABLE = {
        'transform', 'clip-path', 'opacity', 'mask', 'filter', 'id', 'class',
        'x', 'y', 'dx', 'dy', 'width', 'height', 'cx', 'cy', 'r', 'rx', 'ry',
        'x1', 'y1', 'x2', 'y2', 'd', 'points', 'viewBox', 'href', 'style'
    }

    def recursive_parse(element, level=0, inherited_attributes=None):
        nonlocal struct_ret
        try: tag = etree.QName(element).localname
        except: return

        # Treat nested <svg> (legend/annotation overlays) as groups to avoid
        # creating additional viewports that can trap siblings inside.
        if tag == 'svg' and level > 0:
            tag = 'g'

        inherited_attributes = inherited_attributes or {}
        if tag not in ALLOWED_TAGS and tag not in ignore_tags: return
        if tag in ignore_tags: return

        if tag == 'defs':
            struct_ret += "  " * level + f"<{tag}>\n"
            for child in element: recursive_parse(child, level + 1, inherited_attributes)
            struct_ret += "  " * level + f"</{tag}>\n"; return

        current_attributes = inherited_attributes.copy()
        if tag == 'g':
            if group_attr_inherit:
                inheritable = {k: v for k, v in element.attrib.items() if k not in NON_INHERITABLE}
                current_attributes.update(inheritable)

            if include_group:
                struct_ret += "  " * level + f"<{tag}>\n"
                group_attribs = dict(element.attrib)
                if 'style' in group_attribs:
                    styles = _parse_style_string(group_attribs['style'])
                    group_attribs.update(styles)

                attr_str = ""
                for k in basic_shape_attrs:
                    if k in group_attribs and k not in ignore_attrs:
                        v = group_attribs[k]
                        v = str(v)
                        attr_str += f" {k}={v}"
                
                if attr_str: struct_ret += "  " * (level + 1) + attr_str.strip() + "\n"

            for child in element: recursive_parse(child, level + 1, current_attributes)
            if include_group: struct_ret += "  " * level + f"</{tag}>\n"
            return

        if tag == 'clipPath':
            struct_ret += "  " * level + f"<{tag}>"
            if 'id' in element.attrib: struct_ret += f" id={element.attrib['id']}"
            elif 'class' in element.attrib: struct_ret += f" class={element.attrib['class']}"
            struct_ret += "\n"
            for child in element: recursive_parse(child, level + 1, current_attributes)
            struct_ret += "  " * level + f"</{tag}>\n"; return

        attributes = {**current_attributes, **element.attrib}
        if '{http://www.w3.org/1999/xlink}href' in attributes:
            attributes['href'] = attributes['{http://www.w3.org/1999/xlink}href']

        if 'style' in attributes:
            styles = _parse_style_string(attributes['style'])
            for key in basic_shape_attrs:
                if key in styles: attributes[key] = styles[key]

        if tag == "text" or tag == "tspan":
            struct_ret += "  " * level + f"<{tag}>\n"
            text_inherited = current_attributes.copy()
            inheritable = {k: v for k, v in element.attrib.items() if k not in NON_INHERITABLE}
            text_inherited.update(inheritable)
            
            for attr in basic_shape_attrs:
                if tag == 'tspan' and attr == 'y' and 'dy' in element.attrib and 'y' not in element.attrib:
                    continue 

                if attr in attributes:
                    val = attributes[attr]
                    if attr in ['x', 'y', 'width', 'height', 'font-size', 'stroke-width', 'opacity', 'dy', 'dx']:
                         val = str(_parse_number_preserve(val))
                    elif attr == 'transform': val = _parse_transform(val)
                    val = str(val)
                    struct_ret += "  " * (level + 1) + f"{attr}={val}\n"

            if element.text and element.text.strip():
                content = element.text.strip()
                struct_ret += "  " * (level + 1) + f"text-content={content}\n"

            for child in element:
                recursive_parse(child, level + 1, text_inherited)
                if child.tail and child.tail.strip():
                     content = child.tail.strip()
                     struct_ret += "  " * (level + 1) + f"text-content={content}\n"
            
            struct_ret += "  " * level + f"</{tag}>\n"; return

        if tag == 'use':
            attr_str = "".join([f' {k}="{v}"' for k, v in attributes.items()])
            struct_ret += "  " * level + f"<{tag}{attr_str} />\n"; return

        if tag not in ignore_tags: struct_ret += "  " * level + f"<{tag}>\n"

        if tag == "path" and "d" in attributes:
            struct_ret += "  " * (level + 1) + f"d="
            struct_ret += _parse_path_d(attributes['d'], level + 2, float_coords=True)
            struct_ret += f"{_gather_path_attr(attributes)}\n"

        elif tag == "svg":
             attr_str = ""
             allowed = ['width', 'height', 'viewBox', 'preserveAspectRatio', 'xmlns', 'version', 'x', 'y']
             for attr, value in attributes.items():
                 if attr in allowed:
                     val = value
                     if attr in ['width', 'height', 'x', 'y']: val = _parse_number(val)
                     elif attr == 'viewBox': val = _parse_point_attr(val, float_coords=True)
                     val = str(val)
                     attr_str += f"{attr}={val} "
             if attr_str: struct_ret += "  " * (level + 1) + attr_str + "\n"
             for child in element: recursive_parse(child, level + 1, current_attributes)

        elif tag in shape_tags and (not path_only):
            attr_str = ""
            for attr in basic_shape_attrs:
                if attr in attributes and attr not in ignore_attrs:
                    value = attributes[attr]
                    if attr in ['points', 'viewBox']: value = _parse_point_attr(value, float_coords=True)
                    elif attr in ['x', 'y', 'cx', 'cy', 'r', 'width', 'height', 'x1', 'y1', 'x2', 'y2', 'stroke-width', 'opacity']:
                        value = _parse_number(value)
                    elif attr == 'transform': value = _parse_transform(value)
                    value = str(value)
                    attr_str += f"{attr}={value} "
            struct_ret += "  " * (level + 1) + attr_str + "\n"

        elif tag in gradient_tags:
            attr_str = "".join([f"{k}={v} " for k, v in attributes.items()])
            if attr_str: struct_ret += "  " * (level + 1) + attr_str + "\n"

        if tag not in ignore_tags and tag != "svg": struct_ret += "  " * level + f"</{tag}>\n"

    recursive_parse(tree)
    flatten_struct_ret = struct_ret.replace("\n", "")
    svg_desc_ret = _to_svg_description(flatten_struct_ret)
    return struct_ret, _clean_svg_desc_output(svg_desc_ret)

# Helpers
def _parse_style_string(style_str):
    styles = {}
    if not style_str: return styles
    for item in style_str.split(';'):
        if ':' in item:
            k, v = item.split(':', 1)
            styles[k.strip()] = v.strip()
    return styles

def _parse_number_preserve(num_str):
    s = str(num_str).lower()
    if 'em' in s or '%' in s: return str(num_str)
    return _parse_number(num_str)

def _parse_number(num_str: str, round_num: int = 2) -> Union[int, float, str]:
    try:
        clean_str = str(num_str).lower().replace('px', '').replace('%', '')
        if 'em' in str(num_str): return num_str
        num = float(clean_str)
        if num.is_integer():
            return str(int(num))
        return f"{num:.1f}"
    except: return num_str

def _parse_transform(transform_str: str) -> str:
    coord_pattern = re.compile(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?')
    def replace_num(match): return str(_parse_number(match.group(0)))
    return coord_pattern.sub(replace_num, transform_str)

def _clean_svg_desc_output(svg_string: str):
    svg_string = re.sub(r'\s*\[\s*', '[', svg_string)
    svg_string = re.sub(r'\s*\]\s*', ']', svg_string)
    svg_string = re.sub(r'(\d)\s+', r'\1 ', svg_string)
    svg_string = re.sub(r'\]\s*\[', '][', svg_string)
    return svg_string

def _to_svg_description(input_string: str):
    combined_mapper = {**PathMapper, **PathCMDMapper, **ShapeMapper, **ContainerMapper, **GradientsMapper, **AttribMapper}
    keys = sorted(combined_mapper.keys(), key=len, reverse=True)
    pattern = re.compile('|'.join(map(re.escape, keys)))
    def replacement(match): return combined_mapper[match.group(0)]
    return re.sub(r'\]\s+\[', '][', pattern.sub(replacement, input_string))

def _gather_path_attr(attrs):
    ret = ""
    for k in ['id', 'fill', 'stroke', 'stroke-width', 'opacity', 'transform', 'class', 'style', 'clip-path']:
        if k in attrs: ret += f" {k}={str(attrs[k])}"
    return ret

def _parse_point_attr(d_string, float_coords=False):
    coord_pattern = re.compile(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?')
    vals = coord_pattern.findall(d_string)
    if float_coords: vals = [_parse_number(x) for x in vals]
    return " ".join(map(str, vals))

def _parse_path_d(d_string, level=0, float_coords=False):
    path_command_pattern = re.compile(r'([a-zA-Z])\s*([-0-9.,\s]*)')
    coord_pattern = re.compile(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?')
    matches = path_command_pattern.findall(d_string)
    ret = ""
    for command, points in matches:
        point_values = coord_pattern.findall(points)
        if float_coords: point_values = [_parse_number(p) for p in point_values]
        ret += f"[{command}]{' '.join(map(str, point_values))} "
    return ret

def _extract_elements_and_attributes(svg_description): return []

# ==============================================================================
# DECODER (V21 - Hierarchy Separation & Auto-Close)
# ==============================================================================
def parse_svg_description(svg_description):
    root_params = {'width': '512', 'height': '512', 'viewBox': '0 0 512 512', 'xmlns': 'http://www.w3.org/2000/svg', 'overflow': 'visible'}
    svg_elements = []
    stack = []
    
    LEAF_TAGS = {'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'path', 'stop', 'use'}
    CONTAINER_TAGS = {'g', 'defs', 'clipPath', 'linearGradient', 'radialGradient', 'text', 'tspan', 'svg'}
    
    TOKEN_MAP = {}
    for k, v in ShapeIdentifier.items(): TOKEN_MAP[k] = v
    for k, v in GradientIdentifier.items(): TOKEN_MAP[k] = v
    TOKEN_MAP['linearGradient'] = 'linearGradient'; TOKEN_MAP['radialGradient'] = 'radialGradient'
    TOKEN_MAP['START_OF_GROUP'] = 'g'; TOKEN_MAP['clipPath'] = 'clipPath'
    TOKEN_MAP['defs'] = 'defs'; TOKEN_MAP['path'] = 'path'; TOKEN_MAP['START_OF_SVG'] = 'svg'

    CLOSING_MAP = {
        'END_OF_GROUP': 'g', '/clipPath': 'clipPath', '/defs': 'defs',
        '/linearGradient': 'linearGradient', '/radialGradient': 'radialGradient',
        '/stop': 'stop', '/text': 'text', '/tspan': 'tspan', '/path': 'path', 'END_OF_SVG': 'svg'
    }
    for tag in ShapeIdentifier.keys(): CLOSING_MAP[f'/{tag}'] = ShapeIdentifier[tag]

    def to_attr(attributes):
        attr_list = []
        RESTORE_SPACE_ATTRS = {
            'fill', 'stroke', 'style', 'd', 'points', 'viewBox', 'transform', 'gradientTransform',
            'stroke-dasharray', 'font-family', 'class', 'clip-path'
        }
        
        if 'style' in attributes:
            style_parts = [p.strip() for p in attributes['style'].replace('_', ' ').split(';') if p.strip()]
            style_map = {}
            for p in style_parts:
                if ':' in p:
                    k, v = p.split(':', 1)
                    style_map[k.strip()] = v.strip()
            
            SYNC_KEYS = ['text-anchor', 'dominant-baseline', 'font-family', 'font-size', 'visibility', 'display', 'fill', 'stroke', 'opacity']
            updated = False
            for k in SYNC_KEYS:
                if k in attributes and k not in style_map:
                    val_str = str(attributes[k]).replace('_', ' ')
                    style_map[k] = val_str
                    updated = True
            if updated:
                new_style = "; ".join(f"{k}: {v}" for k, v in style_map.items())
                attributes['style'] = new_style

        for k, v in attributes.items():
            if v is not None and not k.startswith('__'):
                val_str = str(v)
                if k in RESTORE_SPACE_ATTRS:
                    val_str = val_str.replace('_', ' ')
                attr_list.append(f'{k}={quoteattr(val_str)}')
        return " ".join(attr_list)

    def flush_tag(item, self_closing=False):
        if item['flushed']: return
        tag = item['tag']; attrs = item['attrs']
        
        if tag == 'path' and item['path_data']: attrs['d'] = " ".join(item['path_data']).replace("_", " ")
        content = attrs.pop('__text_content__', None)
        raw_content = item.get('raw_content')
        if raw_content: raw_content = raw_content.replace("_", " ")

        in_structural = any(p['tag'] in ['clipPath', 'defs'] for p in stack)

        # [FIX] V20 Logic for <g> coordinates
        if tag == 'g' and ('x' in attrs or 'y' in attrs):
            gx = attrs.pop('x', '0')
            gy = attrs.pop('y', '0')
            new_trans = f"translate({gx}, {gy})"
            if 'transform' in attrs: attrs['transform'] = f"{attrs['transform']} {new_trans}"
            else: attrs['transform'] = new_trans

        # [FIX] V19 Logic for nested SVG overflow
        if tag == 'svg' and len(stack) > 1: attrs['overflow'] = 'visible'

        if not in_structural:
            has_fill = 'fill' in attrs or 'style' in attrs
            has_stroke = 'stroke' in attrs or 'style' in attrs
            
            if tag in ['rect', 'circle', 'ellipse', 'polygon']:
                if not has_fill: attrs['fill'] = "none"
            elif tag in ['path', 'line', 'polyline']:
                if not has_stroke and not has_fill:
                     attrs['stroke'] = "#000000"; attrs['fill'] = "none"
                if ('stroke' in attrs or has_stroke) and 'stroke-width' not in attrs:
                     if attrs.get('fill') == 'none' or not has_fill: attrs['stroke-width'] = "1.0"
            elif tag == 'text':
                if not has_fill: attrs['fill'] = "#000000"

        attr_str = to_attr(attrs)
        if self_closing and not content and not raw_content: svg_elements.append(f'<{tag} {attr_str} />')
        else:
            svg_elements.append(f'<{tag} {attr_str}>')
            if content: svg_elements.append(escape(str(content).replace("_", " ")))
            if raw_content: svg_elements.append(escape(raw_content))
        item['flushed'] = True

    tokens = svg_description.split('[<|')
    for i, token in enumerate(tokens):
        if '|>]' not in token: continue
        tag_raw, content = token.split('|>]', 1)
        tag = tag_raw.strip(); content = content.strip()
        
        if content: content = content.replace("_", " ")
        if tag.endswith('='): tag = tag[:-1]
            
        if tag in CLOSING_MAP:
            target_tag = CLOSING_MAP[tag]
            if target_tag == 'svg':
                 if len(stack) == 1 and stack[-1]['tag'] == 'svg':
                     item = stack.pop(); 
                     if not item['flushed']: flush_tag(item, self_closing=False)
                     svg_elements.append('</svg>'); break
                 if stack and stack[-1]['tag'] == 'svg':
                     item = stack.pop(); 
                     if not item['flushed']: flush_tag(item, self_closing=False)
                     svg_elements.append('</svg>'); break

            if stack and stack[-1]['tag'] == target_tag:
                item = stack.pop()
                is_empty = not item['attrs'].get('__text_content__')
                if not item['flushed']:
                    flush_tag(item, self_closing=(item['tag'] in LEAF_TAGS and is_empty))
                if not (item['tag'] in LEAF_TAGS and is_empty): svg_elements.append(f'</{target_tag}>')
            else:
                while stack and stack[-1]['tag'] != target_tag:
                    mismatch = stack.pop()
                    is_empty = not mismatch['attrs'].get('__text_content__')
                    if not mismatch['flushed']:
                        flush_tag(mismatch, self_closing=(mismatch['tag'] in LEAF_TAGS and is_empty))
                    if not (mismatch['tag'] in LEAF_TAGS and is_empty): svg_elements.append(f'</{mismatch["tag"]}>')
                if stack and stack[-1]['tag'] == target_tag:
                    item = stack.pop(); 
                    is_empty = not item['attrs'].get('__text_content__')
                    if not item['flushed']:
                        flush_tag(item, self_closing=(item['tag'] in LEAF_TAGS and is_empty))
                    if not (item['tag'] in LEAF_TAGS and is_empty): svg_elements.append(f'</{target_tag}>')
            continue

        new_tag = TOKEN_MAP.get(tag)
        if new_tag:
            # [FIX] V21: Auto-close nested SVG if a new root-level semantic group appears
            # e.g. if we are in <svg class="legend"> and we encounter <g class="annotation">
            # we should probably close the legend svg.
            # Heuristic: If we are deep in stack and see a 'g' with 'annotation' class, check if we are in 'legend'
            if tag == 'g' and 'annotation' in content:
                # Iterate stack backwards
                for j in range(len(stack)-1, -1, -1):
                    p_tag = stack[j]['tag']
                    p_attrs = stack[j]['attrs']
                    # If we are inside a legend SVG
                    if p_tag == 'svg' and 'legend' in str(p_attrs.get('class', '')):
                        # Close everything up to this svg
                        while len(stack) > j:
                            item = stack.pop()
                            if not item['flushed']: flush_tag(item, self_closing=False)
                            svg_elements.append(f'</{item["tag"]}>')
                        break

            if stack:
                parent = stack[-1]
                if parent['tag'] in LEAF_TAGS:
                    stack.pop(); 
                    if not parent['flushed']: flush_tag(parent, self_closing=True)
                elif not parent['flushed']: flush_tag(parent, self_closing=False)
            stack.append({'tag': new_tag, 'attrs': {}, 'flushed': False, 'path_data': [], 'raw_content': content})
            continue

        if stack:
            current = stack[-1]
            if tag == 'text-content':
                if current['flushed']: svg_elements.append(escape(str(content)))
                else: current['attrs']['__text_content__'] = content
            elif tag in PathCMDIdentifier: current['path_data'].append(f"{PathCMDIdentifier[tag]} {content}")
            else:
                current['attrs'][tag] = content
                if len(stack) == 1 and current['tag'] == 'svg':
                    if tag in ['width', 'height', 'viewBox', 'xmlns']: root_params[tag] = content

    while stack:
        item = stack.pop()
        if not item['flushed']: flush_tag(item, self_closing=(item['tag'] in LEAF_TAGS))
        if item['tag'] in CONTAINER_TAGS: svg_elements.append(f'</{item["tag"]}>')

    full_svg = "".join(svg_elements)
    if not full_svg.strip().startswith("<svg"):
        header = f'<svg height="{root_params["height"]}" width="{root_params["width"]}" viewBox="{root_params["viewBox"]}" xmlns="{root_params["xmlns"]}" overflow="visible">'
        full_svg = header + full_svg + "</svg>"
    
    if "<svg" in full_svg and "xmlns=" not in full_svg.split(">")[0]:
        full_svg = full_svg.replace("<svg", f'<svg xmlns="{root_params["xmlns"]}"', 1)
        
    if "<svg" in full_svg and "viewBox" not in full_svg.split(">")[0]:
        full_svg = full_svg.replace("<svg", f'<svg viewBox="{root_params["viewBox"]}" width="{root_params["width"]}" height="{root_params["height"]}" overflow="visible"', 1)

    return full_svg

def syntactic2svg(svg_description: str, print_info: bool = False) -> str:
    return parse_svg_description(svg_description)
