#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize GraphML knowledge graphs.

Usage examples:
    python visualize_graphml.py path/to/graph_chunk_entity_relation.graphml
    python visualize_graphml.py path/to/graph.graphml --format html --output graph.html
    python visualize_graphml.py path/to/graph.graphml --format png --output graph.png --layout kamada_kawai

The script supports two output formats:
    - html: interactive visualization using PyVis (requires 'pyvis').
    - png: static image using matplotlib (requires 'matplotlib').
"""

import argparse
import sys
from html import escape
from pathlib import Path
from typing import Dict, Any, Optional

import json
import networkx as nx


PALETTE = [
    "#5B8FF9",
    "#61DDAA",
    "#65789B",
    "#F6BD16",
    "#7262FD",
    "#78D3F8",
    "#9661BC",
    "#F6903D",
    "#008685",
    "#F08BB4",
]

LAYOUTS_2D = {
    "spring": nx.spring_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "circular": nx.circular_layout,
    "shell": nx.shell_layout,
    "spectral": nx.spectral_layout,
    "random": nx.random_layout,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a GraphML knowledge graph")
    parser.add_argument("graphml", type=Path, help="Path to the GraphML file")
    parser.add_argument(
        "--format",
        choices=["html", "png"],
        default="html",
        help="Output format (default: html)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path. Defaults to graph.html or graph.png next to the input file.",
    )
    parser.add_argument(
        "--layout",
        choices=sorted(LAYOUTS_2D.keys()),
        default="spring",
        help="Layout algorithm for PNG output (default: spring)",
    )
    parser.add_argument(
        "--title-field",
        default="description",
        help="Node attribute to use as title/hover tooltip (default: description)",
    )
    parser.add_argument(
        "--label-field",
        default="entity_id",
        help="Node attribute to use as label (default: entity_id)",
    )
    parser.add_argument(
        "--edge-label-field",
        default="description",
        help="Edge attribute to use as hover title (default: description)",
    )
    return parser.parse_args()


def ensure_graph_exists(path: Path) -> Path:
    if not path.exists():
        print(f"[错误] GraphML 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix.lower() != ".graphml":
        print(f"[警告] 指定文件不是 .graphml: {path}")
    return path


def load_graph(graph_path: Path) -> nx.Graph:
    print(f"[信息] 读取图谱: {graph_path}")
    graph = nx.read_graphml(graph_path)
    print(
        f"[信息] 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}"
    )
    return graph


def build_node_title(data: Dict[str, Any]) -> str:
    parts = []
    for key, value in data.items():
        if value is None:
            continue
        parts.append(f"{key}: {value}")
    return "<br>".join(parts) if parts else ""


def visualize_html(graph: nx.Graph, output_path: Path, title_field: str, label_field: str, edge_label_field: str) -> None:
    try:
        from pyvis.network import Network
    except ImportError:  # pragma: no cover - optional dependency
        print("[错误] 需要安装 pyvis (pip install pyvis) 才能导出 HTML", file=sys.stderr)
        sys.exit(1)

    net = Network(
        height="750px",
        width="100%",
        notebook=False,
        directed=graph.is_directed(),
        bgcolor="#f5f7fb",
    )
    net.from_nx(graph)

    color_map: Dict[Optional[str], str] = {}

    def get_color(entity_type: Optional[str]) -> str:
        if entity_type not in color_map:
            color_map[entity_type] = PALETTE[len(color_map) % len(PALETTE)]
        return color_map[entity_type]

    # 更新节点标签和提示信息
    for node_id, data in graph.nodes(data=True):
        node = net.get_node(node_id)
        if not node:
            continue
        title = data.get(title_field) or build_node_title(data)
        label = data.get(label_field) or node_id
        node["title"] = title
        node["label"] = str(label)
        node["color"] = get_color(data.get("entity_type"))
        node["shape"] = "dot"
        node["size"] = 18

    # 更新边的提示
    for edge in net.edges:
        src, dst = edge["from"], edge["to"]
        edge_data = graph.get_edge_data(src, dst, default={})
        if isinstance(edge_data, dict):
            description = edge_data.get(edge_label_field) or build_node_title(edge_data)
            if description:
                edge["title"] = description

    print(f"[信息] 导出 HTML: {output_path}")
    options = {
        "nodes": {
            "font": {"color": "#1f2d3d", "size": 14, "face": "Helvetica"},
            "borderWidth": 1,
        },
        "edges": {
            "color": {"color": "#94a3b8", "highlight": "#2563eb"},
            "width": 1.5,
            "smooth": {"type": "dynamic"},
        },
        "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.015,
                "springConstant": 0.12,
                "springLength": 130,
            },
            "stabilization": {"iterations": 120},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": False,
        },
    }
    net.set_options(json.dumps(options))

    net.write_html(output_path.as_posix(), notebook=False, open_browser=False)
    append_summary_html(output_path, graph, title_field, label_field, edge_label_field)


def visualize_png(graph: nx.Graph, output_path: Path, layout_name: str, title_field: str, label_field: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        print("[错误] 需要安装 matplotlib (pip install matplotlib) 才能导出 PNG", file=sys.stderr)
        sys.exit(1)

    layout_func = LAYOUTS_2D.get(layout_name, nx.spring_layout)
    pos = layout_func(graph)

    node_labels = {
        node: data.get(label_field, node) for node, data in graph.nodes(data=True)
    }
    node_colors = [hash(data.get("entity_type", "default")) % 10 for _, data in graph.nodes(data=True)]

    plt.figure(figsize=(12, 8))
    nx.draw_networkx(
        graph,
        pos,
        labels=node_labels,
        node_color=node_colors,
        cmap=plt.cm.Set3,
        with_labels=True,
        font_size=8,
        node_size=800,
        edge_color="#999999",
    )

    plt.axis("off")
    plt.tight_layout()
    print(f"[信息] 导出 PNG: {output_path}")
    plt.savefig(output_path, dpi=150)
    plt.close()


def format_node_li(node_id: str, data: Dict[str, Any], label_field: str, title_field: str) -> str:
    label = data.get(label_field) or node_id
    entity_type = data.get("entity_type")
    file_path = data.get("file_path")
    description = data.get(title_field)

    pieces = [f"<span class=\"entity-name\">{escape(str(label))}</span>"]
    if entity_type:
        pieces.append(f" <span class=\"entity-type\">{escape(str(entity_type))}</span>")
    if file_path:
        pieces.append(f"<div class=\"meta\">{escape(str(file_path))}</div>")
    if description:
        pieces.append(f"<div class=\"description\">{escape(str(description))}</div>")
    return "".join(pieces)


def format_edge_li(src: str, dst: str, data: Dict[str, Any], edge_label_field: str) -> str:
    desc = data.get(edge_label_field) or data.get("description")
    file_path = data.get("file_path")
    weight = data.get("weight")

    parts = [f"<strong>{escape(str(src))}</strong> ⇄ <strong>{escape(str(dst))}</strong>"]
    if desc:
        parts.append(f"<div class=\"description\">{escape(str(desc))}</div>")
    if file_path:
        parts.append(f"<div class=\"meta\">{escape(str(file_path))}</div>")
    if weight is not None:
        parts.append(f"<div class=\"meta\">weight: {escape(str(weight))}</div>")
    return "".join(parts)


def append_summary_html(output_path: Path, graph: nx.Graph, title_field: str, label_field: str, edge_label_field: str) -> None:
    try:
        html_text = output_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        print(f"[警告] 无法读取生成的 HTML: {exc}")
        return

    nodes = sorted(graph.nodes(data=True), key=lambda item: str(item[1].get(label_field, item[0])))
    edges = sorted(graph.edges(data=True), key=lambda item: (str(item[0]), str(item[1])))

    node_items = "".join(
        f"<li>{format_node_li(node_id, data, label_field, title_field)}</li>" for node_id, data in nodes
    )
    edge_items = "".join(
        f"<li>{format_edge_li(src, dst, data, edge_label_field)}</li>" for src, dst, data in edges
    )

    summary_block = f"""
    <style>
      body {{ background-color: #f5f7fb; color: #1f2d3d; }}
      #graph-summary {{ padding: 28px 8% 40px; font-family: 'Helvetica Neue', Arial, sans-serif; }}
      #graph-summary h2 {{ margin: 0 0 16px; font-size: 22px; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
      #graph-summary .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 28px; }}
      #graph-summary ol {{ margin: 0; padding-left: 22px; background: #ffffff; border-radius: 12px; box-shadow: 0 6px 24px rgba(15, 23, 42, 0.08); padding: 24px; max-height: 520px; overflow-y: auto; }}
      #graph-summary li {{ margin-bottom: 14px; line-height: 1.5; }}
      #graph-summary .entity-name {{ font-weight: 600; font-size: 16px; }}
      #graph-summary .entity-type {{ font-size: 13px; color: #64748b; margin-left: 6px; text-transform: uppercase; letter-spacing: .04em; }}
      #graph-summary .meta {{ font-size: 12px; color: #94a3b8; margin-top: 4px; word-break: break-all; }}
      #graph-summary .description {{ font-size: 13px; color: #334155; margin-top: 6px; }}
    </style>
    <div id=\"graph-summary\">
      <div class=\"summary-grid\">
        <div>
          <h2>实体列表 ({len(nodes)})</h2>
          <ol>{node_items}</ol>
        </div>
        <div>
          <h2>关系列表 ({len(edges)})</h2>
          <ol>{edge_items}</ol>
        </div>
      </div>
    </div>
    </body>
    """

    if "</body>" in html_text:
        html_text = html_text.replace("</body>", summary_block, 1)
    else:
        html_text = html_text + summary_block + "</html>"

    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    graph_path = ensure_graph_exists(args.graphml)
    output_path = args.output

    if output_path is None:
        suffix = ".html" if args.format == "html" else ".png"
        output_path = graph_path.with_suffix(suffix)

    graph = load_graph(graph_path)

    if args.format == "html":
        visualize_html(graph, output_path, args.title_field, args.label_field, args.edge_label_field)
    else:
        visualize_png(graph, output_path, args.layout, args.title_field, args.label_field)

    print("[完成] 可视化输出:", output_path)


if __name__ == "__main__":
    main()
