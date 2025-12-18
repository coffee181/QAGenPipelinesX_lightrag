#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GraphML Knowledge Graph Visualizer (Optimized)

这是一个用于将 GraphML 文件转换为高颜值、交互式 HTML 或静态 PNG 的可视化工具。
主要特点：
1. 现代化的 UI 设计（侧边栏、磨砂玻璃效果）。
2. HTML 模式支持双向交互（点击列表聚焦节点，搜索过滤）。
3. PNG 模式支持基于节点重要性的动态大小和曲线边。

Usage:
    python visualize.py data.graphml --format html
    python visualize.py data.graphml --format png
"""

import argparse
import sys
import json
import random
from html import escape
from pathlib import Path
from typing import Dict, Any, Optional, List

import networkx as nx

# --- 配色方案 (Morandi/Modern Palette) ---
PALETTE = [
    "#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
    "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#14b8a6"
]

# --- 布局算法映射 ---
LAYOUTS_2D = {
    "spring": nx.spring_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "circular": nx.circular_layout,
    "shell": nx.shell_layout,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成精美的知识图谱可视化")
    parser.add_argument("graphml", type=Path, help="GraphML 文件路径")
    parser.add_argument("--format", choices=["html", "png"], default="html", help="输出格式 (默认: html)")
    parser.add_argument("--output", type=Path, help="输出文件路径")
    parser.add_argument("--layout", choices=LAYOUTS_2D.keys(), default="spring", help="PNG 布局算法")
    parser.add_argument("--title-field", default="description", help="用作悬浮提示的属性")
    parser.add_argument("--label-field", default="entity_id", help="用作标签的属性")
    return parser.parse_args()

def ensure_graph_exists(path: Path) -> Path:
    if not path.exists():
        print(f"❌ 错误: 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)
    return path

def load_graph(graph_path: Path) -> nx.Graph:
    print(f"📂 读取图谱: {graph_path.name} ...", end="", flush=True)
    graph = nx.read_graphml(graph_path)
    print(f" 完成 (节点: {graph.number_of_nodes()}, 边: {graph.number_of_edges()})")
    return graph

# ==============================================================================
# HTML Visualization Logic (PyVis + Custom JS/CSS)
# ==============================================================================

def visualize_html(graph: nx.Graph, output_path: Path, title_field: str, label_field: str) -> None:
    try:
        from pyvis.network import Network
    except ImportError:
        print("❌ 错误: 请安装 pyvis (pip install pyvis)", file=sys.stderr)
        sys.exit(1)

    # 1. 初始化 PyVis 网络
    net = Network(height="100vh", width="100%", bgcolor="#f8fafc", font_color="#334155", notebook=False)
    net.from_nx(graph)

    # 2. 预处理数据以增强视觉效果
    # 计算度中心性以调整节点大小
    degrees = dict(graph.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    # 颜色映射缓存
    type_color_map = {}

    def get_color(e_type: str) -> str:
        if not e_type: return "#94a3b8" # Default gray
        if e_type not in type_color_map:
            type_color_map[e_type] = PALETTE[len(type_color_map) % len(PALETTE)]
        return type_color_map[e_type]

    for node in net.nodes:
        nid = node["id"]
        nx_data = graph.nodes[nid]
        
        # 获取标签和属性
        lbl = str(nx_data.get(label_field, nid))
        desc = str(nx_data.get(title_field, ""))
        e_type = str(nx_data.get("entity_type", "Unknown"))
        
        # 视觉样式
        node["label"] = lbl
        node["title"] = f"<b>{lbl}</b><br><i>{e_type}</i><br><br>{desc}"
        node["color"] = get_color(e_type)
        node["group"] = e_type  # 用于 PyVis 图例
        
        # 动态大小 (基础大小 15 + 基于度的增量)
        deg = degrees.get(nid, 0)
        node["size"] = 15 + (deg / max_degree) * 25
        node["borderWidth"] = 2
        node["borderWidthSelected"] = 4

    # 3. 配置物理引擎 (力导向图参数)
    options = {
        "nodes": {
            "font": {"face": "Inter, system-ui", "size": 14, "strokeWidth": 0, "color": "#1e293b"},
            "shadow": {"enabled": True, "color": "rgba(0,0,0,0.1)", "size": 10, "x": 5, "y": 5}
        },
        "edges": {
            "color": {"color": "#cbd5e1", "highlight": "#6366f1"},
            "width": 1,
            "smooth": {"type": "continuous", "roundness": 0.5},
            "selectionWidth": 2
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08,
                "damping": 0.4
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"enabled": True, "iterations": 200}
        },
        "interaction": {
            "hover": True, 
            "navigationButtons": True, 
            "keyboard": False
        }
    }
    net.set_options(json.dumps(options))

    # 4. 生成临时 HTML
    # PyVis write_html 会生成包含 graph 数据的 HTML
    net.write_html(str(output_path), notebook=False)

    # 5. 注入自定义 UI (侧边栏 + JS 交互)
    inject_custom_interface(output_path, graph, label_field, title_field)
    print(f"✨ HTML 可视化已生成: {output_path}")


def inject_custom_interface(html_path: Path, graph: nx.Graph, label_field: str, title_field: str):
    """
    读取 PyVis 生成的 HTML，强力注入现代化的侧边栏 UI 和交互 JS 代码。
    """
    
    # 准备数据列表
    nodes_data = []
    for nid, data in graph.nodes(data=True):
        nodes_data.append({
            "id": nid,
            "label": str(data.get(label_field, nid)),
            "type": str(data.get("entity_type", "N/A")),
            "desc": str(data.get(title_field, ""))
        })
    # 按标签排序
    nodes_data.sort(key=lambda x: x["label"])

    edges_data = []
    for u, v, data in graph.edges(data=True):
        edges_data.append({
            "source": u,
            "target": v,
            "desc": str(data.get("description", ""))
        })

    # 将数据转为 JSON 嵌入 HTML，供前端 JS 使用
    json_nodes = json.dumps(nodes_data)
    json_edges = json.dumps(edges_data)

    # --- CSS 样式 (Tailwind-like + Glassmorphism) ---
    css_styles = """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        body, html { margin: 0; padding: 0; font-family: 'Inter', sans-serif; overflow: hidden; }
        
        /* 侧边栏容器 */
        #ui-container {
            position: absolute; top: 20px; right: 20px; bottom: 20px; width: 380px;
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.5);
            display: flex; flex-direction: column;
            z-index: 999;
            transition: transform 0.3s ease;
        }
        
        /* 收起/展开按钮 */
        #toggle-btn {
            position: absolute; top: 15px; left: -40px; width: 32px; height: 32px;
            background: white; border-radius: 8px; border: none; cursor: pointer;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; color: #64748b;
        }

        /* 头部 */
        .ui-header { padding: 20px; border-bottom: 1px solid rgba(0,0,0,0.05); }
        .ui-title { margin: 0; font-size: 18px; font-weight: 600; color: #0f172a; }
        .ui-subtitle { margin: 4px 0 0; font-size: 13px; color: #64748b; }

        /* 搜索框 */
        .search-box {
            margin: 15px 20px 10px;
            position: relative;
        }
        .search-input {
            width: 100%; padding: 10px 15px; border-radius: 8px;
            border: 1px solid #e2e8f0; background: rgba(255,255,255,0.6);
            outline: none; font-size: 14px; box-sizing: border-box;
            transition: border-color 0.2s;
        }
        .search-input:focus { border-color: #6366f1; background: white; }

        /* Tabs */
        .tabs { display: flex; padding: 0 20px; gap: 15px; margin-bottom: 10px; border-bottom: 1px solid #f1f5f9; }
        .tab { 
            padding: 10px 0; font-size: 14px; font-weight: 500; color: #94a3b8; 
            cursor: pointer; position: relative; 
        }
        .tab.active { color: #6366f1; }
        .tab.active::after {
            content: ''; position: absolute; bottom: -1px; left: 0; width: 100%; height: 2px; background: #6366f1;
        }

        /* 列表区域 */
        .list-viewport { flex: 1; overflow-y: auto; padding: 10px 20px; scroll-behavior: smooth; }
        
        /* 列表项卡片 */
        .list-item {
            background: rgba(255,255,255,0.5); border: 1px solid rgba(0,0,0,0.02);
            border-radius: 8px; padding: 12px; margin-bottom: 10px;
            cursor: pointer; transition: all 0.2s;
        }
        .list-item:hover { background: white; transform: translateY(-1px); box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
        .list-item.active { border-left: 3px solid #6366f1; background: white; }
        
        .item-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
        .item-name { font-weight: 600; font-size: 14px; color: #334155; }
        .item-tag { 
            font-size: 11px; padding: 2px 6px; border-radius: 4px; 
            background: #e0e7ff; color: #4338ca; text-transform: uppercase;
        }
        .item-desc { font-size: 12px; color: #64748b; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

        /* 滚动条美化 */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
    </style>
    """

    # --- HTML 结构 ---
    html_structure = f"""
    <div id="ui-container">
        <button id="toggle-btn" onclick="toggleSidebar()">⇄</button>
        <div class="ui-header">
            <h1 class="ui-title">知识图谱浏览器</h1>
            <p class="ui-subtitle">包含 {len(nodes_data)} 个实体，{len(edges_data)} 条关系</p>
        </div>
        
        <div class="search-box">
            <input type="text" id="search-input" class="search-input" placeholder="搜索实体..." onkeyup="filterList()">
        </div>

        <div class="tabs">
            <div class="tab active" onclick="switchTab('nodes')">实体列表</div>
            <div class="tab" onclick="switchTab('edges')">关系详情</div>
        </div>

        <div id="nodes-list" class="list-viewport">
            <!-- JS Populated -->
        </div>
        <div id="edges-list" class="list-viewport" style="display:none;">
            <!-- JS Populated -->
        </div>
    </div>

    <script>
        const nodesData = {json_nodes};
        const edgesData = {json_edges};
        let networkInstance = null; // Will hold the pyvis network

        // 等待 PyVis 初始化
        window.addEventListener("load", function() {{
            // PyVis creates a global 'network' variable in the script it generates
            if (typeof network !== 'undefined') {{
                networkInstance = network;
                
                // 绑定点击事件：图 -> 列表
                networkInstance.on("click", function(params) {{
                    if (params.nodes.length > 0) {{
                        const nodeId = params.nodes[0];
                        highlightListItem(nodeId);
                    }}
                }});
            }}
            renderNodes(nodesData);
            renderEdges(edgesData);
        }});

        function renderNodes(data) {{
            const container = document.getElementById('nodes-list');
            container.innerHTML = data.map(n => `
                <div class="list-item" id="item-${{n.id}}" onclick="focusNode('${{n.id}}')">
                    <div class="item-head">
                        <span class="item-name">${{n.label}}</span>
                        <span class="item-tag">${{n.type}}</span>
                    </div>
                    <div class="item-desc">${{n.desc || '暂无描述'}}</div>
                </div>
            `).join('');
        }}

        function renderEdges(data) {{
            const container = document.getElementById('edges-list');
            container.innerHTML = data.map((e, idx) => `
                <div class="list-item">
                    <div class="item-head">
                        <span class="item-name">${{e.source}} ➝ ${{e.target}}</span>
                    </div>
                    <div class="item-desc">${{e.desc || '...'}}</div>
                </div>
            `).join('');
        }}

        // 交互：列表 -> 图
        function focusNode(nodeId) {{
            if (!networkInstance) return;
            
            // 高亮列表项
            document.querySelectorAll('.list-item').forEach(el => el.classList.remove('active'));
            const el = document.getElementById('item-' + nodeId);
            if (el) {{
                el.classList.add('active');
                el.scrollIntoView({{behavior: "smooth", block: "center"}});
            }}

            // 聚焦图谱
            networkInstance.focus(nodeId, {{
                scale: 1.2,
                animation: {{ duration: 1000, easingFunction: "easeInOutQuad" }}
            }});
            networkInstance.selectNodes([nodeId]);
        }}

        // 交互：图 -> 列表 (反向高亮)
        function highlightListItem(nodeId) {{
            switchTab('nodes');
            const el = document.getElementById('item-' + nodeId);
            if (el) {{
                document.querySelectorAll('.list-item').forEach(e => e.classList.remove('active'));
                el.classList.add('active');
                el.scrollIntoView({{behavior: "smooth", block: "center"}});
            }}
        }}

        // 搜索过滤
        function filterList() {{
            const query = document.getElementById('search-input').value.toLowerCase();
            const filtered = nodesData.filter(n => 
                n.label.toLowerCase().includes(query) || 
                n.desc.toLowerCase().includes(query)
            );
            renderNodes(filtered);
        }}

        // Tab 切换
        function switchTab(tab) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            
            document.getElementById('nodes-list').style.display = tab === 'nodes' ? 'block' : 'none';
            document.getElementById('edges-list').style.display = tab === 'edges' ? 'block' : 'none';
        }}
        
        // 侧边栏开关
        function toggleSidebar() {{
            const ui = document.getElementById('ui-container');
            if (ui.style.transform === 'translateX(110%)') {{
                ui.style.transform = 'translateX(0)';
            }} else {{
                ui.style.transform = 'translateX(110%)';
            }}
        }}
    </script>
    """

    # 读取文件，替换 Body 结束标签
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 插入 CSS 到 Head，插入 UI 到 Body
    content = content.replace("</head>", f"{css_styles}</head>")
    content = content.replace("</body>", f"{html_structure}</body>")
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)

# ==============================================================================
# PNG Visualization Logic (Matplotlib Optimized)
# ==============================================================================

def visualize_png(graph: nx.Graph, output_path: Path, layout_name: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("❌ 错误: 请安装 matplotlib", file=sys.stderr)
        sys.exit(1)

    print(f"🎨 正在绘制 PNG (Layout: {layout_name})...")
    
    plt.figure(figsize=(16, 12), dpi=200) # 高清画布
    
    # 1. 布局计算
    layout_func = LAYOUTS_2D.get(layout_name, nx.spring_layout)
    # k 参数控制节点间距，iterations 控制迭代次数让图更展开
    pos = layout_func(graph, k=0.5, iterations=50) if layout_name == "spring" else layout_func(graph)

    # 2. 节点样式逻辑
    # 获取度数用于计算大小
    d = dict(graph.degree)
    # 归一化大小: 最小 300, 最大 3000
    node_sizes = [300 + (d.get(n, 0) * 100) for n in graph.nodes()]
    
    # 获取实体类型用于颜色
    types = [graph.nodes[n].get("entity_type", "default") for n in graph.nodes()]
    unique_types = list(set(types))
    # 建立颜色映射
    color_map = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(unique_types)}
    node_colors = [color_map[t] for t in types]

    # 3. 绘制边 (使用弧线 connectionstyle="arc3,rad=0.1" 增加美感)
    nx.draw_networkx_edges(
        graph, pos, 
        alpha=0.4, 
        edge_color="#94a3b8", 
        width=1.0, 
        connectionstyle="arc3,rad=0.1"
    )

    # 4. 绘制节点
    nx.draw_networkx_nodes(
        graph, pos, 
        node_size=node_sizes, 
        node_color=node_colors, 
        alpha=0.9, 
        edgecolors="white", # 节点白色描边
        linewidths=2
    )

    # 5. 绘制标签 (只有大节点才显示标签，避免拥挤)
    # 计算度数阈值，只显示 Top 80% 重要的节点标签
    # 或者简单点：全部显示但调整字体
    labels = {n: n for n in graph.nodes()}
    nx.draw_networkx_labels(
        graph, pos, 
        labels=labels, 
        font_size=8, 
        font_family="sans-serif",
        font_color="#1e293b",
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    )

    # 6. 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=t,
                   markerfacecolor=color_map[t], markersize=10)
        for t in unique_types
    ]
    plt.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=10)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"✨ PNG 图片已导出: {output_path}")

# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    graph_path = ensure_graph_exists(args.graphml)
    
    # 自动确定输出路径
    output_path = args.output
    if output_path is None:
        suffix = ".html" if args.format == "html" else ".png"
        output_path = graph_path.with_suffix(suffix)

    graph = load_graph(graph_path)

    if args.format == "html":
        visualize_html(graph, output_path, args.title_field, args.label_field)
    else:
        visualize_png(graph, output_path, args.layout)

if __name__ == "__main__":
    main()