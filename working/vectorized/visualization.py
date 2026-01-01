#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GraphML Knowledge Graph Visualizer (Stream Loading Edition)

特性：
1. 【流式加载】解决大数据白屏问题，节点快速逐批出现（动画效果）。
2. 【聚焦模式】点击节点，仅保留其邻居，双击空白还原。
3. 【性能优化】加载完成后自动冻结物理引擎，不再持续占用 CPU。

Usage:
    python visualize_stream.py data.graphml --output graph.html
"""

import argparse
import sys
import json
import random
from pathlib import Path
import networkx as nx

# --- 配色方案 ---
PALETTE = [
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", 
    "#ec4899", "#06b6d4", "#84cc16", "#6366f1", "#14b8a6"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>知识图谱 - 流式加载</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #2563eb; --glass: rgba(255, 255, 255, 0.9); }
        body, html { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; font-family: 'Inter', sans-serif; background: #0f172a; }
        
        /* 1. 加载遮罩层 */
        #loader {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: #0f172a; z-index: 999;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            transition: opacity 0.5s;
        }
        .loader-text { color: white; font-size: 24px; margin-bottom: 20px; font-weight: 600; }
        .progress-bar { width: 300px; height: 6px; background: #334155; border-radius: 3px; overflow: hidden; }
        .progress-fill { height: 100%; background: #3b82f6; width: 0%; transition: width 0.1s; }
        
        /* 2. 画布 */
        #mynetwork { width: 100%; height: 100%; position: absolute; top: 0; left: 0; z-index: 1; }

        /* 3. 顶部状态栏 */
        #status-bar {
            position: absolute; top: 20px; left: 50%; transform: translateX(-50%);
            background: var(--glass); padding: 8px 20px; border-radius: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5); z-index: 10;
            display: flex; align-items: center; gap: 15px;
            opacity: 0; pointer-events: none; transition: opacity 0.3s;
        }
        #status-bar.visible { opacity: 1; pointer-events: auto; }
        .status-text { color: #1e293b; font-size: 14px; font-weight: 600; }
        .btn-reset {
            background: #ef4444; color: white; border: none; padding: 5px 12px;
            border-radius: 15px; cursor: pointer; font-size: 12px;
        }
        .btn-reset:hover { background: #dc2626; }

        /* 4. 侧边栏 (极简版) */
        #sidebar {
            position: absolute; top: 20px; right: 20px; width: 300px; bottom: 20px;
            background: var(--glass); backdrop-filter: blur(10px);
            border-radius: 12px; padding: 20px; display: flex; flex-direction: column;
            transform: translateX(120%); transition: transform 0.3s; z-index: 10;
        }
        #sidebar.open { transform: translateX(0); }
        .sidebar-title { margin: 0 0 10px 0; font-size: 18px; color: #1e293b; }
        .list-container { flex: 1; overflow-y: auto; }
        .list-item { 
            padding: 8px; border-bottom: 1px solid #e2e8f0; cursor: pointer; font-size: 13px;
        }
        .list-item:hover { background: #eff6ff; color: var(--primary); }

        /* 右下角工具 */
        .tools { position: absolute; bottom: 20px; right: 20px; z-index: 20; display: flex; gap: 10px; }
        .btn-tool { background: white; border: none; width: 40px; height: 40px; border-radius: 50%; cursor: pointer; font-size: 18px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }
    </style>
</head>
<body>

    <!-- 加载页 -->
    <div id="loader">
        <div class="loader-text">正在构建知识宇宙...</div>
        <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
        <div style="color:#64748b; margin-top:10px; font-size:12px;" id="loader-status">0 / 0</div>
    </div>

    <!-- 聚焦状态条 -->
    <div id="status-bar">
        <span class="status-text" id="focus-name">节点名</span>
        <button class="btn-reset" onclick="resetGraph()">重置视图</button>
    </div>

    <div id="mynetwork"></div>

    <div class="tools">
        <button class="btn-tool" onclick="togglePhysics()" title="开启/停止物理运动">❄️</button>
        <button class="btn-tool" onclick="network.fit()" title="全图适配">🔍</button>
    </div>

    <!-- 数据源 -->
    <script id="graph-data" type="application/json">__GRAPH_JSON__</script>

    <script>
        // 1. 数据解析
        const rawData = JSON.parse(document.getElementById('graph-data').textContent);
        const allNodes = rawData.nodes;
        const allEdges = rawData.edges;

        // 初始化 DataSet (一开始是空的)
        const nodesDataSet = new vis.DataSet([]);
        const edgesDataSet = new vis.DataSet([]);

        // 创建 View 用于聚焦过滤
        // 核心逻辑：filterFunction 决定显示哪些节点
        let filterState = {
            active: false,
            allowedIds: new Set()
        };

        const nodesView = new vis.DataView(nodesDataSet, {
            filter: function (node) {
                if (!filterState.active) return true;
                return filterState.allowedIds.has(node.id);
            }
        });

        const edgesView = new vis.DataView(edgesDataSet, {
            filter: function (edge) {
                if (!filterState.active) return true;
                return filterState.allowedIds.has(edge.from) && filterState.allowedIds.has(edge.to);
            }
        });

        // 2. 初始化 Network
        const container = document.getElementById('mynetwork');
        const data = { nodes: nodesView, edges: edgesView };
        
        const options = {
            nodes: {
                shape: 'dot',
                font: { face: 'Inter', size: 14, color: '#e2e8f0' }, // 深色背景下的字体
                shadow: { enabled: false }
            },
            edges: {
                color: { color: '#475569', highlight: '#3b82f6', opacity: 0.5 },
                width: 1,
                smooth: { type: 'continuous' }
            },
            physics: {
                enabled: true,
                solver: 'forceAtlas2Based', // 适合这种“爆炸”式出现的布局
                forceAtlas2Based: {
                    gravitationalConstant: -50,
                    centralGravity: 0.005,
                    springLength: 100,
                    springConstant: 0.08,
                    damping: 0.4
                },
                stabilization: { enabled: false } // 关闭初始稳定化，实现动态出现效果
            },
            interaction: { hover: true, tooltipDelay: 200 }
        };

        const network = new vis.Network(container, data, options);

        // 3. 流式加载逻辑 (Streaming Animation)
        let loadedCount = 0;
        const totalNodes = allNodes.length;
        const BATCH_SIZE = 50; // 每次加载 50 个，保证速度快且有动画感
        
        function loadNextBatch() {
            if (loadedCount >= totalNodes) {
                // 加载完毕
                finishLoading();
                return;
            }

            // 提取一批数据
            const end = Math.min(loadedCount + BATCH_SIZE, totalNodes);
            const nodeBatch = allNodes.slice(loadedCount, end);
            
            // 找出这批节点相关的边 (为了让边和节点一起出现)
            // 简单的做法是：只要边的两个端点都已经在 DataSet 里了，就添加
            // 但为了速度，我们可以先全部把节点加完，最后统一加边；或者分批加。
            // 这里的策略：先加节点，让它们飘一会儿
            
            nodesDataSet.add(nodeBatch);
            
            loadedCount = end;
            
            // 更新 UI
            const pct = Math.round((loadedCount / totalNodes) * 100);
            document.getElementById('progress').style.width = pct + '%';
            document.getElementById('loader-status').innerText = `${loadedCount} / ${totalNodes}`;

            // 下一帧继续
            requestAnimationFrame(loadNextBatch);
        }

        function finishLoading() {
            // 节点加完了，现在一次性把边加上（或者也分批，但边一般不影响渲染崩溃，只影响物理）
            document.querySelector('.loader-text').innerText = "正在建立连接...";
            
            setTimeout(() => {
                edgesDataSet.add(allEdges);
                
                // 隐藏遮罩
                document.getElementById('loader').style.opacity = 0;
                setTimeout(() => { 
                    document.getElementById('loader').style.display = 'none'; 
                    // 开启物理引擎跑一会，整理形状
                    network.fit();
                }, 500);

                // 5秒后自动冻结物理引擎，防止发热
                setTimeout(() => {
                    console.log("自动冻结物理引擎");
                    network.setOptions({ physics: { enabled: false } });
                }, 5000);
            }, 100);
        }

        // 开始加载
        requestAnimationFrame(loadNextBatch);


        // 4. 交互逻辑：聚焦模式
        network.on("click", function (params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                enterFocusMode(nodeId);
            } else {
                // 点击空白
                resetGraph();
            }
        });

        function enterFocusMode(nodeId) {
            const node = nodesDataSet.get(nodeId);
            
            // 获取邻居
            const connected = network.getConnectedNodes(nodeId);
            const neighborhood = new Set(connected);
            neighborhood.add(nodeId);

            // 设置过滤器
            filterState.active = true;
            filterState.allowedIds = neighborhood;
            
            // 刷新视图
            nodesView.refresh();
            edgesView.refresh();

            // UI
            document.getElementById('status-bar').classList.add('visible');
            document.getElementById('focus-name').innerText = node.label;
            
            // 开启一点点物理，让它们聚拢，然后fit
            network.setOptions({ physics: { enabled: true } });
            setTimeout(() => {
                network.fit({ animation: true });
                // 再次冻结
                // network.setOptions({ physics: { enabled: false } }); 
            }, 500);
        }

        window.resetGraph = function() {
            filterState.active = false;
            filterState.allowedIds.clear();
            nodesView.refresh();
            edgesView.refresh();
            document.getElementById('status-bar').classList.remove('visible');
            network.fit();
        };

        window.togglePhysics = function() {
            const status = network.physics.physicsEnabled;
            network.setOptions({ physics: { enabled: !status } });
        };

    </script>
</body>
</html>
"""

def process(graph_path, output_path=None):
    if not Path(graph_path).exists():
        print(f"Error: {graph_path} not found.")
        return

    print(f"Reading {graph_path}...")
    G = nx.read_graphml(graph_path)
    
    # 预处理数据
    nodes = []
    # 计算度用于大小
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1

    # 颜色
    types = list(set([str(G.nodes[n].get("entity_type", "Unknown")) for n in G.nodes]))
    color_map = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types)}

    for n, data in G.nodes(data=True):
        lbl = str(data.get("label", n))
        # 兼容性处理
        if lbl == str(n) and "name" in data: lbl = data["name"]
        
        etype = str(data.get("entity_type", "Unknown"))
        
        nodes.append({
            "id": n,
            "label": lbl,
            "group": etype,
            "title": f"{lbl} ({etype})\n{str(data.get('description', ''))[:50]}...",
            "value": 10 + (degrees.get(n, 0) / max_deg) * 40,
            "color": color_map.get(etype, "#64748b")
        })
    
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "from": u, 
            "to": v,
            "id": f"{u}-{v}-{random.randint(0,100000)}"
        })

    # 打乱顺序，让出现效果更随机好看
    random.shuffle(nodes)

    data_json = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    
    html = HTML_TEMPLATE.replace("__GRAPH_JSON__", data_json)
    
    if not output_path:
        output_path = Path(graph_path).with_suffix(".html")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Success! Open {output_path} to see the animation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graphml", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    process(args.graphml, args.output)