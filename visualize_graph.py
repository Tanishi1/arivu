import sqlite3
import json
import networkx as nx
import matplotlib.pyplot as plt
import argparse

# =====================================================================
# VERSION 1 & 2: NORMAL SPRING LAYOUT (Static Web)
# =====================================================================
def plot_normal_layout(version_id, ts, algo, edges, top_x=None):
    """
    Plots the graph using a traditional NetworkX spring layout (nodes clustered together).
    Does not unroll time (t-1, t0 are treated as normal nodes).
    """
    edges.sort(key=lambda x: abs(x['coeff']), reverse=True)
    
    if top_x is not None:
        edges_to_plot = edges[:top_x]
        title_suffix = f"(Top {top_x} Links)"
    else:
        edges_to_plot = edges
        title_suffix = f"(All {len(edges)} Links)"
        
    G = nx.DiGraph()
    for e in edges_to_plot:
        source = e['source']
        target = e['target']
        lag = e['lag']
        coeff = e['coeff']
        label = f"L{lag}\n{coeff:.2f}"
        G.add_edge(source, target, weight=abs(coeff), label=label, coeff=coeff)
        
    fig = plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=1.5, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color='#8CB9DF', alpha=0.9, edgecolors='black')
    
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['coeff'] >= 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['coeff'] < 0]
    
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, arrowstyle='-|>', arrowsize=20, edge_color='#2CA02C', width=2, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, arrowstyle='-|>', arrowsize=20, edge_color='#D62728', width=2, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)
    
    plt.title(f"Normal Spring Layout {title_suffix}\nVersion: {version_id[:8]} | Algo: {algo}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    return fig


# =====================================================================
# VERSION 3 & 4: TIME-UNROLLED LAYOUT (Temporal Flow)
# =====================================================================
def plot_time_unrolled(version_id, ts, algo, edges, top_x=None):
    """
    Plots the graph with time unrolled across the X-axis (t-3 -> t-2 -> t-1 -> t0).
    Arrows clearly flow forward through time.
    """
    edges.sort(key=lambda x: abs(x['coeff']), reverse=True)
    
    if top_x is not None:
        edges_to_plot = edges[:top_x]
        title_suffix = f"(Top {top_x} Links)"
    else:
        edges_to_plot = edges
        title_suffix = f"(All {len(edges)} Links)"
        
    G = nx.DiGraph()
    columns = {0: set()}
    
    for e in edges_to_plot:
        source_base = e['source']
        target_base = e['target']
        lag = e['lag']
        coeff = e['coeff']
        
        source_node = f"{source_base}\n(t-{lag})"
        target_node = f"{target_base}\n(t0)"
        
        if lag not in columns:
            columns[lag] = set()
            
        columns[lag].add(source_node)
        columns[0].add(target_node)
        
        G.add_edge(source_node, target_node, weight=abs(coeff), coeff=coeff)
        
    pos = {}
    for col_lag, nodes in columns.items():
        sorted_nodes = list(nodes)
        sorted_nodes.sort() 
        y_step = 2.0 / (len(sorted_nodes) + 1) if len(sorted_nodes) > 0 else 0
        current_y = 1.0 - y_step
        x = -col_lag 
        for node in sorted_nodes:
            pos[node] = (x, current_y)
            current_y -= y_step
            
    fig = plt.figure(figsize=(16, 9))
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='#E1F5FE', edgecolors='#0288D1', linewidths=2)
    
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['coeff'] >= 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['coeff'] < 0]
    
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, arrowstyle='-|>', arrowsize=30, edge_color='#43A047', width=2.5)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, arrowstyle='-|>', arrowsize=30, edge_color='#E53935', width=2.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    max_lag = max(columns.keys()) if columns.keys() else 0
    for i in range(max_lag, -1, -1):
        if i in columns:
            time_label = "Current Time (t0)" if i == 0 else f"{i*10}s Ago (t-{i})"
            plt.text(-i, 1.1, time_label, horizontalalignment='center', fontsize=12, fontweight='bold')
    
    plt.title(f"Time-Unrolled Causal Chain {title_suffix}\nVersion: {version_id[:8]} | Algo: {algo}", fontsize=15, pad=30)
    plt.axis('off')
    
    plt.plot([], [], color='#43A047', linewidth=2.5, label='Positive Causation (↑ drives ↑)')
    plt.plot([], [], color='#E53935', linewidth=2.5, label='Negative Causation (↑ drives ↓)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize Arivu Causal Graphs")
    parser.add_argument("--version", type=str, help="Specific graph version ID to plot (default: latest)", default=None)
    args = parser.parse_args()

    db_path = "data/arivu.db"
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    cur = conn.cursor()
    if args.version:
        cur.execute("SELECT version_id, timestamp, algorithm, edges FROM causal_graphs WHERE version_id LIKE ? ORDER BY timestamp DESC LIMIT 1", (f"%{args.version}%",))
    else:
        cur.execute("SELECT version_id, timestamp, algorithm, edges FROM causal_graphs ORDER BY timestamp DESC LIMIT 1")
        
    row = cur.fetchone()
    if not row:
        print("No graph found.")
        return
        
    version_id, ts, algo, edges_json = row
    edges = json.loads(edges_json)
    print(f"Loaded Graph: {version_id[:8]} | Total Edges: {len(edges)}")
    
    # 1. Normal layout (Static Web) - Top 20 Links
    plot_normal_layout(version_id, ts, algo, edges, top_x=20)
    
    # 3. Time-Unrolled layout (Temporal Flow) - Top 20 Links
    plot_time_unrolled(version_id, ts, algo, edges, top_x=20)
    
    # =====================================================================
    
    print("Opening plot windows. Close them to exit the script.")
    plt.show()

if __name__ == "__main__":
    main()
