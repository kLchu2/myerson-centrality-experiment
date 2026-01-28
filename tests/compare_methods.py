"""DFS実装と既存実装の比較"""

import networkx as nx

from centrality.path_limited_myerson import PathCounter
from centrality.revised_myerson import count_all_paths_with_node

# 簡単なパスグラフ
G = nx.path_graph(3)
print("グラフ: 0-1-2\n")

# DFS実装
counter = PathCounter(G)
dfs_counts = counter.compute_all_path_counts(max_length=2, verbose=False)

print("=== DFS実装（深さ優先探索） ===")
for node in G.nodes():
    print(
        f"頂点{node}: B_1={dfs_counts[node].get(1, 0)}, B_2={dfs_counts[node].get(2, 0)}"
    )

# 既存実装（revised_myerson.py）
print("\n=== 既存実装（nx.all_simple_paths） ===")
for node in G.nodes():
    b1 = count_all_paths_with_node(G, node, 1)
    b2 = count_all_paths_with_node(G, node, 2)
    print(f"頂点{node}: B_1={b1}, B_2={b2}")
