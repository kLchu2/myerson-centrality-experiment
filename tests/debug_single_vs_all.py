"""path_limited_myerson_centrality関数のデバッグ"""

import networkx as nx

from centrality.path_limited_myerson import (
    all_path_limited_myerson_centralities,
    path_limited_myerson_centrality,
)
from centrality.revised_myerson import revised_myerson_centrality

G = nx.path_graph(3)

print("=== 単一頂点計算 ===")
# パス長制限版
plm = path_limited_myerson_centrality(G, 1, L_max=10, r=1.0)
print(f"path_limited_myerson_centrality(G, 1, L_max=10): {plm}")

# 改訂版
rm = revised_myerson_centrality(G, 1, r=1.0)
print(f"revised_myerson_centrality(G, 1): {rm}")

print("\n=== 全頂点一括計算 ===")
# パス長制限版（一括）
all_plm = all_path_limited_myerson_centralities(G, L_max=10, r=1.0)
print(f"all_path_limited_myerson_centralities(G, L_max=10): {all_plm}")
