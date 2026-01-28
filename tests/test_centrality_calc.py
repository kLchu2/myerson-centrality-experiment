"""中心性計算のデバッグ"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from centrality.path_limited_myerson import all_path_limited_myerson_centralities
from graph_utils.generator import create_grid_graph

# 小さい格子グラフを作成
G = create_grid_graph(3, 3)
print(f"グラフ: {G.number_of_nodes()}頂点, {G.number_of_edges()}辺")
print(f"頂点: {list(G.nodes())}")

# L_max=2で中心性を計算
print("\nL_max=2で計算:")
centralities = all_path_limited_myerson_centralities(G, L_max=2, r=0.8)
print(f"結果の型: {type(centralities)}")
print(f"結果のキー数: {len(centralities)}")
for node, cent in list(centralities.items())[:5]:
    print(f"  {node}: {cent}")
