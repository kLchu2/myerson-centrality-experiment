"""80頂点グリッドでキャッシュを使った計算のテスト"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from centrality.path_limited_myerson import all_path_limited_myerson_centralities
from graph_utils.generator import create_grid_graph

# 10x8グリッドを作成
print("=" * 60)
print("10x8グリッドグラフのテスト（キャッシュ使用）")
print("=" * 60)
G = create_grid_graph(10, 8)
print(f"グラフ: {G.number_of_nodes()}頂点, {G.number_of_edges()}辺")

# L_max=10で中心性を計算
print("\nL_max=10, r=0.8で計算:")
centralities = all_path_limited_myerson_centralities(G, L_max=10, r=0.8)
print(f"結果のキー数: {len(centralities)}")

# 最初の5頂点を表示
print("\n最初の5頂点の中心性:")
for node, cent in list(centralities.items())[:5]:
    print(f"  {node}: {cent}")

# 合計を表示
total = sum(centralities.values())
print(f"\n合計: {total}")
if total > 0:
    print(f"平均: {total / len(centralities)}")
else:
    print("ERROR: 合計が0です！")
