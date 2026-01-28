"""
バッチ計算の効果検証（より大きなグラフで）
"""

import time

from centrality.path_limited_myerson import all_path_limited_myerson_centralities
from graph_utils.generator import create_comb_graph, create_grid_graph

# ========================================
# コームグラフで検証
# ========================================
print("=" * 70)
print("コームグラフ（n=10, l=4）での速度比較")
print("=" * 70)

G = create_comb_graph(n=10, l=4)
print(f"グラフ: 頂点数={G.number_of_nodes()}, 辺数={G.number_of_edges()}")
print()

L_max_values = [1, 2, 3, 5, 8, 10]
r_values = [0.2, 0.5, 0.8]

# 従来方式
print("【従来方式】18パターンを個別に計算 (6 L_max × 3 r):")
start = time.time()
results_old = []
for L_max in L_max_values:
    for r in r_values:
        result = all_path_limited_myerson_centralities(
            G, L_max=L_max, r=r, verbose=False, use_cache=False
        )
        results_old.append(result)
elapsed_old = time.time() - start
print(f"計算時間: {elapsed_old:.4f}秒")

# バッチ計算方式
print("\n【バッチ計算方式】一度に18パターン計算:")
start = time.time()
results_new = all_path_limited_myerson_centralities(
    G, L_max=L_max_values, r=r_values, verbose=False, use_cache=False
)
elapsed_new = time.time() - start
print(f"計算時間: {elapsed_new:.4f}秒")

speedup = elapsed_old / elapsed_new
print(f"\n高速化率: {speedup:.2f}倍")
print(f"時間短縮: {(elapsed_old - elapsed_new):.4f}秒")

# ========================================
# 格子グラフで検証
# ========================================
print("\n" + "=" * 70)
print("格子グラフ（5×4）での速度比較")
print("=" * 70)

G = create_grid_graph(m=5, n=4)
print(f"グラフ: 頂点数={G.number_of_nodes()}, 辺数={G.number_of_edges()}")
print()

L_max_values = [1, 2, 3, 4, 5]
r_values = [0.3, 0.5, 0.7]

# 従来方式
print("【従来方式】15パターンを個別に計算 (5 L_max × 3 r):")
start = time.time()
results_old = []
for L_max in L_max_values:
    for r in r_values:
        result = all_path_limited_myerson_centralities(
            G, L_max=L_max, r=r, verbose=False, use_cache=False
        )
        results_old.append(result)
elapsed_old = time.time() - start
print(f"計算時間: {elapsed_old:.4f}秒")

# バッチ計算方式
print("\n【バッチ計算方式】一度に15パターン計算:")
start = time.time()
results_new = all_path_limited_myerson_centralities(
    G, L_max=L_max_values, r=r_values, verbose=False, use_cache=False
)
elapsed_new = time.time() - start
print(f"計算時間: {elapsed_new:.4f}秒")

speedup = elapsed_old / elapsed_new
print(f"\n高速化率: {speedup:.2f}倍")
print(f"時間短縮: {(elapsed_old - elapsed_new):.4f}秒")

print("\n" + "=" * 70)
print("結論: パスの数え上げが一度で済むため、パターン数が多いほど効果大！")
print("=" * 70)
