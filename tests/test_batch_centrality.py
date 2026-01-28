"""
バッチ計算機能のテスト
"""

import time

from centrality.myerson import all_myerson_centralities
from centrality.path_limited_myerson import all_path_limited_myerson_centralities
from graph_utils.generator import create_comb_graph

# 小さなコームグラフを作成
G = create_comb_graph(n=5, l=2)
print(f"グラフ: 頂点数={G.number_of_nodes()}, 辺数={G.number_of_edges()}")
print()

# ========================================
# 1. パス長制限付きMyerson（L_maxとrのリスト）
# ========================================
print("=" * 60)
print("1. パス長制限付きMyerson中心性（バッチ計算）")
print("=" * 60)

L_max_list = [1, 2, 3, 5]
r_list = [0.2, 0.5, 0.8]

start = time.time()
results = all_path_limited_myerson_centralities(
    G, L_max=L_max_list, r=r_list, verbose=True, use_cache=False
)
elapsed = time.time() - start

print(f"\n計算時間: {elapsed:.4f}秒")
print(f"結果数: {len(results)}パターン")
print("\n各パターンの結果（頂点0の値のみ表示）:")
for (L_max, r), cent_dict in sorted(results.items()):
    print(f"  L_max={L_max}, r={r}: 頂点0の中心性 = {cent_dict[0]:.6f}")

# ========================================
# 2. 通常のMyerson（rのリスト）
# ========================================
print("\n" + "=" * 60)
print("2. 通常のMyerson中心性（バッチ計算）")
print("=" * 60)

r_list = [0.2, 0.5, 0.8]

start = time.time()
results = all_myerson_centralities(G, r=r_list, verbose=True)
elapsed = time.time() - start

print(f"\n計算時間: {elapsed:.4f}秒")
print(f"結果数: {len(results)}パターン")
print("\n各パターンの結果（頂点0の値のみ表示）:")
for r, cent_dict in sorted(results.items()):
    print(f"  r={r}: 頂点0の中心性 = {cent_dict[0]:.6f}")

# ========================================
# 3. 単一パラメータ（後方互換性チェック）
# ========================================
print("\n" + "=" * 60)
print("3. 単一パラメータでの動作確認（後方互換性）")
print("=" * 60)

result = all_path_limited_myerson_centralities(G, L_max=3, r=0.5, use_cache=False)
print(f"型: {type(result)}")
print(f"頂点0の中心性: {result[0]:.6f}")

# ========================================
# 4. 速度比較（従来方式 vs バッチ計算）
# ========================================
print("\n" + "=" * 60)
print("4. 速度比較（3つのL_max値で計算）")
print("=" * 60)

L_max_values = [2, 3, 5]

# 従来方式（個別に計算）
print("\n【従来方式】個別に3回計算:")
start = time.time()
for L_max in L_max_values:
    result = all_path_limited_myerson_centralities(
        G, L_max=L_max, r=0.5, verbose=False, use_cache=False
    )
elapsed_old = time.time() - start
print(f"計算時間: {elapsed_old:.4f}秒")

# バッチ計算方式
print("\n【バッチ計算方式】一度に3パターン計算:")
start = time.time()
results = all_path_limited_myerson_centralities(
    G, L_max=L_max_values, r=0.5, verbose=False, use_cache=False
)
elapsed_new = time.time() - start
print(f"計算時間: {elapsed_new:.4f}秒")

speedup = elapsed_old / elapsed_new
print(f"\n高速化率: {speedup:.2f}倍")

print("\n" + "=" * 60)
print("テスト完了！")
print("=" * 60)
