"""
実験スクリプトでキャッシュ機能を実際に使用する例
"""

import time

from centrality.path_limited_myerson import all_path_limited_myerson_centralities
from graph_utils.generator import create_comb_graph, create_grid_graph

print("=" * 70)
print("実験シナリオ：複数のパラメータで実験を繰り返す")
print("=" * 70)

# グラフを作成
G = create_grid_graph(m=5, n=4)
print(f"\nグラフ: {G.number_of_nodes()}頂点, {G.number_of_edges()}辺")

# 実験1: L_maxを変えて影響を調査
print("\n" + "-" * 70)
print("実験1: L_maxを変えて情報損失を調査")
print("-" * 70)
L_max_values = [1, 2, 3, 4, 5, 6]
r = 0.5

start = time.time()
results1 = all_path_limited_myerson_centralities(
    G, L_max=L_max_values, r=r, verbose=True, use_cache=True
)
elapsed1 = time.time() - start

print(f"\n実行時間: {elapsed1:.4f}秒")
first_node = list(G.nodes())[0]
print(f"\n頂点{first_node}の中心性の変化:")
for L_max in L_max_values:
    print(f"  L_max={L_max}: {results1[(L_max, r)][first_node]:.6f}")

# 実験2: rの値を変えて比較（キャッシュを再利用）
print("\n" + "-" * 70)
print("実験2: 影響定数rを変えて比較（同じグラフ）")
print("-" * 70)
L_max = 5
r_values = [0.2, 0.4, 0.6, 0.8, 1.0]

start = time.time()
results2 = all_path_limited_myerson_centralities(
    G, L_max=L_max, r=r_values, verbose=True, use_cache=True
)
elapsed2 = time.time() - start

print(f"\n実行時間: {elapsed2:.4f}秒（キャッシュ再利用！）")
print(f"\n頂点{first_node}の中心性の変化:")
for r_val in r_values:
    print(f"  r={r_val}: {results2[(L_max, r_val)][first_node]:.6f}")

# 実験3: 全パターンを一度に計算（キャッシュ再利用）
print("\n" + "-" * 70)
print("実験3: 全パターンを一度に計算（キャッシュ再利用）")
print("-" * 70)
L_max_all = [2, 3, 5]
r_all = [0.3, 0.5, 0.7]

start = time.time()
results3 = all_path_limited_myerson_centralities(
    G, L_max=L_max_all, r=r_all, verbose=True, use_cache=True
)
elapsed3 = time.time() - start

print(
    f"\n実行時間: {elapsed3:.4f}秒（{len(L_max_all) * len(r_all)}パターン、キャッシュ利用）"
)

# 実験4: 別のグラフで実験（新しいキャッシュを作成）
print("\n" + "-" * 70)
print("実験4: 別のグラフ（コームグラフ）で実験")
print("-" * 70)
G2 = create_comb_graph(n=8, l=3)
print(f"グラフ: {G2.number_of_nodes()}頂点, {G2.number_of_edges()}辺")

start = time.time()
results4 = all_path_limited_myerson_centralities(
    G2, L_max=[1, 2, 3, 5], r=[0.3, 0.5, 0.7], verbose=True, use_cache=True
)
elapsed4 = time.time() - start

print(f"\n実行時間: {elapsed4:.4f}秒（新しいキャッシュを作成）")

# 再度同じグラフで実験（キャッシュを利用）
print("\n" + "-" * 70)
print("実験5: 同じコームグラフで追加実験（キャッシュ利用）")
print("-" * 70)

start = time.time()
results5 = all_path_limited_myerson_centralities(
    G2, L_max=[2, 4], r=[0.4, 0.6], verbose=True, use_cache=True
)
elapsed5 = time.time() - start

print(f"\n実行時間: {elapsed5:.4f}秒（キャッシュから瞬時に計算！）")

# まとめ
print("\n" + "=" * 70)
print("まとめ")
print("=" * 70)
print(f"実験1（初回、キャッシュ作成）: {elapsed1:.4f}秒")
print(f"実験2（キャッシュ再利用）    : {elapsed2:.4f}秒")
print(f"実験3（キャッシュ再利用）    : {elapsed3:.4f}秒")
print(f"実験4（新グラフ、初回）      : {elapsed4:.4f}秒")
print(f"実験5（キャッシュ再利用）    : {elapsed5:.4f}秒")
print("\n💡 同じグラフなら、何度実験してもキャッシュから高速に計算できる！")
