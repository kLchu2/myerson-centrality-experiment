"""
キャッシュ機能のテスト
"""

import shutil
import time
from pathlib import Path

from centrality.path_limited_myerson import all_path_limited_myerson_centralities
from graph_utils.generator import create_comb_graph, create_grid_graph

# キャッシュディレクトリをクリーン
cache_dir = "cache"
if Path(cache_dir).exists():
    shutil.rmtree(cache_dir)

print("=" * 70)
print("キャッシュ機能のテスト")
print("=" * 70)

# ========================================
# テスト1: コームグラフでキャッシュの効果を確認
# ========================================
print("\n【テスト1】コームグラフでキャッシュの効果を確認")
print("-" * 70)

G = create_comb_graph(n=10, l=4)
print(f"グラフ: 頂点数={G.number_of_nodes()}, 辺数={G.number_of_edges()}")
print()

L_max_values = [1, 2, 3, 5, 8, 10]
r_values = [0.2, 0.5, 0.8]

# 1回目: キャッシュなし（計算＋保存）
print("【1回目】キャッシュなし - パス数を計算してキャッシュに保存")
start = time.time()
results1 = all_path_limited_myerson_centralities(
    G, L_max=L_max_values, r=r_values, verbose=True, use_cache=True, cache_dir=cache_dir
)
elapsed1 = time.time() - start
print(f"計算時間: {elapsed1:.4f}秒")

# 2回目: キャッシュあり（読み込みのみ）
print("\n【2回目】キャッシュあり - キャッシュから読み込み")
start = time.time()
results2 = all_path_limited_myerson_centralities(
    G, L_max=L_max_values, r=r_values, verbose=True, use_cache=True, cache_dir=cache_dir
)
elapsed2 = time.time() - start
print(f"計算時間: {elapsed2:.4f}秒")

# 結果の一致を確認
match = all(
    abs(results1[(l, r)][0] - results2[(l, r)][0]) < 1e-10
    for l in L_max_values  # noqa: E741
    for r in r_values
)
print(f"\n結果の一致: {'✓ 一致' if match else '✗ 不一致'}")
print(f"高速化率: {elapsed1 / elapsed2:.2f}倍")

# 3回目: 異なるパラメータでもキャッシュを再利用
print("\n【3回目】異なるL_maxとrでもキャッシュを再利用")
new_L_max = [2, 4, 6]
new_r = [0.3, 0.7]
start = time.time()
results3 = all_path_limited_myerson_centralities(
    G, L_max=new_L_max, r=new_r, verbose=True, use_cache=True, cache_dir=cache_dir
)
elapsed3 = time.time() - start
print(f"計算時間: {elapsed3:.4f}秒")

# ========================================
# テスト2: 格子グラフでキャッシュの効果を確認
# ========================================
print("\n" + "=" * 70)
print("【テスト2】格子グラフでキャッシュの効果を確認")
print("-" * 70)

G2 = create_grid_graph(m=5, n=4)
print(f"グラフ: 頂点数={G2.number_of_nodes()}, 辺数={G2.number_of_edges()}")
print()

L_max_values2 = [1, 2, 3, 4, 5]
r_values2 = [0.3, 0.5, 0.7]

# 1回目: キャッシュなし
print("【1回目】キャッシュなし")
start = time.time()
results_g1 = all_path_limited_myerson_centralities(
    G2,
    L_max=L_max_values2,
    r=r_values2,
    verbose=True,
    use_cache=True,
    cache_dir=cache_dir,
)
elapsed_g1 = time.time() - start
print(f"計算時間: {elapsed_g1:.4f}秒")

# 2回目: キャッシュあり
print("\n【2回目】キャッシュあり")
start = time.time()
results_g2 = all_path_limited_myerson_centralities(
    G2,
    L_max=L_max_values2,
    r=r_values2,
    verbose=True,
    use_cache=True,
    cache_dir=cache_dir,
)
elapsed_g2 = time.time() - start
print(f"計算時間: {elapsed_g2:.4f}秒")
print(f"高速化率: {elapsed_g1 / elapsed_g2:.2f}倍")

# ========================================
# テスト3: キャッシュ無効化オプション
# ========================================
print("\n" + "=" * 70)
print("【テスト3】キャッシュ無効化オプション")
print("-" * 70)

print("use_cache=Falseで実行（キャッシュを使わない）")
start = time.time()
results_no_cache = all_path_limited_myerson_centralities(
    G, L_max=3, r=0.5, verbose=True, use_cache=False
)
elapsed_no_cache = time.time() - start
print(f"計算時間: {elapsed_no_cache:.4f}秒")

# ========================================
# まとめ
# ========================================
print("\n" + "=" * 70)
print("まとめ")
print("=" * 70)
print("コームグラフ（50頂点）:")
print(f"  初回計算: {elapsed1:.4f}秒")
print(f"  キャッシュ利用: {elapsed2:.4f}秒 ({elapsed1 / elapsed2:.1f}倍高速)")
print("\n格子グラフ（20頂点）:")
print(f"  初回計算: {elapsed_g1:.4f}秒")
print(f"  キャッシュ利用: {elapsed_g2:.4f}秒 ({elapsed_g1 / elapsed_g2:.1f}倍高速)")
print(f"\nキャッシュディレクトリ: {cache_dir}/")
print(f"キャッシュファイル数: {len(list(Path(cache_dir).glob('*.json')))}")

print("\n✓ キャッシュ機能のテスト完了")
