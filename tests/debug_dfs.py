"""DFS実装のデバッグ"""

import networkx as nx

from centrality.path_limited_myerson import PathCounter

# 簡単なパスグラフ A-B-C (0-1-2)
G = nx.path_graph(3)
print("グラフ: 0-1-2")
print(f"頂点: {list(G.nodes())}")
print(f"辺: {list(G.edges())}")

# 手計算（順序を区別しない）:
# 長さ1のパス: {0,1}, {1,2} = 2本
#   - 頂点0を含む: {0,1} = 1本
#   - 頂点1を含む: {0,1}, {1,2} = 2本
#   - 頂点2を含む: {1,2} = 1本
#
# 長さ2のパス: {0,1,2} = 1本
#   - 頂点0を含む: {0,1,2} = 1本
#   - 頂点1を含む: {0,1,2} = 1本
#   - 頂点2を含む: {0,1,2} = 1本

print("\n=== 期待される結果（手計算・順序なし） ===")
print("B_1(0) = 1, B_1(1) = 2, B_1(2) = 1")
print("B_2(0) = 1, B_2(1) = 1, B_2(2) = 1")

counter = PathCounter(G)
path_counts = counter.compute_all_path_counts(max_length=2, verbose=False)

print("\n=== 実際の結果（DFS） ===")
for node in G.nodes():
    print(f"頂点{node}: ", end="")
    for length in [1, 2]:
        count = path_counts[node].get(length, 0)
        print(f"B_{length}({node})={count}", end=" ")
    print()

print("\n=== 中心性計算 (r=1.0) ===")
print("Y*(v) = Σ B_l(v)/(l+1) * r^l")
for node in G.nodes():
    y = 0
    for length in [1, 2]:
        B_l = path_counts[node].get(length, 0)
        y += B_l / (length + 1)
    print(f"Y*({node}) = {y:.4f}")
# 手計算: Y*(0) = 1/2 + 1/3 = 0.5 + 0.333 = 0.833
#         Y*(1) = 2/2 + 1/3 = 1.0 + 0.333 = 1.333
#         Y*(2) = 1/2 + 1/3 = 0.5 + 0.333 = 0.833
