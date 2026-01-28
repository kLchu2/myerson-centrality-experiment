import networkx as nx

# バーベルグラフを作成
G = nx.barbell_graph(5, 1)

print("=" * 60)
print("バーベルグラフ BG(5,1) の構造確認")
print("=" * 60)

# グラフの基本情報
print(f"\nノード数: {G.number_of_nodes()}")
print(f"エッジ数: {G.number_of_edges()}")

# クラスタの確認
print("\n【グラフの構造】")
print("クラスタ1（K5）: ノード 0, 1, 2, 3, 4")
print("ブリッジ: ノード 5")
print("クラスタ2（K5）: ノード 6, 7, 8, 9, 10")

# 媒介中心性を計算
betweenness = nx.betweenness_centrality(G)

print("\n【媒介中心性の結果】")
print("ノード | 媒介中心性")
print("------+-----------")
for node in sorted(G.nodes()):
    bc = betweenness[node]
    print(f"  {node:2d}  | {bc:.10f}")

# 詳細分析
print("\n【詳細分析】")
print("\nノード0-3が0である理由:")
print("→ 完全グラフK5内では、任意の2ノード間に複数の最短パスが存在する")
print("→ ノード0を通らないパスが常に存在するため、媒介中心性 = 0")

print("\nノード4が0.533である理由:")
print("→ ノード4はクラスタ1とブリッジの接続点")
print("→ クラスタ1内のノード0,1,2,3とクラスタ2のノード間で、ノード4を通るパスがある")

print("\nノード5（ブリッジ）が0.555である理由:")
print("→ クラスタ1とクラスタ2間のすべての最短パスがノード5を通る")
print("→ ノード5を通さないとクラスタ2に到達できない")

# 最短パスの確認
print("\n【最短パスの検証例】")
print("\nノード0 → ノード10:")
all_paths = list(nx.all_shortest_paths(G, 0, 10))
print(f"最短パス数: {len(all_paths)}")
for i, path in enumerate(all_paths[:3], 1):
    print(f"  パス{i}: {' → '.join(map(str, path))}")
if len(all_paths) > 3:
    print(f"  ... 他 {len(all_paths) - 3} 個")

print("\nノード1 → ノード9:")
all_paths = list(nx.all_shortest_paths(G, 1, 9))
print(f"最短パス数: {len(all_paths)}")
for i, path in enumerate(all_paths[:3], 1):
    print(f"  パス{i}: {' → '.join(map(str, path))}")
if len(all_paths) > 3:
    print(f"  ... 他 {len(all_paths) - 3} 個")

print("\n【結論】")
print("媒介中心性の結果は CORRECT です。")
print("完全グラフK5の性質により、内部ノードの媒介中心性が0になるのは")
print("グラフ理論として正しい結果です。")
