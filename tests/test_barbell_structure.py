import networkx as nx

# バーベルグラフを作成
G = nx.barbell_graph(5, 1)

print("バーベルグラフBG(5,1)の構造:")
print(f"ノード数: {G.number_of_nodes()}")
print(f"エッジ数: {G.number_of_edges()}")
print(f"\nノード: {sorted(G.nodes())}")
print("\nエッジ:")
for u, v in sorted(G.edges()):
    print(f"  {u}-{v}")

# 媒介中心性を計算
betweenness = nx.betweenness_centrality(G)
print("\n媒介中心性:")
for node in sorted(G.nodes()):
    print(f"  ノード{node}: {betweenness[node]:.10f}")

# ノード5（ブリッジ）の接続を確認
print(f"\nノード5の隣接ノード: {list(G.neighbors(5))}")
print(f"ノード4の隣接ノード: {list(G.neighbors(4))}")
print(f"ノード6の隣接ノード: {list(G.neighbors(6))}")

# 最短パスの例を確認
print(f"\n0から10への最短パス長: {nx.shortest_path_length(G, 0, 10)}")
print(f"0から10への最短パス数: {len(list(nx.all_shortest_paths(G, 0, 10)))}")
