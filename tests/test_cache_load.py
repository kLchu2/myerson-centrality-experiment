"""path_countsの内容を確認"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from centrality.base import PathCountsCache  # noqa: E402
from graph_utils.generator import create_grid_graph  # noqa: E402

# 10x8グリッドを作成
G = create_grid_graph(10, 8)

# キャッシュから読み込み
cache = PathCountsCache()
result = cache.load(G)

if result:
    path_counts, actuaL_max_length = result
    print("キャッシュ読み込み成功")
    print(f"actuaL_max_length: {actuaL_max_length}")
    print(f"path_countsの型: {type(path_counts)}")
    print(f"path_countsのキー数: {len(path_counts)}")

    # 最初の頂点を確認
    first_node = list(path_counts.keys())[0]
    print(f"\n最初の頂点: {first_node}")
    print(f"型: {type(first_node)}")
    print(f"長さ辞書の型: {type(path_counts[first_node])}")
    print(f"長さ辞書の内容: {dict(path_counts[first_node])}")

    # グラフの頂点を確認
    print(f"\nグラフの頂点（最初の5個）: {list(G.nodes())[:5]}")
    print(f"グラフ頂点の型: {type(list(G.nodes())[0])}")

    # キーが一致するか確認
    graph_first_node = list(G.nodes())[0]
    print(f"\nグラフの最初の頂点: {graph_first_node}, 型: {type(graph_first_node)}")
    print(f"キャッシュの最初の頂点: {first_node}, 型: {type(first_node)}")
    print(f"一致: {graph_first_node == first_node}")
    print(f"path_counts[graph_first_node]が存在: {graph_first_node in path_counts}")
else:
    print("キャッシュが見つかりません")
