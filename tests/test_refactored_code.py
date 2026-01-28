"""リファクタリング後のコードをテスト"""

import shutil
import tempfile
import time

import networkx as nx

from centrality.path_limited_myerson import (
    CentralityCalculator,
    PathCounter,
    PathCountsCache,
    all_path_limited_myerson_centralities,
    path_limited_myerson_centrality,
)


def test_class_based_api():
    """クラスベースAPIのテスト"""
    print("\n=== クラスベースAPIのテスト ===")

    # 一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp()

    try:
        # グリッドグラフを作成
        G = nx.grid_2d_graph(4, 5)
        G = nx.convert_node_labels_to_integers(G)

        # PathCounterで計算
        counter = PathCounter(G)
        start = time.time()
        path_counts, actuaL_max = counter.compute_all_path_counts(
            max_length=5, verbose=True
        )
        calc_time = time.time() - start
        print(f"パス計算時間: {calc_time:.4f}秒")

        # キャッシュに保存（一時ディレクトリに）
        cache = PathCountsCache(cache_dir=temp_dir)
        cache_file = cache.save(G, path_counts)
        print(f"キャッシュ保存: {cache_file}")

        # キャッシュから読み込み
        start = time.time()
        loaded_counts = cache.load(G)
        load_time = time.time() - start
        print(f"キャッシュ読み込み時間: {load_time:.4f}秒")
        print(f"高速化率: {calc_time / load_time:.1f}x")

        # CentralityCalculatorで中心性を計算
        centralities = {}
        for v in G.nodes():
            centralities[v] = CentralityCalculator.calculate_centrality(
                loaded_counts, v, L_max=5, r=1.0
            )

        print(f"中心性値（最大5頂点）: {dict(list(centralities.items())[:5])}")
    finally:
        # 一時ディレクトリを削除
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_backward_compatibility():
    """後方互換性のテスト"""
    print("\n=== 後方互換性のテスト ===")

    G = nx.path_graph(6)

    # 既存の関数APIで計算
    start = time.time()
    c3 = path_limited_myerson_centrality(G, 3, L_max=2, r=1.0)
    single_time = time.time() - start
    print(f"単一頂点計算: node=3, centrality={c3:.4f}, 時間={single_time:.4f}秒")

    # バッチ計算
    start = time.time()
    all_centralities = all_path_limited_myerson_centralities(
        G, L_max=[2, 3], r=[1.0, 0.8], use_cache=False
    )
    batch_time = time.time() - start
    print(f"バッチ計算時間: {batch_time:.4f}秒")
    print(f"計算されたパターン数: {len(all_centralities)}")
    print(f"結果の一部: {list(all_centralities.items())[:2]}")


def test_tree_optimization():
    """木構造の最適化テスト"""
    print("\n=== 木構造の最適化テスト ===")

    # コームグラフ（木構造）
    G = nx.Graph()
    # スパイン
    for i in range(50):
        if i > 0:
            G.add_edge(i - 1, i)
    # 枝
    for i in range(0, 50, 2):
        G.add_edge(i, 50 + i // 2)

    print(f"頂点数: {G.number_of_nodes()}, 辺数: {G.number_of_edges()}")
    print(f"木構造: {G.number_of_edges() == G.number_of_nodes() - 1}")

    # 計算
    counter = PathCounter(G)
    start = time.time()
    path_counts, actuaL_max = counter.compute_all_path_counts(
        max_length=10, verbose=False
    )
    calc_time = time.time() - start
    print(f"パス計算時間: {calc_time:.4f}秒")

    # 中心性計算
    start = time.time()
    centralities = {
        v: CentralityCalculator.calculate_centrality(path_counts, v, L_max=10, r=1.0)
        for v in G.nodes()
    }
    cent_time = time.time() - start
    print(f"中心性計算時間: {cent_time:.4f}秒")

    max_node = max(centralities, key=centralities.get)
    print(f"最大中心性頂点: {max_node}, 値={centralities[max_node]:.4f}")


if __name__ == "__main__":
    test_class_based_api()
    test_backward_compatibility()
    test_tree_optimization()
    print("\n✅ 全てのテストが完了しました")
