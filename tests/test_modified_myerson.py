"""
ウォークベース修正版マイヤーソン中心性のテスト

テスト内容:
1. 単純なグラフでの計算結果の検証
2. 式 (2.8) の導出根拠に基づく検証
3. 既存のパスベース実装との比較
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from centrality.modified_myerson import (
    all_modified_myerson_centralities,
    all_modified_myerson_centralities_cumulative,
    compute_walk_appearance_count,
    compute_walk_count_matrix_power,
    modified_myerson_centrality,
)


class TestComputeWalkCountMatrixPower:
    """隣接行列のべき乗計算のテスト"""

    def test_path_graph_3(self):
        """パスグラフ P_3 (0-1-2) でのテスト"""
        G = nx.path_graph(3)  # 0-1-2
        matrix_powers, nodes, node_to_idx, M = compute_walk_count_matrix_power(G, 3)

        # 隣接行列 M の確認
        expected_M = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(M, expected_M)

        # M^1 = M
        np.testing.assert_array_equal(matrix_powers[1], expected_M)

        # M^2 の確認
        expected_M2 = np.array(
            [
                [1, 0, 1],
                [0, 2, 0],
                [1, 0, 1],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(matrix_powers[2], expected_M2)

        # M^3 の確認
        expected_M3 = np.array(
            [
                [0, 2, 0],
                [2, 0, 2],
                [0, 2, 0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(matrix_powers[3], expected_M3)


class TestComputeWalkAppearanceCount:
    """ウォーク出現回数計算のテスト"""

    def test_path_graph_3_length_1(self):
        """パスグラフ P_3 で長さ 1 のウォークでの出現回数"""
        G = nx.path_graph(3)  # 0-1-2
        matrix_powers, nodes, node_to_idx, M = compute_walk_count_matrix_power(G, 1)
        n = len(nodes)

        # 長さ 1 のウォーク: 0-1, 1-0, 1-2, 2-1 (合計 4 本)
        # 各頂点の出現回数:
        # - 頂点 0: 0-1 と 1-0 に出現 -> 2回
        # - 頂点 1: 0-1, 1-0, 1-2, 2-1 の全てに出現 -> 4回
        # - 頂点 2: 1-2 と 2-1 に出現 -> 2回

        s_0 = compute_walk_appearance_count(matrix_powers, M, n, node_to_idx[0], 1)
        s_1 = compute_walk_appearance_count(matrix_powers, M, n, node_to_idx[1], 1)
        s_2 = compute_walk_appearance_count(matrix_powers, M, n, node_to_idx[2], 1)

        assert s_0 == 2.0, f"頂点0の出現回数が2ではなく{s_0}"
        assert s_1 == 4.0, f"頂点1の出現回数が4ではなく{s_1}"
        assert s_2 == 2.0, f"頂点2の出現回数が2ではなく{s_2}"

    def test_path_graph_3_length_2(self):
        """パスグラフ P_3 で長さ 2 のウォークでの出現回数"""
        G = nx.path_graph(3)  # 0-1-2
        matrix_powers, nodes, node_to_idx, M = compute_walk_count_matrix_power(G, 2)
        n = len(nodes)

        # 長さ 2 のウォーク:
        # 0から始まる: 0-1-0, 0-1-2
        # 1から始まる: 1-0-1, 1-2-1
        # 2から始まる: 2-1-0, 2-1-2
        # 合計 6 本

        # 各頂点の出現回数（3点×6本のウォーク）:
        # - 頂点 0: 0-1-0(2回), 0-1-2(1回), 1-0-1(1回), 2-1-0(1回) -> 合計5回
        # - 頂点 1: 0-1-0(1回), 0-1-2(1回), 1-0-1(2回), 1-2-1(2回), 2-1-0(1回), 2-1-2(1回) -> 合計8回
        # - 頂点 2: 0-1-2(1回), 1-2-1(1回), 2-1-0(1回), 2-1-2(2回) -> 合計5回

        s_0 = compute_walk_appearance_count(matrix_powers, M, n, node_to_idx[0], 2)
        s_1 = compute_walk_appearance_count(matrix_powers, M, n, node_to_idx[1], 2)
        s_2 = compute_walk_appearance_count(matrix_powers, M, n, node_to_idx[2], 2)

        assert s_0 == 5.0, f"頂点0の出現回数が5ではなく{s_0}"
        assert s_1 == 8.0, f"頂点1の出現回数が8ではなく{s_1}"
        assert s_2 == 5.0, f"頂点2の出現回数が5ではなく{s_2}"


class TestWalkBasedRevisedMyersonCentrality:
    """修正版マイヤーソン中心性のテスト"""

    def test_normalization(self):
        """正規化（l+1で割る）のテスト"""
        G = nx.path_graph(3)

        # 長さ 1 のときは 1+1=2 で割る
        sigma_1 = all_modified_myerson_centralities(G, walk_length=1, normalize=True)
        s_1 = all_modified_myerson_centralities(G, walk_length=1, normalize=False)

        for node in G.nodes():
            assert sigma_1[node] == s_1[node] / 2, f"頂点{node}の正規化が正しくない"

        # 長さ 2 のときは 2+1=3 で割る
        sigma_2 = all_modified_myerson_centralities(G, walk_length=2, normalize=True)
        s_2 = all_modified_myerson_centralities(G, walk_length=2, normalize=False)

        for node in G.nodes():
            assert abs(sigma_2[node] - s_2[node] / 3) < 1e-10, (
                f"頂点{node}の正規化が正しくない"
            )

    def test_star_graph(self):
        """スターグラフでの中心性"""
        # スターグラフ: 中心頂点 0 と 葉 1, 2, 3
        G = nx.star_graph(3)

        # 長さ 1 のウォーク
        centralities = all_modified_myerson_centralities(G, walk_length=1)

        # 中心頂点は全てのエッジに含まれるので最も高い
        center = 0
        for leaf in [1, 2, 3]:
            assert centralities[center] > centralities[leaf], (
                f"中心頂点の中心性({centralities[center]})が葉({centralities[leaf]})より高くない"
            )

    def test_single_vertex_function(self):
        """単一頂点計算関数のテスト"""
        G = nx.path_graph(3)

        # 単一頂点計算と全頂点計算の結果が一致することを確認
        all_centralities = all_modified_myerson_centralities(G, walk_length=2)

        for node in G.nodes():
            single_centrality = modified_myerson_centrality(G, node, walk_length=2)
            assert abs(all_centralities[node] - single_centrality) < 1e-10, (
                f"頂点{node}の結果が一致しない"
            )


class TestCumulativeCentrality:
    """累積中心性のテスト"""

    def test_cumulative_equals_sum(self):
        """累積中心性が各長さの合計と一致することのテスト"""
        G = nx.path_graph(5)
        max_length = 4
        r = 0.5

        # 累積計算
        cumulative = all_modified_myerson_centralities_cumulative(
            G, max_walk_length=max_length, r=r
        )

        # 手動で各長さの合計を計算
        manual_sum = {node: 0.0 for node in G.nodes()}
        for wlen in range(1, max_length + 1):
            sigma_l = all_modified_myerson_centralities(G, walk_length=wlen)
            for node in G.nodes():
                manual_sum[node] += sigma_l[node] * (r**wlen)

        # 結果の比較
        for node in G.nodes():
            assert abs(cumulative[node] - manual_sum[node]) < 1e-10, (
                f"頂点{node}の累積中心性が一致しない"
            )


class TestSymmetry:
    """対称性のテスト"""

    def test_path_graph_symmetry(self):
        """パスグラフの対称性: 両端の頂点は同じ中心性を持つ"""
        G = nx.path_graph(5)  # 0-1-2-3-4

        centralities = all_modified_myerson_centralities(G, walk_length=3)

        # 頂点 0 と 4 は対称
        assert abs(centralities[0] - centralities[4]) < 1e-10

        # 頂点 1 と 3 は対称
        assert abs(centralities[1] - centralities[3]) < 1e-10

    def test_cycle_graph_symmetry(self):
        """サイクルグラフの対称性: 全ての頂点は同じ中心性を持つ"""
        G = nx.cycle_graph(5)

        centralities = all_modified_myerson_centralities(G, walk_length=3)

        # 全ての頂点が同じ中心性を持つ
        values = list(centralities.values())
        for i in range(1, len(values)):
            assert abs(values[0] - values[i]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
