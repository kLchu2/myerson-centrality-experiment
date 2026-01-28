import unittest

import networkx as nx

from centrality import (
    all_path_limited_myerson_centralities,
    all_revised_myerson_centralities,
)
from graph_utils import create_path, create_star


class TestPathLimitedMyersonCentrality(unittest.TestCase):
    def test_path_graph_L1(self):
        """パスグラフ A-B-C-D：L_max=1（隣接頂点のみ）"""
        G = create_path(4)
        centralities = all_path_limited_myerson_centralities(
            G, L_max=1, use_cache=False
        )

        print(f"\n[パス長制限 L=1: パスグラフ] {centralities}")

        # L=1では隣接頂点のみカウント
        # 両端（A, D）は1つの隣接、中間（B, C）は2つの隣接
        self.assertGreater(centralities["B"], centralities["A"])
        self.assertGreater(centralities["C"], centralities["D"])
        self.assertAlmostEqual(centralities["B"], centralities["C"], places=5)

    def test_path_graph_L2(self):
        """パスグラフ A-B-C-D：L_max=2"""
        G = create_path(4)

        centralities_L1 = all_path_limited_myerson_centralities(
            G, L_max=1, use_cache=False
        )
        centralities_L2 = all_path_limited_myerson_centralities(
            G, L_max=2, use_cache=False
        )

        print(f"\n[パス長制限 L=1] {centralities_L1}")
        print(f"[パス長制限 L=2] {centralities_L2}")

        # L=2の方が大きい（より多くのパスを考慮）
        for node in G.nodes():
            self.assertGreaterEqual(centralities_L2[node], centralities_L1[node])

    def test_comparison_with_generalized(self):
        """改訂版Myerson中心性との比較"""
        G = create_path(5)  # A-B-C-D-E（直径=4）

        # L_max=2で制限
        limited_vals = all_path_limited_myerson_centralities(
            G, L_max=2, use_cache=False
        )
        # 制限なし（改訂版）
        generalized_vals = all_revised_myerson_centralities(G)

        print(f"\n[パス長制限 L=2] {limited_vals}")
        print(f"[改訂版（制限なし）] {generalized_vals}")

        # パス長制限の方が小さいか等しい
        for node in G.nodes():
            self.assertLessEqual(limited_vals[node], generalized_vals[node])

    def test_star_graph_L1(self):
        """スターグラフ：L_max=1（中心から直接接続のみ）"""
        G = create_star(4)
        centralities = all_path_limited_myerson_centralities(
            G, L_max=1, use_cache=False
        )

        print(f"\n[パス長制限 L=1: スターグラフ] {centralities}")

        # L=1では中心頂点が最大
        center_value = centralities[0]
        for node in [1, 2, 3, 4]:
            self.assertGreater(center_value, centralities[node])

    def test_star_graph_L2(self):
        """スターグラフ：L_max=2（2ホップまで）"""
        G = create_star(4)

        centralities_L1 = all_path_limited_myerson_centralities(
            G, L_max=1, use_cache=False
        )
        centralities_L2 = all_path_limited_myerson_centralities(
            G, L_max=2, use_cache=False
        )

        print(f"\n[スターグラフ L=1] {centralities_L1}")
        print(f"[スターグラフ L=2] {centralities_L2}")

        # L=2では周辺頂点間のパス（中心経由）も考慮されるため増加
        for node in G.nodes():
            self.assertGreaterEqual(centralities_L2[node], centralities_L1[node])

    def test_L_max_larger_than_diameter(self):
        """L_maxがグラフの直径より大きい場合"""
        G = create_path(3)  # A-B-C（直径=2）

        # L_max=10だが、直径=2なので実質L_max=2として計算
        centralities_large = all_path_limited_myerson_centralities(
            G, L_max=10, use_cache=False
        )
        # 改訂版（制限なし）と同じになるはず
        centralities_generalized = all_revised_myerson_centralities(G)

        print(f"\n[パス長制限 L=10（直径2）] {centralities_large}")
        print(f"[改訂版] {centralities_generalized}")

        # 同じ値になるはず
        for node in G.nodes():
            self.assertAlmostEqual(
                centralities_large[node], centralities_generalized[node], places=5
            )

    def test_cycle_graph_L1_vs_L2(self):
        """サイクルグラフ：L=1とL=2の比較"""
        G = nx.cycle_graph(6)

        centralities_L1 = all_path_limited_myerson_centralities(
            G, L_max=1, use_cache=False
        )
        centralities_L2 = all_path_limited_myerson_centralities(
            G, L_max=2, use_cache=False
        )

        print(f"\n[サイクル L=1] {centralities_L1}")
        print(f"[サイクル L=2] {centralities_L2}")

        # 全頂点対称
        values_L1 = list(centralities_L1.values())
        values_L2 = list(centralities_L2.values())

        for i in range(len(values_L1) - 1):
            self.assertAlmostEqual(values_L1[i], values_L1[i + 1], places=5)
            self.assertAlmostEqual(values_L2[i], values_L2[i + 1], places=5)

        # L=2の方が大きい
        self.assertGreater(values_L2[0], values_L1[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
