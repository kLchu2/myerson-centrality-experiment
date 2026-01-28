import unittest

import networkx as nx

from centrality import (
    all_myerson_centralities,
    all_revised_myerson_centralities,
    revised_myerson_centrality,
)
from graph_utils import create_path, create_star, create_triangle


class TestGeneralizedMyersonCentrality(unittest.TestCase):
    def test_triangle_graph(self):
        """三角形グラフ：全頂点対称"""
        G = create_triangle()
        centralities = all_revised_myerson_centralities(G)

        print(f"\n[改訂版: 三角形グラフ] {centralities}")

        # 全て等しい（対称性）
        values = list(centralities.values())
        self.assertAlmostEqual(values[0], values[1], places=5)
        self.assertAlmostEqual(values[1], values[2], places=5)

        # 中心性が0より大きい
        self.assertGreater(values[0], 0)

    def test_path_graph(self):
        """パスグラフ A-B-C：Bが最大"""
        G = create_path(3)
        centralities = all_revised_myerson_centralities(G)

        print(f"\n[改訂版: パスグラフ] {centralities}")

        # B > A, B > C
        self.assertGreater(centralities["B"], centralities["A"])
        self.assertGreater(centralities["B"], centralities["C"])

        # A = C（対称性）
        self.assertAlmostEqual(centralities["A"], centralities["C"], places=5)

    def test_star_graph(self):
        """スターグラフ：中心が最大"""
        G = create_star(4)
        centralities = all_revised_myerson_centralities(G)

        print(f"\n[改訂版: スターグラフ] {centralities}")

        # 中心頂点0が最大
        center_value = centralities[0]
        for node in [1, 2, 3, 4]:
            self.assertGreater(center_value, centralities[node])

    def test_comparison_with_myerson(self):
        """通常のMyerson中心性との比較"""
        G = create_path(4)  # A-B-C-D

        myerson_vals = all_myerson_centralities(G)
        generalized_vals = all_revised_myerson_centralities(G)

        print(f"\n[通常Myerson] {myerson_vals}")
        print(f"[改訂版Myerson] {generalized_vals}")

        # 改訂版は全パスを数えるため、通常より大きいか等しい
        for node in G.nodes():
            self.assertGreaterEqual(generalized_vals[node], myerson_vals[node])

    def test_single_node(self):
        """単一頂点"""
        G = nx.Graph()
        G.add_node("A")

        c = revised_myerson_centrality(G, "A")

        print(f"\n[改訂版: 単一頂点] A: {c}")

        # 単一頂点の中心性は0
        self.assertEqual(c, 0.0)

    def test_square_graph(self):
        """正方形グラフ：全頂点対称、通常と改訂版で差が出る"""
        G = nx.cycle_graph(4)
        G = nx.relabel_nodes(G, {0: "A", 1: "B", 2: "C", 3: "D"})

        myerson_vals = all_myerson_centralities(G)
        generalized_vals = all_revised_myerson_centralities(G)

        print(f"\n[正方形: 通常Myerson] {myerson_vals}")
        print(f"[正方形: 改訂版Myerson] {generalized_vals}")

        # 正方形では対角線に複数パスがあるため、改訂版の方が大きい
        for node in G.nodes():
            self.assertGreater(generalized_vals[node], myerson_vals[node])


if __name__ == "__main__":
    unittest.main(verbosity=2)
