import unittest

import networkx as nx

from centrality import all_myerson_centralities, myerson_centrality
from graph_utils import create_path, create_star, create_triangle


class TestMyersonCentrality(unittest.TestCase):
    def test_triangle_graph(self):
        """三角形グラフ：全頂点対称"""
        G = create_triangle()
        centralities = all_myerson_centralities(G)

        print(f"\n[三角形グラフ] {centralities}")

        # 全て等しい（対称性）
        values = list(centralities.values())
        self.assertAlmostEqual(values[0], values[1], places=5)
        self.assertAlmostEqual(values[1], values[2], places=5)

        # 中心性が0より大きい
        self.assertGreater(values[0], 0)

    def test_path_graph(self):
        """パスグラフ A-B-C：Bが最大"""
        G = create_path(3)
        centralities = all_myerson_centralities(G)

        print(f"\n[パスグラフ] {centralities}")

        # B > A, B > C
        self.assertGreater(centralities["B"], centralities["A"])
        self.assertGreater(centralities["B"], centralities["C"])

        # A = C（対称性）
        self.assertAlmostEqual(centralities["A"], centralities["C"], places=5)

    def test_star_graph(self):
        """スターグラフ：中心が最大"""
        G = create_star(4)
        centralities = all_myerson_centralities(G)

        print(f"\n[スターグラフ] {centralities}")

        # 中心頂点0が最大
        center_value = centralities[0]
        for node in [1, 2, 3, 4]:
            self.assertGreater(center_value, centralities[node])

    def test_single_node(self):
        """単一頂点"""
        G = nx.Graph()
        G.add_node("A")

        c = myerson_centrality(G, "A")

        print(f"\n[単一頂点] A: {c}")

        # 単一頂点の中心性は0
        self.assertEqual(c, 0.0)

    def test_two_nodes(self):
        """2頂点 A-B"""
        G = nx.Graph()
        G.add_edge("A", "B")

        centralities = all_myerson_centralities(G)

        print(f"\n[2頂点] {centralities}")

        # 対称なので等しい
        self.assertAlmostEqual(centralities["A"], centralities["B"], places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
