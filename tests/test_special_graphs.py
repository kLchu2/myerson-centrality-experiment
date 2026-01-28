"""
特殊グラフ構造のテスト

パス長制限による情報損失が顕著なグラフ構造のテスト
"""

import networkx as nx
import pytest

from centrality.myerson import all_myerson_centralities, myerson_centrality
from graph_utils.generator import (
    create_barbell_graph,
    create_comb_graph,
    create_grid_graph,
    create_kite_graph,
    create_lollipop_graph,
    create_path_graph,
)


class TestBarbellGraph:
    """バーベルグラフのテスト"""

    def test_basic_structure(self):
        """基本構造の確認"""
        G = create_barbell_graph(5, 3)
        assert G.number_of_nodes() == 5 + 5 + 3  # 2つのクリーク + 橋
        assert nx.is_connected(G)

    def test_bridge_centrality(self):
        """橋の頂点が適切な中心性を持つことを確認"""
        G = create_barbell_graph(4, 2)
        centralities = all_myerson_centralities(G)

        # 橋の頂点を特定（次数2）
        bridge_nodes = [v for v in G.nodes() if G.degree(v) == 2]
        assert len(bridge_nodes) == 2

        # 橋の中心性が0より大きいことを確認
        for node in bridge_nodes:
            assert centralities[node] > 0


class TestLollipopGraph:
    """ロリポップグラフのテスト"""

    def test_basic_structure(self):
        """基本構造の確認"""
        G = create_lollipop_graph(5, 4)
        assert G.number_of_nodes() == 5 + 4  # クリーク + 尻尾
        assert nx.is_connected(G)

    def test_tail_centrality(self):
        """尻尾の頂点の中心性がクリークより低いことを確認"""
        G = create_lollipop_graph(4, 3)
        centralities = all_myerson_centralities(G)

        # 尻尾の先端（最も次数が低い）
        tail_tip = min(G.nodes(), key=lambda v: G.degree(v))

        # クリークの頂点
        clique_nodes = [v for v in G.nodes() if G.degree(v) > 2]

        # 尻尾の先端はクリークより中心性が低い
        if len(clique_nodes) > 0:
            max_clique_cent = max(centralities[v] for v in clique_nodes)
            assert centralities[tail_tip] < max_clique_cent


class TestKiteGraph:
    """クラックハートの凧グラフのテスト"""

    def test_basic_structure(self):
        """基本構造の確認"""
        G = create_kite_graph()
        assert G.number_of_nodes() == 10  # Krackhardt Kite Graphは10頂点
        assert nx.is_connected(G)

    def test_bridge_node_centrality(self):
        """橋渡し役の頂点の中心性を確認"""
        G = create_kite_graph()
        centralities = all_myerson_centralities(G)

        # Krackhardt Kite Graphでは、頂点7が橋渡し役
        bridge_node = 7
        tail_node = 9  # 最も周辺の頂点

        # 橋渡し役は尻尾の先端より中心性が高い
        assert centralities[bridge_node] > centralities[tail_node]


class TestCombGraph:
    """コームグラフのテスト"""

    def test_basic_structure(self):
        """基本構造の確認"""
        G = create_comb_graph(5, 3)
        # 背骨5頂点 + 各背骨から3頂点の歯 = 5 + 5*3 = 20頂点
        assert G.number_of_nodes() == 5 + 5 * 3
        assert nx.is_connected(G)

    def test_spine_center_centrality(self):
        """背骨の中心頂点が最も高い中心性を持つことを確認"""
        G = create_comb_graph(5, 2)
        centralities = all_myerson_centralities(G)

        # 背骨の中心頂点
        spine_center = 2  # 5頂点の背骨の中央

        # 歯の先端（次数1）
        tooth_tips = [v for v in G.nodes() if G.degree(v) == 1]

        # 背骨の中心は歯の先端より中心性が高い
        for tip in tooth_tips:
            assert centralities[spine_center] > centralities[tip]


class TestGridGraph:
    """格子グラフのテスト"""

    def test_basic_structure(self):
        """基本構造の確認"""
        G = create_grid_graph(4, 5)
        assert G.number_of_nodes() == 4 * 5
        assert nx.is_connected(G)

    def test_center_centrality(self):
        """中心頂点が角頂点より高い中心性を持つことを確認"""
        G = create_grid_graph(5, 5)
        centralities = all_myerson_centralities(G)

        center_node = (2, 2)  # 中央
        corner_node = (0, 0)  # 角

        # 中心は角より中心性が高い
        assert centralities[center_node] > centralities[corner_node]


class TestPathGraph:
    """パスグラフのテスト"""

    def test_basic_structure(self):
        """基本構造の確認"""
        G = create_path_graph(10)
        assert G.number_of_nodes() == 10
        assert nx.is_connected(G)
        assert nx.diameter(G) == 9

    def test_center_centrality(self):
        """中心頂点が端の頂点より高い中心性を持つことを確認"""
        n = 9
        G = create_path_graph(n)
        centralities = all_myerson_centralities(G)

        center_node = n // 2  # 中央
        end_node = 0  # 端

        # 中心は端より中心性が高い
        assert centralities[center_node] > centralities[end_node]

    def test_symmetry(self):
        """両端の頂点の中心性が等しいことを確認"""
        n = 10
        G = create_path_graph(n)
        centralities = all_myerson_centralities(G)

        # 両端の中心性は等しい（対称性）
        assert abs(centralities[0] - centralities[n - 1]) < 1e-6


class TestPathLimitedCentrality:
    """パス長制限付き中心性のテスト"""

    def test_path_limited_reduces_centrality(self):
        """パス長制限により中心性が減少することを確認"""
        from centrality.path_limited_myerson import path_limited_myerson_centrality

        G = create_path_graph(10)
        center = 5

        # 制限なし
        full_cent = myerson_centrality(G, center)

        # 制限あり（L_max = 3）
        limited_cent = path_limited_myerson_centrality(G, center, L_max=3)

        # 制限により中心性が減少
        assert limited_cent < full_cent

    def test_path_limited_information_loss(self):
        """バーベルグラフで橋の中心性が制限により減少することを確認"""
        from centrality.path_limited_myerson import (
            all_path_limited_myerson_centralities,
        )

        G = create_barbell_graph(4, 2)

        # 制限なし
        centralities_full = all_myerson_centralities(G)

        # 制限あり（L_max = 2）
        centralities_limited = all_path_limited_myerson_centralities(
            G, L_max=2, use_cache=False
        )

        # 橋の頂点
        bridge_nodes = [v for v in G.nodes() if G.degree(v) == 2]

        # 橋の中心性が制限により減少
        for node in bridge_nodes:
            assert centralities_limited[node] < centralities_full[node]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
