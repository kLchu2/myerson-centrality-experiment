"""
テスト用グラフ生成
"""

import networkx as nx


def create_triangle():
    """三角形グラフ"""
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
    return G


def create_path(n=3):
    """パスグラフ"""
    G = nx.path_graph(n)
    # 頂点名をA, B, C...に変更
    mapping = {i: chr(65 + i) for i in range(n)}
    return nx.relabel_nodes(G, mapping)


def create_star(n=4):
    """スターグラフ（中心+n個の末端）"""
    return nx.star_graph(n)


def create_tree(n=2, height=3):
    """完全n分木"""
    return nx.balanced_tree(n, height)


def create_cycle(n=5):
    """サイクルグラフ"""
    return nx.cycle_graph(n)


def create_karate_club():
    """空手クラブネットワーク（実データ）"""
    return nx.karate_club_graph()


def create_complete(n=5):
    """完全グラフ"""
    return nx.complete_graph(n)


# ===== ランダムグラフ生成 =====


def create_erdos_renyi(n, p, seed=None):
    """
    Erdős-Rényiランダムグラフ

    Args:
        n: 頂点数
        p: 辺生成確率
        seed: 乱数シード

    Returns:
        nx.Graph: ER-random graph
    """
    return nx.erdos_renyi_graph(n, p, seed=seed)


def create_barabasi_albert(n, m, seed=None):
    """
    Barabási-Albertスケールフリーネットワーク

    Args:
        n: 頂点数
        m: 各頂点が追加されるときに接続する既存頂点数
        seed: 乱数シード

    Returns:
        nx.Graph: BA model graph
    """
    return nx.barabasi_albert_graph(n, m, seed=seed)


def create_watts_strogatz(n, k, p, seed=None):
    """
    Watts-Strogatzスモールワールドネットワーク

    Args:
        n: 頂点数
        k: 各頂点の隣接頂点数
        p: 再配線確率
        seed: 乱数シード

    Returns:
        nx.Graph: WS model graph
    """
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def create_powerlaw_cluster(n, m, p, seed=None):
    """
    べき乗則クラスターグラフ

    Args:
        n: 頂点数
        m: 各頂点が追加されるときに接続する辺数
        p: 三角形形成確率
        seed: 乱数シード

    Returns:
        nx.Graph: Powerlaw cluster graph
    """
    return nx.powerlaw_cluster_graph(n, m, p, seed=seed)


def create_random_regular(n, d, seed=None):
    """
    ランダム正則グラフ

    Args:
        n: 頂点数
        d: 各頂点の次数
        seed: 乱数シード

    Returns:
        nx.Graph: Random regular graph
    """
    return nx.random_regular_graph(d, n, seed=seed)


# ===== パス長制限による情報損失が顕著なグラフ構造 =====


def create_barbell_graph(n, bridge_length):
    """
    バーベルグラフ BG(n, l)

    2つの完全グラフ（クリーク）をパス（橋）で連結した構造。
    橋の頂点は2つのコミュニティを媒介する重要な役割を持つが、
    パス長制限により過小評価される可能性がある。

    Args:
        n: 各完全グラフの頂点数
        bridge_length: 橋となるパスの頂点数

    Returns:
        nx.Graph: Barbell graph
    """
    return nx.barbell_graph(n, bridge_length)


def create_lollipop_graph(n, tail_length):
    """
    ロリポップグラフ LG(n, l)

    完全グラフ（クリーク）にパス（尻尾）が接続された構造。
    尻尾の頂点は周縁部への到達能力を持つが、
    パス長制限により未知領域への到達可能性が正確に反映されない。

    Args:
        n: 完全グラフの頂点数
        tail_length: 尻尾となるパスの頂点数

    Returns:
        nx.Graph: Lollipop graph
    """
    return nx.lollipop_graph(n, tail_length)


def create_kite_graph():
    """
    クラックハートの凧グラフ (Krackhardt Kite Graph)

    10頂点で構成されるソーシャルネットワーク分析の古典的な例。
    異なる中心性（次数、近接、媒介）の違いを説明するためによく用いられる。

    Returns:
        nx.Graph: Krackhardt Kite graph
    """
    # NetworkXの標準関数を使用
    return nx.krackhardt_kite_graph()


def create_comb_graph(n, tooth_length):
    """
    コームグラフ CG(n, l)

    パス（背骨）の各頂点から、パス（歯）が分岐している構造。
    中心と周辺の区別が重要だが、パス長制限により大域的中心性が喪失する。

    Args:
        n: 背骨となるパスの頂点数
        tooth_length: 各歯となるパスの頂点数

    Returns:
        nx.Graph: Comb graph
    """
    G = nx.Graph()

    # 背骨（spine）の作成: 頂点0からn-1
    for i in range(n - 1):
        G.add_edge(i, i + 1)

    # 各背骨頂点から歯（tooth）を追加
    node_id = n
    for i in range(n):
        # 各背骨頂点から長さtooth_lengthのパスを追加
        prev_node = i
        for j in range(tooth_length):
            G.add_edge(prev_node, node_id)
            prev_node = node_id
            node_id += 1

    return G


def create_grid_graph(m, n):
    """
    格子グラフ GG(m, n)

    m × n の格子状に連結されたグラフ。
    大域的な中心性を持つ頂点が存在するが、
    パス長制限により中心と周辺の区別がつかなくなる。

    Args:
        m: 行数
        n: 列数

    Returns:
        nx.Graph: Grid graph
    """
    return nx.grid_2d_graph(m, n)


def create_path_graph(n):
    """
    パスグラフ PG(n)

    n個の頂点が直線状に連結されたグラフ。
    中心頂点が最も高い中心性を持つが、
    パス長制限により端の頂点との距離情報が失われる。

    Args:
        n: 頂点数

    Returns:
        nx.Graph: Path graph
    """
    return nx.path_graph(n)
