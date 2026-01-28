"""
ウォークベース修正版マイヤーソン中心性の計算

定理 2.1 (式 2.8)
長さ l のすべてのウォークにおける頂点 v_i の総出現回数 s_i(l) は:
    s_i(l) = Σ_{j=1}^{n} m_{ij}^{(l)} + Σ_{v=1}^{n} [m_{vi} Σ_{j=1}^{n} m_{ij}^{(l-1)}
             + m_{vi}^{(2)} Σ_{j=1}^{n} m_{ij}^{(l-2)} + ... + m_{vi}^{(l)}]

    第1項: 頂点 i がウォークの終点として出現する回数
    第2項: 頂点 i がウォークの中間点として出現する回数

定義 2.2 (式 2.10)
修正版マイヤーソン中心性 σ_i(l) は:
    σ_i(l) = s_i(l) / (l + 1)

    長さ l のウォークにおける出現回数をパス上の頂点数 l+1 で正規化
"""

from typing import Dict, Union

import networkx as nx
import numpy as np


def compute_walk_count_matrix_power(
    G: nx.Graph, max_length: int
) -> Dict[int, np.ndarray]:
    """
    隣接行列のべき乗を計算し、各長さ l に対する M^(l) を返す

    Args:
        G: NetworkXグラフ
        max_length: 最大ウォーク長

    Returns:
        Dict[int, np.ndarray]: {l: M^(l)} l=1 から max_length まで
    """
    # ノードリストを取得（順序を固定）
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 隣接行列 M を作成
    M = np.zeros((n, n), dtype=np.float64)
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        M[i, j] = 1.0
        M[j, i] = 1.0  # 無向グラフ

    # 各長さについて M^(l) を計算
    matrix_powers = {}
    M_power = np.eye(n)  # M^0 = I

    for length in range(1, max_length + 1):
        M_power = M_power @ M  # M^(length) = M^(length-1) @ M
        matrix_powers[length] = M_power.copy()

    return matrix_powers, nodes, node_to_idx, M


def compute_walk_appearance_count(
    matrix_powers: Dict[int, np.ndarray],
    M: np.ndarray,
    n: int,
    node_idx: int,
    walk_length: int,
) -> float:
    """
    定理 2.1 (式 2.8) に基づき、長さ l のウォークにおける頂点 i の総出現回数を計算

    s_i(l) = Σ_{j=1}^{n} m_{ij}^{(l)} + Σ_{v=1}^{n} [m_{vi} Σ_{j=1}^{n} m_{ij}^{(l-1)}
             + m_{vi}^{(2)} Σ_{j=1}^{n} m_{ij}^{(l-2)} + ... + m_{vi}^{(l)}]

    Args:
        matrix_powers: {l: M^(l)} の辞書
        M: 隣接行列
        n: 頂点数
        node_idx: 対象頂点のインデックス i
        walk_length: ウォーク長 l

    Returns:
        float: s_i(l) - 総出現回数
    """
    i = node_idx
    wlen = walk_length

    # 第1項: 頂点 i がウォークの終点として出現する回数
    # Σ_{j=1}^{n} m_{ij}^{(wlen)} = M^(wlen) の i 行の合計
    term1 = np.sum(matrix_powers[wlen][i, :])

    # 第2項: 頂点 i がウォークの中間点として出現する回数
    # Σ_{v=1}^{n} [m_{vi} Σ_{j=1}^{n} m_{ij}^{(wlen-1)} + m_{vi}^{(2)} Σ_{j=1}^{n} m_{ij}^{(wlen-2)} + ... + m_{vi}^{(wlen)}]
    term2 = 0.0

    for k in range(1, wlen + 1):  # k = 1, 2, ..., wlen
        # m_{vi}^{(k)} は M^(k) の全ての v から i への値
        # つまり M^(k) の i 列の合計
        sum_m_vi_k = np.sum(matrix_powers[k][:, i])

        # Σ_{j=1}^{n} m_{ij}^{(wlen-k)} = M^(wlen-k) の i 行の合計
        remaining_length = wlen - k
        if remaining_length == 0:
            # M^(0) = I なので、i 行の合計は 1（自分自身のみ）
            sum_m_ij_remaining = 1.0
        else:
            sum_m_ij_remaining = np.sum(matrix_powers[remaining_length][i, :])

        term2 += sum_m_vi_k * sum_m_ij_remaining

    return term1 + term2


def modified_myerson_centrality(
    G: nx.Graph,
    v: Union[int, str],
    walk_length: int,
    normalize: bool = True,
) -> float:
    """
    単一頂点の修正版マイヤーソン中心性を計算（ウォークベース）

    定義 2.2 (式 2.10): σ_i(l) = s_i(l) / (l + 1)

    Args:
        G: NetworkXグラフ
        v: 対象頂点
        walk_length: ウォーク長 l
        normalize: True の場合 l+1 で正規化（修正版）、False の場合 s_i(l) のまま

    Returns:
        float: 修正版マイヤーソン中心性 σ_i(l) または出現回数 s_i(l)
    """
    if v not in G.nodes():
        return 0.0

    # 非連結グラフの場合は連結成分のみで計算
    if not nx.is_connected(G):
        cc = nx.node_connected_component(G, v)
        G = G.subgraph(cc)

    matrix_powers, nodes, node_to_idx, M = compute_walk_count_matrix_power(
        G, walk_length
    )
    node_idx = node_to_idx[v]
    n = len(nodes)

    s_i_l = compute_walk_appearance_count(matrix_powers, M, n, node_idx, walk_length)

    if normalize:
        return s_i_l / (walk_length + 1)
    else:
        return s_i_l


def all_modified_myerson_centralities(
    G: nx.Graph,
    walk_length: int,
    normalize: bool = True,
    verbose: bool = False,
) -> Dict[Union[int, str], float]:
    """
    全頂点の修正版マイヤーソン中心性を一括計算（ウォークベース）

    Args:
        G: NetworkXグラフ
        walk_length: ウォーク長 l
        normalize: True の場合 l+1 で正規化（修正版）、False の場合 s_i(l) のまま
        verbose: 進捗表示

    Returns:
        dict: {node: centrality}
    """
    if verbose:
        print(f"[ウォークベース] 長さ {walk_length} のウォーク出現回数を計算中...")

    matrix_powers, nodes, node_to_idx, M = compute_walk_count_matrix_power(
        G, walk_length
    )
    n = len(nodes)

    result = {}
    for node in nodes:
        node_idx = node_to_idx[node]
        s_i_l = compute_walk_appearance_count(
            matrix_powers, M, n, node_idx, walk_length
        )

        if normalize:
            result[node] = s_i_l / (walk_length + 1)
        else:
            result[node] = s_i_l

    if verbose:
        print("[ウォークベース] 計算完了")

    return result


def all_modified_myerson_centralities_cumulative(
    G: nx.Graph,
    max_walk_length: int,
    r: float = 1.0,
    normalize_each: bool = True,
    verbose: bool = False,
) -> Dict[Union[int, str], float]:
    """
    累積修正版マイヤーソン中心性を計算（ウォークベース）

    Σ_{l=1}^{L_max} σ_i(l) * r^l

    Args:
        G: NetworkXグラフ
        max_walk_length: 最大ウォーク長 L_max
        r: 影響定数（減衰係数）
        normalize_each: True の場合各長さで l+1 で正規化
        verbose: 進捗表示

    Returns:
        dict: {node: cumulative_centrality}
    """
    if verbose:
        print(f"[累積ウォークベース] 長さ 1〜{max_walk_length} の累積中心性を計算中...")

    matrix_powers, nodes, node_to_idx, M = compute_walk_count_matrix_power(
        G, max_walk_length
    )
    n = len(nodes)

    result = {node: 0.0 for node in nodes}

    for wlen in range(1, max_walk_length + 1):
        for node in nodes:
            node_idx = node_to_idx[node]
            s_i_l = compute_walk_appearance_count(matrix_powers, M, n, node_idx, wlen)

            if normalize_each:
                contribution = (s_i_l / (wlen + 1)) * (r**wlen)
            else:
                contribution = s_i_l * (r**wlen)

            result[node] += contribution

        if verbose and wlen % 5 == 0:
            print(f"  進捗: 長さ {wlen}/{max_walk_length} 完了")

    if verbose:
        print("[累積ウォークベース] 計算完了")

    return result
