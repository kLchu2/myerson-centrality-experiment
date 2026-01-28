"""
Myerson中心性の計算（多項式時間アルゴリズム）

このモジュールは、Myerson中心性を多項式時間 O(VE) で計算するアルゴリズムを提供します。
Brandesのアルゴリズムを拡張し、各最短パスに対して距離に応じた重み
f(s, t) = r^d(s, t) / (d(s, t) + 1)
を割り当てて累積します。

定義:
Y_G(v) = Σ_{s≠t} (σ_st(v) / σ_st) * (r^d(s, t) / (d(s, t) + 1))
- σ_st: 頂点s, t間の最短パスの総数
- σ_st(v): そのうち頂点vを通る最短パスの数
- d(s, t): 頂点s, t間の最短距離
"""

from collections import deque

import networkx as nx


def all_myerson_centralities(G, r=1.0, verbose=False, normalize=False):
    """
    全頂点のMyerson中心性を一括計算 (O(VE))

    Args:
        G: NetworkXグラフ
        r: 影響定数（単一値または数値のリスト）
        verbose: 進捗表示
        normalize: 合計値で正規化するかどうか

    Returns:
        - rが単一値の場合: dict {node: centrality_value}
        - rがリストの場合: dict {r_value: {node: centrality_value}}
    """
    if verbose:
        print(
            f"Myerson中心性を計算中... (頂点数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()})"
        )

    r_list = r if isinstance(r, list) else [r]
    nodes = list(G.nodes())

    # 非連結グラフの場合は連結成分ごとに計算
    if not nx.is_connected(G):
        all_results = {curr_r: {} for curr_r in r_list}
        for cc in nx.connected_components(G):
            G_sub = G.subgraph(cc)
            sub_res = all_myerson_centralities(
                G_sub, r_list, verbose=False, normalize=False
            )
            if isinstance(r, list):
                for curr_r in r_list:
                    all_results[curr_r].update(sub_res[curr_r])
            else:
                all_results[r].update(sub_res)

        # 最終的な結果の整形（正規化含む）
        for curr_r in r_list:
            if normalize:
                total = sum(all_results[curr_r].values())
                if total > 0:
                    all_results[curr_r] = {
                        v: val / total for v, val in all_results[curr_r].items()
                    }

        return all_results if isinstance(r, list) else all_results[r]

    # メインの計算処理
    # 結果格納用（rごとに保持）
    results = {curr_r: {v: 0.0 for v in nodes} for curr_r in r_list}

    for s in nodes:
        # 1. BFSにより最短パス数(sigma)と距離(dist)を算出
        S = []  # 最短パスDAGの頂点を距離順に格納するスタック
        P = {v: [] for v in nodes}  # 前駆頂点のリスト
        sigma = {v: 0.0 for v in nodes}
        sigma[s] = 1.0
        dist = {v: -1 for v in nodes}
        dist[s] = 0
        queue = deque([s])

        while queue:
            v = queue.popleft()
            S.append(v)
            for neighbor in G.neighbors(v):
                # 未訪問ノード（最短距離の確定）
                if dist[neighbor] < 0:
                    dist[neighbor] = dist[v] + 1
                    queue.append(neighbor)
                # 最短パス上の隣接ノード
                if dist[neighbor] == dist[v] + 1:
                    sigma[neighbor] += sigma[v]
                    P[neighbor].append(v)

        # 2. 距離の遠い順（スタックから逆順に抽出）に貢献度を累積
        #   Brandes accumulation step with Myerson weight f(s, t)
        for curr_r in r_list:
            delta = {v: 0.0 for v in nodes}
            # Sを逆順に走査（葉から根へ）
            for w in reversed(S):
                # node w が最短パスの「終点(t)」である場合の重み f(s, w) = r^d / (d+1)
                # s != t のパスのみを対象とするため、w == s の場合は重み 0 とする
                if w != s:
                    weight = (curr_r ** dist[w]) / (dist[w] + 1.0)
                else:
                    weight = 0.0

                # current_total は、wを通る（またはwで終わる）全最短パスの寄与の合計
                # δ_s(w) = Σ_{t} (σ_st(w)/σ_st) * f(s, t)
                current_total = weight + delta[w]

                # 前駆（親）ノードへ配分
                for v in P[w]:
                    # σ_sv / σ_sw の割合で分配
                    delta[v] += (sigma[v] / sigma[w]) * current_total

                # w の中心性スコアにソースsからの寄与を加算
                results[curr_r][w] += current_total

    # 中心性値を整形して返す
    for curr_r in r_list:
        # 非正規化時は、ペア{s, t}を(s,t)と(t,s)で2回数えているため2で割る
        # 正規化時は合計値で割るため、2倍の定数項は相殺される
        final_vals = {v: results[curr_r][v] / 2.0 for v in nodes}

        if normalize:
            total = sum(final_vals.values())
            if total > 0:
                final_vals = {v: val / total for v, val in final_vals.items()}
            else:
                final_vals = {v: 0.0 for v in nodes}

        results[curr_r] = final_vals

    if verbose:
        print("計算完了")

    if isinstance(r, list):
        return results
    return results[r]


def myerson_centrality(G, v, r=1.0):
    """
    単一頂点のMyerson中心性を計算

    注意: 効率化のため背後で全頂点の計算を行いますが、指定された頂点の値のみを返します。
    """
    if v not in G:
        return 0.0

    # 連結成分に絞って計算（高速化）
    if not nx.is_connected(G):
        cc = nx.node_connected_component(G, v)
        G = G.subgraph(cc)

    res = all_myerson_centralities(G, r=r)
    return res.get(v, 0.0)
