"""
パス長制限Myerson中心性の計算（深さ優先探索版）

定義: Y^{L_{max}}_G(v) = Σ(l=1 to L_{max}) B_l(v)/(l+1) * r^l
- B_l(v): 頂点vを含む長さlの全パスの本数（最短パスに限定しない）
- L_{max}: パス長制限（ユーザー指定、コード内ではL_maxとして実装）
- 深さ優先探索により全てのパスを効率的に数える
"""

from typing import DefaultDict, Dict, Optional, Union

import networkx as nx

from .base import (
    DEFAULT_CACHE_DIR,
    DEFAULT_INFLUENCE_CONSTANT,
    CentralityCalculator,
    PathCounter,
    PathCountsCache,
)

# ============================================================
# 公開API
# ============================================================


def path_limited_myerson_centrality(
    G: nx.Graph,
    v: Union[int, str],
    L_max: Optional[int],
    r: float = DEFAULT_INFLUENCE_CONSTANT,
) -> float:
    """
    単一頂点のパス長制限Myerson中心性を計算

    Args:
        G: NetworkXグラフ
        v: 対象頂点
        L_max: パス長制限
        r: 影響定数（デフォルト1.0）

    Returns:
        float: パス長制限Myerson中心性

    Raises:
        ValueError: L_maxが指定されていない場合
    """
    if L_max is None:
        raise ValueError("L_max を指定してください")

    if v not in G.nodes():
        return 0.0

    # 非連結グラフの場合は連結成分のみで計算
    if not nx.is_connected(G):
        cc = nx.node_connected_component(G, v)
        G = G.subgraph(cc)

    max_length = min(L_max, len(G.nodes()) - 1)

    counter = PathCounter(G)
    path_counts, _ = counter.compute_all_path_counts(max_length, verbose=False)

    return CentralityCalculator.calculate_centrality(path_counts, v, max_length, r)


def all_path_limited_myerson_centralities(
    G: nx.Graph,
    L_max: int,
    r: float = DEFAULT_INFLUENCE_CONSTANT,
    verbose: bool = False,
    normalize: bool = False,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Dict[Union[int, str], float]:
    """
    全頂点のパス長制限Myerson中心性を一括計算（キャッシュ対応）

    Args:
        G: NetworkXグラフ
        L_max: パス長制限
        r: 影響定数（デフォルト1.0）
        verbose: 進捗表示
        normalize: 合計値で正規化
        use_cache: キャッシュを利用
        cache_dir: キャッシュディレクトリ

    Returns:
        dict: {node: centrality}
    """
    max_length = min(L_max, len(G.nodes()) - 1)

    # キャッシュ処理
    path_counts = _load_or_compute_path_counts(
        G, max_length, use_cache, cache_dir, verbose
    )

    # 中心性を計算
    result = {
        v: CentralityCalculator.calculate_centrality(path_counts, v, max_length, r)
        for v in G.nodes()
    }

    if normalize:
        total = sum(result.values())
        if total > 0:
            result = {node: cent / total for node, cent in result.items()}

    return result


def _load_or_compute_path_counts(
    G: nx.Graph, max_length: int, use_cache: bool, cache_dir: str, verbose: bool
) -> DefaultDict[Union[int, str], DefaultDict[int, int]]:
    path_counts = None
    cached_actuaL_max_length = None
    cached_computed_limit = None

    print(f"[パス計算] 要求: {G.number_of_nodes()}頂点, パス長1〜{max_length}")

    if use_cache:
        cache = PathCountsCache(cache_dir)
        print("[キャッシュ] 確認中...")
        cache_result = cache.load(G)

        if cache_result:
            path_counts, cached_actuaL_max_length, cached_computed_limit = cache_result

            # キャッシュが利用可能か判定
            # 1. 探索範囲が要求以上 (cached_computed_limit >= max_length)
            # 2. 実際の最大パス長が要求以上 (cached_actuaL_max_length >= max_length)
            if (
                cached_computed_limit is not None
                and cached_computed_limit >= max_length
            ) or (cached_actuaL_max_length >= max_length):
                print(
                    f"[キャッシュ] ✓ 利用可能 (探索範囲: {cached_computed_limit}, 実際の最大パス長: {cached_actuaL_max_length} >= 要求: {max_length})"
                )
                return path_counts
            else:
                print(
                    f"[キャッシュ] ✗ 不十分 (探索範囲: {cached_computed_limit}, 実際の最大パス長: {cached_actuaL_max_length} < 要求: {max_length})"
                )
        else:
            print("[キャッシュ] キャッシュファイルが見つかりません")

    # 新規計算または追加計算
    print(f"[計算開始] DFSでパス数を計算中 (要求: 長さ1〜{max_length})...")

    counter = PathCounter(G)
    path_counts, actuaL_max_length = counter.compute_all_path_counts(
        max_length, verbose
    )

    if actuaL_max_length < max_length:
        print(
            f"[計算完了] 実際の最大パス長: {actuaL_max_length} (グラフ構造上、これ以上のパスは存在しません)"
        )
    else:
        print("[計算完了] パス数の計算が完了しました")

    if use_cache:
        print("[キャッシュ] 保存中...")
        cache = PathCountsCache(cache_dir)
        cache.save(G, path_counts, actuaL_max_length, computed_limit=max_length)

    return path_counts
