"""
改訂版Myerson中心性の計算（パス長制限版のラッパー）

定義: Y^*_G(v) = Σ(l=1 to n-1) B_l(v)/(l+1) * r^l
- B_l(v): 頂点vを含む長さlの全パスの本数（最短パスに限定しない）
- n-1: 単純パスの理論的上限（n頂点のグラフにおける単純パスの最大長）
- パス長制限版でL_max=n-1とすることで実現
"""

from typing import Dict, Union

import networkx as nx

from .base import DEFAULT_CACHE_DIR, DEFAULT_INFLUENCE_CONSTANT
from .path_limited_myerson import (
    all_path_limited_myerson_centralities,
    path_limited_myerson_centrality,
)

# ============================================================
# 公開API
# ============================================================


def revised_myerson_centrality(
    G: nx.Graph, v: Union[int, str], r: float = DEFAULT_INFLUENCE_CONSTANT
) -> float:
    """
    単一頂点の改訂版Myerson中心性を計算

    Args:
        G: NetworkXグラフ
        v: 対象頂点
        r: 影響定数（デフォルト1.0）

    Returns:
        float: 改訂版Myerson中心性
    """
    L_max = len(G.nodes()) - 1
    return path_limited_myerson_centrality(G, v, L_max, r)


def all_revised_myerson_centralities(
    G: nx.Graph,
    r: float = DEFAULT_INFLUENCE_CONSTANT,
    verbose: bool = False,
    normalize: bool = False,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Dict[Union[int, str], float]:
    """
    全頂点の改訂版Myerson中心性を一括計算（キャッシュ対応）

    Args:
        G: NetworkXグラフ
        r: 影響定数（デフォルト1.0）
        verbose: 進捗表示
        normalize: 合計値で正規化
        use_cache: キャッシュを利用
        cache_dir: キャッシュディレクトリ

    Returns:
        dict: {node: centrality}
    """
    L_max = len(G.nodes()) - 1
    return all_path_limited_myerson_centralities(
        G, L_max, r, verbose, normalize, use_cache, cache_dir
    )
