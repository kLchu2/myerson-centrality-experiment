"""
Myerson中心性計算の共通基盤モジュール

各種Myerson中心性計算で共通利用されるクラスと関数を提供：
- PathCountsCache: パス数のキャッシュ管理
- PathCounter: 深さ優先探索によるパス数計算
- CentralityCalculator: パス数から中心性値を計算
"""

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import networkx as nx

# 定数
DEFAULT_CACHE_DIR = "cache"
DEFAULT_INFLUENCE_CONSTANT = 1.0
PROGRESS_REPORT_INTERVAL = 1


# ============================================================
# キャッシュ管理クラス
# ============================================================


class PathCountsCache:
    """B_l(v)行列のキャッシュ管理"""

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR) -> None:
        """
        Args:
            cache_dir: キャッシュディレクトリのパス
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_graph_hash(self, G: nx.Graph) -> str:
        """
        グラフのハッシュ値を計算

        Args:
            G: NetworkXグラフ

        Returns:
            str: グラフのハッシュ値
        """
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        edges = sorted([(min(u, v), max(u, v)) for u, v in G.edges()])

        hash_str = f"{n_nodes}_{n_edges}_{edges}"
        return hashlib.md5(hash_str.encode()).hexdigest()

    def get_cache_path(self, G: nx.Graph) -> Path:
        """キャッシュファイルのパスを取得"""
        graph_hash = self.compute_graph_hash(G)
        return self.cache_dir / f"path_counts_{graph_hash}.json"

    def load(
        self, G: nx.Graph
    ) -> Optional[tuple[DefaultDict[Union[int, str], DefaultDict[int, int]], int, int]]:
        """キャッシュファイルからパス数を読み込む

        Returns:
            tuple: (path_counts, actuaL_max_length, computed_limit) or None
        """
        filename = self.get_cache_path(G)

        if not filename.exists():
            return None

        try:
            with open(filename, "r") as f:
                data = json.load(f)

            path_counts = defaultdict(lambda: defaultdict(int))
            for node_str, lengths in data["path_counts"].items():
                # 頂点キーの復元: 整数 or タプル or 文字列
                try:
                    node = int(node_str)
                except ValueError:
                    # タプル形式 "(0, 1)" の場合
                    if node_str.startswith("(") and node_str.endswith(")"):
                        try:
                            node = eval(node_str)  # "(0, 1)" -> (0, 1)
                        except Exception:
                            node = node_str
                    else:
                        node = node_str

                for length_str, count in lengths.items():
                    length = int(length_str)
                    path_counts[node][length] = count

            actuaL_max_length = data.get("actuaL_max_length")
            if actuaL_max_length is None:
                # 古いキャッシュファイルの場合は計算
                actuaL_max_length = (
                    max(
                        max(lengths.keys())
                        for lengths in path_counts.values()
                        if lengths
                    )
                    if path_counts
                    else 0
                )

            # computed_limitの取得（後方互換性のため、ない場合はactuaL_max_lengthと同じと仮定）
            computed_limit = data.get("computed_limit", actuaL_max_length)

            print(
                f"[キャッシュ] 読み込み: {data['graph_info']['n_nodes']}頂点, パス長1〜{actuaL_max_length} (計算上限: {computed_limit})"
            )
            return path_counts, actuaL_max_length, computed_limit

        except Exception as e:
            print(f"[キャッシュ] 読み込みに失敗: {e}")
            return None

    def save(
        self,
        G: nx.Graph,
        path_counts: DefaultDict,
        actuaL_max_length: int,
        computed_limit: int,
    ) -> str:
        """パス数をキャッシュファイルに保存

        Args:
            G: グラフ
            path_counts: パス数
            actuaL_max_length: 実際の最大パス長
            computed_limit: 計算時に指定した最大パス長（探索範囲）
        """
        filename = self.get_cache_path(G)

        serializable_data = {
            "graph_info": {
                "n_nodes": G.number_of_nodes(),
                "n_edges": G.number_of_edges(),
                "hash": self.compute_graph_hash(G),
            },
            "actuaL_max_length": actuaL_max_length,
            "computed_limit": computed_limit,
            "path_counts": {
                str(node): {str(length): count for length, count in lengths.items()}
                for node, lengths in path_counts.items()
            },
        }

        with open(filename, "w") as f:
            json.dump(serializable_data, f, indent=2)

        print(
            f"[キャッシュ] 保存: {G.number_of_nodes()}頂点, パス長1〜{actuaL_max_length} (計算上限: {computed_limit})"
        )
        return str(filename)


# ============================================================
# ヘルパー関数（並列処理用）
# ============================================================


def _process_chunk(
    G: nx.Graph, nodes: List[Union[int, str]], max_length: int
) -> Dict[Union[int, str], Dict[int, int]]:
    """
    並列処理用のワーカー関数：指定されたノード群を開始点としてDFSを実行
    """
    # ローカルな集計用辞書（defaultdictだとpicklingでエラーになることがあるので普通のdictに変換して返す）
    local_counts: DefaultDict[Union[int, str], DefaultDict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    # 再帰関数を内部定義（クロージャ）
    def _recursive_dfs(
        current_node: Union[int, str],
        path_length: int,
        path: List[Union[int, str]],
        visited: set,
    ):
        visited.add(current_node)

        # 長さ1以上のパスをカウント
        if 0 < path_length <= max_length:
            for node in path:
                local_counts[node][path_length] += 1

        # max_lengthに達していなければ再帰
        if path_length < max_length:
            for neighbor in G.neighbors(current_node):
                if neighbor not in visited:
                    path.append(neighbor)
                    _recursive_dfs(neighbor, path_length + 1, path, visited)
                    path.pop()

        visited.remove(current_node)

    # 指定された各ノードを開始点としてDFS
    for start_node in nodes:
        # visitedはパス探索ごとにリセット（setを使用すると高速）
        _recursive_dfs(start_node, 0, [start_node], set())

    # defaultdictを通常のdictに変換して返却（Pickle化の安全性のため）
    return {k: dict(v) for k, v in local_counts.items()}


# ============================================================
# パス計算クラス
# ============================================================


class PathCounter:
    """深さ優先探索によるグラフのパス数計算（並列処理対応）"""

    def __init__(self, G: nx.Graph) -> None:
        self.G = G

    def compute_all_path_counts(
        self,
        max_length: Optional[int] = None,
        verbose: bool = False,
        n_workers: Optional[int] = None,
    ) -> Tuple[DefaultDict[Union[int, str], DefaultDict[int, int]], int]:
        """
        全頂点・全長さでのパス数B_l(v)を計算（並列処理）

        Args:
            max_length: 最大パス長（Noneの場合は頂点数-1）
            verbose: 進捗表示
            n_workers: 並列数（Noneの場合はCPUコア数）

        Returns:
            tuple: (path_counts, actuaL_max_length)
        """
        if max_length is None:
            max_length = len(self.G.nodes()) - 1

        if n_workers is None:
            n_workers = os.cpu_count() or 1

        nodes = list(self.G.nodes())
        total_nodes = len(nodes)

        # ワーカー数が1以下の場合は並列化しない
        if n_workers <= 1:
            if verbose:
                print(f"シングルプロセスでDFSを実行中 (パス長1〜{max_length})...")

            # _process_chunkを直接呼び出す（結果はdictのdict）
            raw_counts = _process_chunk(self.G, nodes, max_length)

            # DefaultDict形式に変換
            path_counts = defaultdict(lambda: defaultdict(int))
            for node, lengths in raw_counts.items():
                for length, count in lengths.items():
                    path_counts[node][length] = count

        else:
            if verbose:
                print(
                    f"並列プロセス({n_workers})でDFSを実行中 (パス長1〜{max_length})..."
                )

            # ノードリストをチャンクに分割
            chunk_size = (total_nodes + n_workers - 1) // n_workers
            if chunk_size < 1:
                chunk_size = 1
            chunks = [
                nodes[i : i + chunk_size] for i in range(0, total_nodes, chunk_size)
            ]

            path_counts = defaultdict(lambda: defaultdict(int))
            start_time = time.time()

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # タスクのサブミット
                futures = [
                    executor.submit(_process_chunk, self.G, chunk, max_length)
                    for chunk in chunks
                ]

                # 結果の集計
                for i, future in enumerate(as_completed(futures)):
                    try:
                        chunk_result = future.result()

                        # 集計
                        for node, lengths in chunk_result.items():
                            for length, count in lengths.items():
                                path_counts[node][length] += count

                        if verbose:
                            # 進捗表示（チャンク単位なので概算）
                            # 今回完了したチャンクのノード数を加算
                            # 正確には futuresリストとchunksの対応が必要だが、
                            # 概算として均等割り当てとみなすか、単純にタスク数で表示
                            completed_tasks = i + 1
                            total_tasks = len(futures)
                            elapsed = time.time() - start_time
                            sys.stdout.write(
                                f"\r  進捗: {completed_tasks}/{total_tasks} タスク完了 - 経過: {elapsed:.1f}s"
                            )
                            sys.stdout.flush()

                    except Exception as e:
                        print(f"\n[エラー] ワーカープロセスで例外が発生: {e}")
                        raise e

        # 無向グラフでは各パスを両方向から数えるため2で割る
        self._normalize_undirected_counts(path_counts)

        # 実際に計算された最大パス長を取得
        actuaL_max_length = (
            max(max(lengths.keys()) for lengths in path_counts.values() if lengths)
            if path_counts
            else 0
        )

        if verbose:
            print()
            print("計算完了")

        return path_counts, actuaL_max_length

    def _normalize_undirected_counts(
        self, path_counts: DefaultDict[Union[int, str], DefaultDict[int, int]]
    ) -> None:
        """無向グラフのパスカウントを正規化（2で割る）"""
        for node in path_counts:
            for length in path_counts[node]:
                path_counts[node][length] //= 2


# ============================================================
# 中心性計算クラス
# ============================================================


class CentralityCalculator:
    """パス数から中心性を計算"""

    @staticmethod
    def calculate_centrality(
        path_counts: DefaultDict[Union[int, str], DefaultDict[int, int]],
        node: Union[int, str],
        max_length: int,
        r: float,
    ) -> float:
        """
        単一頂点の中心性を計算

        Args:
            path_counts: パス数の辞書 {node: {length: count}}
            node: 対象頂点
            max_length: 最大パス長
            r: 影響定数

        Returns:
            float: 中心性値
        """
        centrality = 0.0
        for length in range(1, max_length + 1):
            B_l = path_counts[node].get(length, 0)
            centrality += (B_l / (length + 1)) * (r**length)
        return centrality


# ============================================================
# ヘルパー関数
# ============================================================


def get_max_cached_length(
    path_counts: DefaultDict[Union[int, str], DefaultDict[int, int]],
) -> int:
    """キャッシュされている最大パス長を取得"""
    if not path_counts:
        return 0
    sample_node = next(iter(path_counts.keys()))
    return max(path_counts[sample_node].keys()) if path_counts[sample_node] else 0


def is_cache_sufficient(
    path_counts: DefaultDict[Union[int, str], DefaultDict[int, int]],
    required_length: int,
) -> bool:
    """キャッシュが必要な長さをカバーしているか確認"""
    if not path_counts:
        return False
    max_cached = get_max_cached_length(path_counts)
    return max_cached >= required_length
