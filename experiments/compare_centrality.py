"""
バーベルグラフBG(5,1)の比較結果をJSON形式で出力

実験設定:
  - グラフ: バーベルグラフ BG(5,1) = K5 + Bridge + K5
  - パス長制限: 2~6
  - 影響定数 r: 0.2, 0.5, 0.8
  - 出力形式: JSON
"""

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from centrality.modified_myerson import all_modified_myerson_centralities  # noqa: E402
from centrality.myerson import (  # noqa: E402
    all_myerson_centralities,
)
from centrality.path_limited_myerson import (  # noqa: E402
    all_path_limited_myerson_centralities,
)
from graph_utils.generator import (  # noqa: E402
    create_barabasi_albert,
    create_barbell_graph,
    create_comb_graph,
    create_complete,
    create_grid_graph,
    create_karate_club,
    create_kite_graph,
    create_lollipop_graph,
    create_path_graph,
    create_watts_strogatz,
)

# ============================================================
# 設定
# ============================================================

# ========== グラフ生成設定 ==========
# 使用可能なグラフ:
#   'barbell': バーベルグラフ BG(n, l)
#   'lollipop': ロリポップグラフ LG(n, l)
#   'kite': クラックハートの凧グラフ
#   'comb': コームグラフ CG(n, l)
#   'grid': 格子グラフ GG(m, n)
#   'path': パスグラフ PG(n)
#   'complete': 完全グラフ K_n
#   'karate_club': 空手クラブネットワーク
#   'barabasi_albert': Barabási-Albertスケールフリーネットワーク BA(n, m)
#   'watts_strogatz': Watts-Strogatzスモールワールドネットワーク WS(n, k, p)

GRAPH_TYPE = "comb"  # グラフの種類を指定

# グラフのパラメータを辞書で指定
GRAPH_PARAMS = {
    "n": 7,  # バーベルグラフの場合: クリークサイズ、ロリポップ/コーム: クリーク/背骨サイズ、グリッド: 行数、パス/完全: 頂点数など
    "l": 3,  # バーベルグラフの場合: 橋の長さ、ロリポップ/コーム: パスの長さ、グリッド: 列数など
    "k": None,  # Watts-Strogatz: 各頂点の隣接頂点数
    "p": None,  # Watts-Strogatz: 再配線確率
    "m": None,  # Barabási-Albert: 各頂点が追加されるときに接続する既存頂点数
}

PATH_LENGTH_VALUES = [2, 3, 4, 5, 6]  # パス長
INFLUENCE_CONSTANT_VALUES = [0.2, 0.5, 0.8]  # 影響定数
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "json"


def create_graph_from_config(graph_type: str, params: dict):
    """
    設定に基づいてグラフを生成

    Args:
        graph_type: グラフのタイプ
        params: グラフのパラメータ

    Returns:
        tuple: (グラフ, グラフ情報dict)
    """
    G = None
    graph_info = {
        "type": graph_type,
        "parameters": {},
        "num_nodes": 0,
        "num_edges": 0,
        "special_nodes": {},
    }

    if graph_type == "barbell":
        n = params.get("n", 5)
        length = params.get("l", 1)
        G = create_barbell_graph(n, length)
        cluster_size = n
        cluster1_nodes = list(range(cluster_size))
        cluster2_nodes = list(range(cluster_size + 1, 2 * cluster_size + 1))
        bridge_nodes = [cluster_size, cluster_size]  # 自己ループ対応
        graph_info["notation"] = f"BG({n},{length})"
        graph_info["parameters"] = {"n": n, "l": length}
        graph_info["special_nodes"] = {
            "cluster1": cluster1_nodes,
            "cluster2": cluster2_nodes,
            "bridge": bridge_nodes,
        }

    elif graph_type == "lollipop":
        n = params.get("n", 5)
        length = params.get("l", 3)
        G = create_lollipop_graph(n, length)
        graph_info["notation"] = f"LG({n},{length})"
        graph_info["parameters"] = {"n": n, "l": length}
        graph_info["special_nodes"] = {
            "clique": list(range(n)),
            "tail": list(range(n, n + length)),
        }

    elif graph_type == "kite":
        G = create_kite_graph()
        graph_info["notation"] = "Kite"
        graph_info["parameters"] = {}
        graph_info["special_nodes"] = {}

    elif graph_type == "comb":
        n = params.get("n", 5)
        length = params.get("l", 2)
        G = create_comb_graph(n, length)
        graph_info["notation"] = f"CG({n},{length})"
        graph_info["parameters"] = {"n": n, "l": length}
        graph_info["special_nodes"] = {"spine": list(range(n))}

    elif graph_type == "grid":
        m = params.get("n", 5)
        n = params.get("l", 5)
        G = create_grid_graph(m, n)
        graph_info["notation"] = f"GG({m},{n})"
        graph_info["parameters"] = {"m": m, "n": n}
        graph_info["special_nodes"] = {}

    elif graph_type == "path":
        n = params.get("n", 10)
        G = create_path_graph(n)
        graph_info["notation"] = f"PG({n})"
        graph_info["parameters"] = {"n": n}
        graph_info["special_nodes"] = {
            "endpoints": [0, n - 1],
            "center": [n // 2],
        }

    elif graph_type == "complete":
        n = params.get("n", 5)
        G = create_complete(n)
        graph_info["notation"] = f"K_{n}"
        graph_info["parameters"] = {"n": n}
        graph_info["special_nodes"] = {}

    elif graph_type == "karate_club":
        G = create_karate_club()
        graph_info["notation"] = "Karate Club"
        graph_info["parameters"] = {}
        graph_info["special_nodes"] = {}

    elif graph_type == "barabasi_albert":
        n = params.get("n", 20)
        m = params.get("m", 3)
        G = create_barabasi_albert(n, m, seed=42)
        graph_info["notation"] = f"BA({n},{m})"
        graph_info["parameters"] = {"n": n, "m": m}
        graph_info["special_nodes"] = {}

    elif graph_type == "watts_strogatz":
        n = params.get("n", 20)
        k = params.get("k", 4)
        p = params.get("p", 0.3)
        G = create_watts_strogatz(n, k, p, seed=42)
        graph_info["notation"] = f"WS({n},{k},{p})"
        graph_info["parameters"] = {"n": n, "k": k, "p": p}
        graph_info["special_nodes"] = {}

    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    if G is None:
        raise ValueError(f"Failed to create graph of type: {graph_type}")

    graph_info["num_nodes"] = G.number_of_nodes()
    graph_info["num_edges"] = G.number_of_edges()

    return G, graph_info


def export_results():
    """結果をJSON形式で出力"""

    # グラフの準備
    G, graph_info = create_graph_from_config(GRAPH_TYPE, GRAPH_PARAMS)

    print("中心性比較実験")
    print("=" * 70)
    print("グラフ情報:")
    print(f"  タイプ: {graph_info['notation']}")
    print(f"  ノード数: {graph_info['num_nodes']}")
    print(f"  エッジ数: {graph_info['num_edges']}")
    if graph_info["special_nodes"]:
        for key, value in graph_info["special_nodes"].items():
            print(f"  {key}: {value}")
    print("=" * 70)

    # 媒介中心性を計算（共通）
    print("\n媒介中心性を計算中...")
    betweenness_dict = nx.betweenness_centrality(G, normalized=False)

    # 通常のマイヤーソン中心性を計算（各r値）
    myerson_dicts = {}
    print("通常マイヤーソン中心性を計算中...")
    for r in INFLUENCE_CONSTANT_VALUES:
        print(f"  r={r}")
        myerson_dicts[r] = all_myerson_centralities(G, r=r, verbose=False)

    # 修正版を計算（各L値）
    modified_dicts = {}
    print("修正版マイヤーソン中心性を計算中...")
    for L in PATH_LENGTH_VALUES:
        print(f"  L={L}")
        modified_dicts[L] = all_modified_myerson_centralities(
            G, walk_length=L, normalize=False, verbose=False
        )

    # パス長制限版を計算（各L値で各r値）
    path_limited_dicts = {}
    print("パス長制限版マイヤーソン中心性を計算中...")
    for r in INFLUENCE_CONSTANT_VALUES:
        path_limited_dicts[r] = {}
        for L in PATH_LENGTH_VALUES:
            print(f"  L={L}, r={r}")
            path_limited_dicts[r][L] = all_path_limited_myerson_centralities(
                G, L_max=L, r=r, verbose=False, use_cache=False
            )

    # 結果を格納する辞書
    results = {
        "experiment": "Comprehensive Centrality Comparison",
        "graph": {
            "type": graph_info["type"],
            "notation": graph_info["notation"],
            "num_nodes": graph_info["num_nodes"],
            "num_edges": graph_info["num_edges"],
            "parameters": graph_info["parameters"],
            "special_nodes": graph_info["special_nodes"],
        },
        "methods": [
            {"name": "betweenness", "label": "媒介中心性"},
            {"name": "myerson_r0.2", "label": "通常マイヤーソン (r=0.2)"},
            {"name": "myerson_r0.5", "label": "通常マイヤーソン (r=0.5)"},
            {"name": "myerson_r0.8", "label": "通常マイヤーソン (r=0.8)"},
            {"name": "modified", "label": "修正版マイヤーソン"},
        ]
        + [
            {"name": f"modified_L{L}", "label": f"修正版マイヤーソン (L={L})"}
            for L in PATH_LENGTH_VALUES
        ]
        + [
            {"name": f"path_limited_L{L}_r{r}", "label": f"パス長制限版 (L={L}, r={r})"}
            for r in INFLUENCE_CONSTANT_VALUES
            for L in PATH_LENGTH_VALUES
        ],
        "centrality_data": {},
    }

    # 各中心性指標ごとにデータを集計

    # 媒介中心性
    results["centrality_data"]["betweenness"] = {
        str(node): float(betweenness_dict[node]) for node in sorted(G.nodes())
    }

    # 通常マイヤーソン（r=0.2, 0.5, 0.8）
    for r in INFLUENCE_CONSTANT_VALUES:
        results["centrality_data"][f"myerson_r{r}"] = {
            str(node): float(myerson_dicts[r][node]) for node in sorted(G.nodes())
        }

    # 修正版（各L値）
    for L in PATH_LENGTH_VALUES:
        results["centrality_data"][f"modified_L{L}"] = {
            str(node): float(modified_dicts[L][node]) for node in sorted(G.nodes())
        }

    # パス長制限版（各r値で各L値）
    for r in INFLUENCE_CONSTANT_VALUES:
        for L in PATH_LENGTH_VALUES:
            results["centrality_data"][f"path_limited_L{L}_r{r}"] = {
                str(node): float(path_limited_dicts[r][L][node])
                for node in sorted(G.nodes())
            }

    # 統計情報を追加
    results["statistics"] = {}

    # 媒介中心性の統計
    betweenness_values = np.array(list(betweenness_dict.values()))
    results["statistics"]["betweenness"] = {
        "mean": float(betweenness_values.mean()),
        "std": float(betweenness_values.std()),
        "min": float(betweenness_values.min()),
        "max": float(betweenness_values.max()),
    }

    # 通常マイヤーソンの統計
    for r in INFLUENCE_CONSTANT_VALUES:
        values = np.array(list(myerson_dicts[r].values()))
        results["statistics"][f"myerson_r{r}"] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    # 修正版の統計（各L値）
    for L in PATH_LENGTH_VALUES:
        modified_values = np.array(list(modified_dicts[L].values()))
        results["statistics"][f"modified_L{L}"] = {
            "mean": float(modified_values.mean()),
            "std": float(modified_values.std()),
            "min": float(modified_values.min()),
            "max": float(modified_values.max()),
        }

    # パス長制限版の統計
    for r in INFLUENCE_CONSTANT_VALUES:
        for L in PATH_LENGTH_VALUES:
            values = np.array(list(path_limited_dicts[r][L].values()))
            results["statistics"][f"path_limited_L{L}_r{r}"] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    # ブリッジとクラスタの比率を追加（バーベルグラフの場合のみ）
    results["bridge_cluster_ratios"] = {}

    if GRAPH_TYPE == "barbell":
        cluster1_nodes = graph_info["special_nodes"]["cluster1"]
        cluster2_nodes = graph_info["special_nodes"]["cluster2"]
        bridge_nodes = graph_info["special_nodes"]["bridge"]

        # 媒介中心性
        cluster_avg = np.mean(
            [betweenness_dict[n] for n in cluster1_nodes + cluster2_nodes]
        )
        bridge_avg = np.mean(
            [betweenness_dict[bridge_nodes[0]], betweenness_dict[bridge_nodes[1]]]
        )
        results["bridge_cluster_ratios"]["betweenness"] = (
            float(bridge_avg / cluster_avg) if cluster_avg > 0 else None
        )

        # 通常マイヤーソン
        for r in INFLUENCE_CONSTANT_VALUES:
            cluster_avg = np.mean(
                [myerson_dicts[r][n] for n in cluster1_nodes + cluster2_nodes]
            )
            bridge_avg = np.mean(
                [myerson_dicts[r][bridge_nodes[0]], myerson_dicts[r][bridge_nodes[1]]]
            )
            results["bridge_cluster_ratios"][f"myerson_r{r}"] = (
                float(bridge_avg / cluster_avg) if cluster_avg > 0 else None
            )

        # 修正版（各L値）
        for L in PATH_LENGTH_VALUES:
            cluster_avg = np.mean(
                [modified_dicts[L][n] for n in cluster1_nodes + cluster2_nodes]
            )
            bridge_avg = np.mean(
                [
                    modified_dicts[L][bridge_nodes[0]],
                    modified_dicts[L][bridge_nodes[1]],
                ]
            )
            results["bridge_cluster_ratios"][f"modified_L{L}"] = (
                float(bridge_avg / cluster_avg) if cluster_avg > 0 else None
            )

        # パス長制限版
        for r in INFLUENCE_CONSTANT_VALUES:
            for L in PATH_LENGTH_VALUES:
                cluster_avg = np.mean(
                    [
                        path_limited_dicts[r][L][n]
                        for n in cluster1_nodes + cluster2_nodes
                    ]
                )
                bridge_avg = np.mean(
                    [
                        path_limited_dicts[r][L][bridge_nodes[0]],
                        path_limited_dicts[r][L][bridge_nodes[1]],
                    ]
                )
                results["bridge_cluster_ratios"][f"path_limited_L{L}_r{r}"] = (
                    float(bridge_avg / cluster_avg) if cluster_avg > 0 else None
                )

    # JSON形式で出力
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{GRAPH_TYPE}_graph_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"結果を保存しました: {output_file}")
    print("=" * 70)

    # サマリーを表示
    print("\n【実験サマリー】")
    print(f"グラフ: {graph_info['notation']}")
    print("\n計算した中心性指標:")
    for method in results["methods"]:
        print(f"  - {method['label']}")
    print("\n完了！")


if __name__ == "__main__":
    export_results()
