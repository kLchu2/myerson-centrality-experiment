"""
パス長制限による情報損失の分析実験

特殊グラフ構造において、パス長制限がMyerson中心性に与える影響を調査する。
以下のグラフを対象とする：
- バーベルグラフ: コミュニティ間の媒介性の喪失
- クラックハートの凧グラフ: 橋渡し役の過小評価
- ロリポップグラフ: 周縁部への到達可能性の喪失
- コームグラフ: 大域的中心性の喪失
- 格子グラフ: 中心と周辺の区別の喪失
- パスグラフ: 端の頂点との距離情報の喪失
"""

import copy
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import kendalltau

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# プロジェクト内モジュールのimport（sys.path設定後に必要）
from centrality.path_limited_myerson import (  # noqa: E402
    all_path_limited_myerson_centralities,
)
from centrality.revised_myerson import all_revised_myerson_centralities  # noqa: E402
from graph_utils.generator import (  # noqa: E402
    create_barbell_graph,
    create_comb_graph,
    create_grid_graph,
    create_kite_graph,
    create_lollipop_graph,
    create_path_graph,
)

# 日本語フォント設定
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "Hiragino Sans",
    "Yu Gothic",
    "Meirio",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"


# ========== 実験設定 ==========

# ----- 分析するグラフの選択 -----
# 0: バーベルグラフ
# 1: ロリポップグラフ
# 2: クラックハートの凧グラフ
# 3: コームグラフ
# 4: 格子グラフ
# 5: パスグラフ
# 複数選択する場合はリストで指定(例: [0, 1, 2])
# 全て選択する場合は 'all' または None
SELECT_GRAPHS = [0, 1, 3, 4, 5]  # デフォルトは全グラフ

# 日本語名のマッピング
GRAPH_NAME_JP = {
    "barbell": "バーベルグラフ",
    "lollipop": "ロリポップグラフ",
    "kite": "クラックハートの凧グラフ",
    "comb": "コームグラフ",
    "grid": "格子グラフ",
    "path": "パスグラフ",
}

# ----- 各グラフのパラメータ -----
# 各パラメータは辞書、または辞書のリストとして指定可能
# リストの場合、それぞれのパラメータ設定で順に実験が行われる
GRAPH_PARAMS = {
    "barbell": [{"n": 5, "l": 5}, {"n": 9, "l": 10}],
    "lollipop": [{"n": 5, "l": 10}, {"n": 8, "l": 20}],
    "kite": {},
    "comb": [{"n": 5, "l": 5}, {"n": 10, "l": 10}],
    "grid": [{"m": 5, "n": 5}, {"m": 6, "n": 7}],
    "path": [{"n": 20}, {"n": 50}],
}

# ----- 影響定数 -----
# 長いパスの影響を調整（小さいほど短いパスを重視）
# 推奨値: 0.2, 0.5, 0.8
R_VALUE = [0.2, 0.5, 0.8]

# ----- パス長制限値（L_max）の設定 -----
# 例: [1, 2, 3, 5, 8] → これらのL_max値で中心性を計算
L_max_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# L_max_VALUES = [3, 6, 9, 12, 15, 18, 21]

# ----- 改訂版Myerson中心性の計算 -----
# True: 改訂版を計算し、それを100%として相対値を表示
# False: 改訂版を計算せず、パス長制限版の絶対値を表示
COMPUTE_FULL_CENTRALITY = True

# ----- 出力 -----
GENERATE_PLOTS = True  # グラフを生成するか
SAVE_PLOTS = True  # グラフを保存するか
PLOT_OUTPUT_DIR = "figures"  # グラフの出力ディレクトリ
JSON_OUTPUT_DIR = "json"  # JSONの出力ディレクトリ

# ----- プロット設定 -----
MAX_PLOT_LINES = 10  # プロットする線の最大数（1〜11、11=全頂点の10分位）
LINE_ALPHA = 1.0  # 線の透明度（0.0〜1.0、1.0=完全不透明）
USE_LINE_STYLES = True  # 線のスタイルを変えるか（実線、破線など）

# ========================================


# グラフタイプのマッピング
GRAPH_TYPE_MAP = {
    0: "barbell",
    1: "lollipop",
    2: "kite",
    3: "comb",
    4: "grid",
    5: "path",
}


def get_selected_graphs() -> Dict[str, bool]:
    """
    SELECT_GRAPHSの設定に基づいて分析対象のグラフを取得

    Returns:
        dict: ANALYZE_GRAPHS形式の辞書
    """
    if SELECT_GRAPHS == "all" or SELECT_GRAPHS is None:
        return {key: True for key in GRAPH_TYPE_MAP.values()}

    if isinstance(SELECT_GRAPHS, (list, tuple)):
        selected = {key: False for key in GRAPH_TYPE_MAP.values()}
        for idx in SELECT_GRAPHS:
            if idx in GRAPH_TYPE_MAP:
                selected[GRAPH_TYPE_MAP[idx]] = True
            else:
                print(f"警告: 無効なグラフインデックス {idx} は無視されます")
        return selected

    if isinstance(SELECT_GRAPHS, int):
        if SELECT_GRAPHS in GRAPH_TYPE_MAP:
            selected = {key: False for key in GRAPH_TYPE_MAP.values()}
            selected[GRAPH_TYPE_MAP[SELECT_GRAPHS]] = True
            return selected
        else:
            raise ValueError(f"無効なグラフインデックス: {SELECT_GRAPHS}")

    raise ValueError(f"SELECT_GRAPHSの形式が無効です: {SELECT_GRAPHS}")


def compute_centralities_with_Lmax(
    G: nx.Graph, r: float
) -> Tuple[Optional[Dict], Dict]:
    """
    パス長制限付き中心性を様々なL_max値で計算

    Args:
        G: NetworkXグラフ
        r: 影響定数

    Returns:
        tuple: (centralities_full, centralities_limited)
               centralities_fullはCOMPUTE_FULL_CENTRALITYがFalseの場合None
    """
    # パス長制限付き中心性（様々なL_max）を計算
    centralities_limited = {}
    L_max_range = L_max_VALUES

    # 改訂版Myerson中心性（L_max = n-1）
    if COMPUTE_FULL_CENTRALITY:
        print("  改訂版Myerson中心性の計算中...")
        centralities_full = all_revised_myerson_centralities(G, r=r, verbose=True)
    else:
        centralities_full = None

    # 最大のL_maxから計算（キャッシュ効率化のため降順）
    for L_max in sorted(L_max_range, reverse=True):
        print(f"  パス長制限 L_max={L_max} の中心性を計算中...")
        centralities_limited[L_max] = all_path_limited_myerson_centralities(
            G, L_max=L_max, r=r, verbose=True
        )

    return centralities_full, centralities_limited


def analyze_graph(
    graph_type: str, params: Dict[str, Any], r: float = 0.5
) -> Dict[str, Any]:
    """
    汎用グラフ分析関数

    指定されたグラフタイプとパラメータでグラフを生成し、パス長制限付き中心性を計算する。

    Args:
        graph_type: グラフタイプ ('barbell', 'lollipop', 'kite', 'comb', 'grid', 'path')
        params: グラフのパラメータ辞書
        r: 影響定数

    Returns:
        dict: 分析結果
    """
    # グラフ作成関数のマッピング
    graph_creators = {
        "barbell": lambda: create_barbell_graph(params["n"], params["l"]),
        "lollipop": lambda: create_lollipop_graph(params["n"], params["l"]),
        "kite": lambda: create_kite_graph(),
        "comb": lambda: create_comb_graph(params["n"], params["l"]),
        "grid": lambda: create_grid_graph(params["m"], params["n"]),
        "path": lambda: create_path_graph(params["n"]),
    }

    if graph_type not in graph_creators:
        raise ValueError(f"未対応のグラフタイプ: {graph_type}")

    # グラフ作成
    G = graph_creators[graph_type]()

    # グラフ情報（diameterは可視化用）
    diameter = nx.diameter(G)
    avg_path_length = nx.average_shortest_path_length(G)

    # 中心性を計算
    centralities_full, centralities_limited = compute_centralities_with_Lmax(G, r)

    # ケンドールの順位相関係数を計算
    kendall_tau_results = {}
    if centralities_full is not None:
        # 共通のキー（頂点）を取得してソート
        nodes = sorted(centralities_full.keys())
        full_values = [centralities_full[n] for n in nodes]

        for L_max, limited_cent in centralities_limited.items():
            limited_values = [limited_cent[n] for n in nodes]
            tau, _ = kendalltau(full_values, limited_values)
            kendall_tau_results[L_max] = tau

    # 格子グラフの場合は頂点キーを文字列に変換
    if graph_type == "grid":
        if centralities_full is not None:
            centralities_full = {str(k): v for k, v in centralities_full.items()}
        centralities_limited = {
            L_max: {str(k): v for k, v in cent.items()}
            for L_max, cent in centralities_limited.items()
        }

    result = {
        "graph_type": graph_type,
        "parameters": params,
        "graph_properties": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "diameter": diameter,
            "avg_path_length": avg_path_length,
        },
        "centralities_full": centralities_full,
        "centralities_limited": centralities_limited,
        "kendall_tau": kendall_tau_results,
    }

    return result


def run_all_experiments(
    analyze_graphs: Optional[Dict[str, bool]] = None,
    graph_params: Optional[Dict[str, Dict[str, int]]] = None,
    r: float = 0.5,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    実験を実行

    Args:
        analyze_graphs: 分析するグラフの辞書（Noneの場合はget_selected_graphs()を使用）
        graph_params: グラフパラメータの辞書（Noneの場合はGRAPH_PARAMSを使用）
        r: 影響定数（デフォルト0.5）

    Returns:
        tuple: (results, metadata)
    """
    if analyze_graphs is None:
        analyze_graphs = get_selected_graphs()
    if graph_params is None:
        graph_params = GRAPH_PARAMS

    # 実行するグラフをカウント
    graphs_to_run = [name for name, enabled in analyze_graphs.items() if enabled]
    total = len(graphs_to_run)

    print("=" * 60)
    print("パス長制限による情報損失の分析実験")
    print(f"影響定数 r = {r}")
    print(f"分析対象: {total}個のグラフ")
    print("=" * 60)

    # 結果辞書の初期化（リスト形式で保存）
    results = {name: [] for name in graphs_to_run}

    count = 0
    for graph_type in graphs_to_run:
        count += 1
        jp_name = GRAPH_NAME_JP.get(graph_type, graph_type)
        print(f"\n[{count}/{total}] {jp_name}の分析...")

        # パラメータを取得（辞書またはリスト）
        params_entry = graph_params.get(graph_type, {})

        # リスト形式に統一して処理
        if isinstance(params_entry, list):
            params_list = params_entry
        else:
            params_list = [params_entry]

        # 各パラメータ設定で実行
        for i, params in enumerate(params_list):
            if len(params_list) > 1:
                print(f"  設定 {i + 1}/{len(params_list)}: {params}")

            result = analyze_graph(graph_type, params, r=r)
            results[graph_type].append(result)

    print("\n" + "=" * 60)
    print(f"{count}個のグラフの実験が完了しました")
    print("=" * 60)

    # メタデータを作成
    metadata = {
        "experiment_name": "パス長制限による情報損失の分析",
        "timestamp": datetime.datetime.now().isoformat(),
        "settings": {
            "centrality_method": "パス長制限付きマイヤーソン中心性",
            "influence_constant_r": r,
            "L_max_values": L_max_VALUES,
        },
        "analyzed_graphs": graphs_to_run,
        "graph_parameters": {name: graph_params[name] for name in graphs_to_run},
    }

    return results, metadata


def _get_param_string(graph_type: str, params: Any) -> str:
    """
    グラフタイプとパラメータからパラメータ文字列を生成

    Args:
        graph_type: グラフタイプ名
        params: グラフパラメータ辞書またはリスト

    Returns:
        str: パラメータ文字列
    """
    if isinstance(params, list):
        return "multi"

    if graph_type in ("barbell", "lollipop", "comb"):
        return f"{params['n']}{params['l']}"
    elif graph_type == "grid":
        return f"{params['m']}{params['n']}"
    elif graph_type == "path":
        return f"{params['n']}"
    else:  # kite or unknown
        return ""


def generate_filename(
    analyze_graphs: Dict[str, bool],
    graph_params: Dict[str, Dict[str, int]],
    r: float,
    extension: str = "json",
) -> str:
    """
    分析対象のグラフとパラメータからファイル名を生成

    Args:
        analyze_graphs: 分析するグラフの辞書
        graph_params: グラフパラメータの辞書
        r: 影響定数
        extension: ファイル拡張子（デフォルト'json'）

    Returns:
        str: 生成されたファイル名
    """
    # グラフタイプの略称マッピング
    graph_abbr = {
        "barbell": "bg",
        "lollipop": "lg",
        "kite": "kg",
        "comb": "cg",
        "grid": "gg",
        "path": "pg",
    }

    # 実行されるグラフを取得
    graphs_to_run = [name for name, enabled in analyze_graphs.items() if enabled]

    # 影響定数部分を生成（小数点なしの2桁）
    r_str = f"{int(r * 100):02d}"

    # 単一グラフの場合
    if len(graphs_to_run) == 1:
        graph_type = graphs_to_run[0]
        abbr = graph_abbr.get(graph_type, graph_type[:2])
        param_str = _get_param_string(graph_type, graph_params[graph_type])

        # ファイル名を組み立て
        if param_str:
            return f"info_loss_{abbr}{param_str}_{r_str}.{extension}"
        else:
            return f"info_loss_{abbr}_{r_str}.{extension}"
    else:
        # 複数グラフの場合
        return f"info_loss_multi_{r_str}.{extension}"


def save_results(
    results: Dict[str, Dict[str, Any]],
    filename: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    結果をJSONファイルに保存（正規化版と非正規化版の両方を保存）

    Args:
        results: 実験結果の辞書
        filename: 出力ファイル名
        metadata: 実験設定のメタデータ（Noneの場合は結果のみ保存）
    """
    output_dir = Path(__file__).parent.parent / "results" / JSON_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    # 中心性値を正規化してコピー（deep copyを使用）
    results_normalized = {}
    for graph_type, result_list in results.items():
        results_normalized[graph_type] = []

        for result in result_list:
            result_copy = copy.deepcopy(result)

            # centralities_fullを正規化
            if (
                "centralities_full" in result_copy
                and result_copy["centralities_full"] is not None
            ):
                cent_full = result_copy["centralities_full"]
                total = sum(cent_full.values())
                if total > 0:
                    result_copy["centralities_full"] = {
                        k: v / total for k, v in cent_full.items()
                    }

            # centralities_limitedを正規化（各L_maxごとに）
            if "centralities_limited" in result_copy:
                cent_limited = {}
                for L_max, cent in result_copy["centralities_limited"].items():
                    total = sum(cent.values())
                    if total > 0:
                        cent_limited[L_max] = {k: v / total for k, v in cent.items()}
                    else:
                        cent_limited[L_max] = cent
                result_copy["centralities_limited"] = cent_limited

            results_normalized[graph_type].append(result_copy)

    # メタデータがある場合は、正規化版と非正規化版の両方を構造化して保存
    if metadata:
        output_data = {
            "metadata": metadata,
            "results_normalized": results_normalized,
            "results_raw": results,
        }
    else:
        output_data = {
            "results_normalized": results_normalized,
            "results_raw": results,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n結果を保存しました: {output_path}")

    # ケンドールの順位相関係数のみを抽出して別ファイルに保存
    kendall_results = {}
    for graph_type, result_list in results.items():
        kendall_results[graph_type] = []
        for result in result_list:
            if "kendall_tau" in result and result["kendall_tau"]:
                kendall_entry = {
                    "parameters": result.get("parameters", {}),
                    "kendall_tau": result["kendall_tau"],
                }
                kendall_results[graph_type].append(kendall_entry)

    # データが存在する場合のみ保存
    if any(k_list for k_list in kendall_results.values()):
        kendall_filename = "kendall_" + filename
        output_path_kendall = output_dir / kendall_filename

        kendall_output_data = {
            "metadata": metadata,
            "results": kendall_results,
        }

        with open(output_path_kendall, "w", encoding="utf-8") as f:
            json.dump(kendall_output_data, f, ensure_ascii=False, indent=2)

        print(f"Kendall's Tau results saved: {output_path_kendall}")


def plot_results(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "figures",
    save: bool = True,
    graph_params: Optional[Dict[str, Dict[str, int]]] = None,
    r: float = 0.5,
) -> None:
    """
    実験結果をグラフ化（中心性値のL_max別プロット）

    Args:
        results: 実験結果の辞書
        output_dir: 出力ディレクトリ名
        save: ファイルに保存するか
        graph_params: グラフパラメータの辞書（ファイル名生成用）
        r: 影響定数（ファイル名生成用）
    """
    if not results:
        print("結果が空のため、グラフは生成されません")
        return

    # デフォルトのグラフパラメータ
    if graph_params is None:
        graph_params = GRAPH_PARAMS

    # 出力ディレクトリの準備
    output_path = Path(__file__).parent.parent / "results" / output_dir
    if save:
        output_path.mkdir(parents=True, exist_ok=True)

    # 各グラフタイプごとにプロット
    for graph_type, result_list in results.items():
        for result_idx, result in enumerate(result_list):
            if "centralities_limited" not in result:
                continue

            centralities_full = result["centralities_full"]
            centralities_limited = result["centralities_limited"]

            fig, ax = plt.subplots(figsize=(8, 3.8))

            L_max_values = sorted(centralities_limited.keys())

            # 頂点リストの取得
            if centralities_full is not None:
                nodes = list(centralities_full.keys())
            else:
                max_L_max = max(L_max_values)
                nodes = list(centralities_limited[max_L_max].keys())

            # 頂点ID順にソート（数値または中身の数値で比較）
            def node_id_key(val):
                if isinstance(val, int):
                    return val
                if isinstance(val, str):
                    # 格子グラフ用のタプル文字列 '(x, y)' の処理
                    if val.strip().startswith("(") and "," in val:
                        try:
                            # タプルとして評価
                            return eval(val)
                        except Exception:
                            pass
                    # 数値文字列の処理
                    if val.replace(".", "", 1).isdigit():
                        return float(val)
                return val

            try:
                sorted_nodes = sorted(nodes, key=node_id_key)
            except TypeError:
                # 比較できない場合は文字列としてソート
                sorted_nodes = sorted(nodes, key=str)

            # 頂点数が多すぎる場合は間引く（ID順に等間隔→頂点番号刻み）
            if len(nodes) > MAX_PLOT_LINES:
                n_total = len(sorted_nodes)
                plot_nodes = [
                    sorted_nodes[
                        min(int(n_total * i / (MAX_PLOT_LINES - 1)), n_total - 1)
                    ]
                    for i in range(MAX_PLOT_LINES)
                ]
            else:
                plot_nodes = sorted_nodes

            # plot_nodesを中心性の大きい順に再ソート（凡例の順序を正しくするため）
            # ユーザー要望により頂点ID順（番号刻み）のままにするためソートを無効化
            # plot_nodes = sorted(
            #     plot_nodes, key=lambda n: sort_centrality[n], reverse=True
            # )

            # カラーマップを使用して色を明確に区別
            colors = plt.cm.tab10(range(len(plot_nodes)))

            # 線のスタイル
            if USE_LINE_STYLES:
                line_styles = ["-", "--", "-.", ":"]
            else:
                line_styles = ["-"]

            # 各頂点をプロット
            for idx, node in enumerate(plot_nodes):
                color = colors[idx % len(colors)]
                linestyle = (
                    line_styles[idx % len(line_styles)] if USE_LINE_STYLES else "-"
                )
                if centralities_full is not None:
                    # 改訂版を100%として正規化
                    full_value = centralities_full[node]
                    if full_value > 0:
                        values = [
                            (centralities_limited[L_max][node] / full_value) * 100
                            for L_max in L_max_values
                        ]
                    else:
                        values = [0] * len(L_max_values)
                else:
                    # 絶対値をそのまま表示
                    values = [
                        centralities_limited[L_max][node] for L_max in L_max_values
                    ]

                ax.plot(
                    L_max_values,
                    values,
                    linestyle,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=rf"頂点 ${node}$",
                    color=color,
                    alpha=LINE_ALPHA,
                )

            # 参考線
            if centralities_full is not None:
                # 100%の参考線（改訂版がある場合）
                ax.axhline(
                    y=100,
                    color="green",
                    linestyle="--",
                    alpha=0.5,
                )

            ax.set_xlabel(r"$L_{\max}$", fontsize=12)
            ax.tick_params(axis="both", which="major", labelsize=12)

            if centralities_full is not None:
                ax.set_ylabel(r"相対値 ($\%$)", fontsize=12)
                ax.set_ylim(0, 105)  # 0〜105%の範囲で表示
            else:
                ax.set_ylabel("中心性値", fontsize=12)

            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12, loc="best")

            plt.tight_layout()

            if save:
                # 単一グラフ用のanalyze_graphs辞書を作成
                single_graph_analyze = {
                    key: (key == graph_type)
                    for key in ["barbell", "lollipop", "kite", "comb", "grid", "path"]
                }

                # 個別プロット保存用に現在のパラメータのみを持つ辞書を作成
                current_params = {graph_type: result["parameters"]}

                img_filename = generate_filename(
                    single_graph_analyze, current_params, r, extension="pdf"
                )
                filename = output_path / img_filename
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"  グラフ保存: {filename}")

            plt.close()

            plt.close()

            # ケンドールの順位相関係数のプロットは不要とのことで削除
            # if "kendall_tau" in result and result["kendall_tau"]:
            #     ...

    print("\nグラフの生成が完了しました")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="パス長制限による情報損失の分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
実験設定:
  コード上部の設定セクションで以下を変更できます：
  - SELECT_GRAPHS: 分析するグラフの選択 (整数または整数のリスト、'all'で全て)
  - GRAPH_PARAMS: 各グラフのパラメータ
  - R_VALUE: 影響定数のデフォルト値
  - L_max_VALUES: パス長制限値のリスト

グラフの整数マッピング:
  0: バーベルグラフ
  1: ロリポップグラフ
  2: クラックハートの凧グラフ
  3: コームグラフ
  4: 格子グラフ
  5: パスグラフ
        """,
    )
    parser.add_argument(
        "--graph",
        type=int,
        nargs="*",
        default=None,
        help="分析するグラフ (0-5の整数、複数指定可、デフォルト: SELECT_GRAPHS)",
    )
    parser.add_argument(
        "--r",
        type=float,
        nargs="+",
        default=None,
        help=f"影響定数（デフォルト: {R_VALUE}、複数指定可）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力ファイル名（デフォルト: 自動生成）",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="改訂版Myerson中心性も計算する（比較・評価用）",
    )

    args = parser.parse_args()

    # コマンドライン引数が指定されていない場合は設定値を使用
    if args.r is not None:
        r_list = args.r
    else:
        if isinstance(R_VALUE, list):
            r_list = R_VALUE
        else:
            r_list = [R_VALUE]

    # 中心性計算の設定を更新
    if args.full:
        COMPUTE_FULL_CENTRALITY = True
        print("設定変更: 改訂版Myerson中心性を計算します")

    # グラフ選択を処理
    if args.graph is not None:
        if len(args.graph) == 0:
            analyze_graphs = get_selected_graphs()
        else:
            analyze_graphs = {key: False for key in GRAPH_TYPE_MAP.values()}
            for idx in args.graph:
                if idx in GRAPH_TYPE_MAP:
                    analyze_graphs[GRAPH_TYPE_MAP[idx]] = True
                else:
                    print(f"警告: 無効なグラフインデックス {idx} は無視されます")
    else:
        analyze_graphs = get_selected_graphs()

    # 各影響定数 r について実験を実行
    for r in r_list:
        print(f"\n{'=' * 60}")
        print(f"影響定数 r = {r} の実験を開始")
        print(f"{'=' * 60}")

        # 出力ファイル名を決定（--outputが指定されていなければ自動生成）
        if args.output is not None:
            # 複数rの場合はファイル名にrを含めるように加工が必要だが、
            # ユーザーが明示した場合は上書きのリスクがあるため、
            # rの値を含むようにサフィックスを追加する
            if len(r_list) > 1:
                base_path = Path(args.output)
                output = f"{base_path.stem}_r{int(r * 100):02d}{base_path.suffix}"
            else:
                output = args.output
        else:
            output = generate_filename(analyze_graphs, GRAPH_PARAMS, r)

        # 実験実行
        results, metadata = run_all_experiments(
            analyze_graphs=analyze_graphs, graph_params=GRAPH_PARAMS, r=r
        )

        # 結果保存（メタデータ付き）
        save_results(results, output, metadata=metadata)

        # グラフ生成
        if GENERATE_PLOTS:
            print(f"\nグラフを生成中 (r={r})...")
            plot_results(
                results,
                output_dir=PLOT_OUTPUT_DIR,
                save=SAVE_PLOTS,
                graph_params=GRAPH_PARAMS,
                r=r,
            )
