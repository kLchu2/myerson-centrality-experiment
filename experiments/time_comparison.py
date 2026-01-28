"""
中心性計算アルゴリズムの計算時間比較実験

比較対象：
- Myerson中心性 (myerson.py)
- Modified Myerson中心性 (modified_myerson.py) - Lmax = 2, 3, 4, 5, 6
- Path-Limited Myerson中心性 (path_limited_myerson.py) - Lmax = 2, 3, 4, 5, 6
- Revised Myerson中心性 (revised_myerson.py) - Lmax = 2, 3, 4, 5, 6

テストグラフ：
- 格子グラフ (Grid Graph)
- コームグラフ (Comb Graph)
- スケールフリーネットワーク (Barabási-Albert)
"""

import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# プロジェクト内モジュールのimport
from centrality.modified_myerson import (  # noqa: E402
    all_modified_myerson_centralities,
)
from centrality.myerson import all_myerson_centralities  # noqa: E402
from centrality.path_limited_myerson import (  # noqa: E402
    all_path_limited_myerson_centralities,
)
from centrality.revised_myerson import all_revised_myerson_centralities  # noqa: E402
from graph_utils.generator import (  # noqa: E402
    create_barabasi_albert,
    create_comb_graph,
    create_complete,
    create_erdos_renyi,
    create_grid_graph,
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


# ========== 実験設定 ==========

# ========== アルゴリズム選択 ==========
RUN_MYERSON = False  # オリジナルMyerson中心性を実行
RUN_MODIFIED = False  # Modified Myerson中心性を実行
RUN_PATH_LIMITED = False  # Path-Limited Myerson中心性を実行
RUN_REVISED = True  # Revised Myerson中心性を実行

# ========== グラフ選択 ==========
RUN_GRID = False  # 格子グラフを実行
RUN_COMB = False  # コームグラフを実行
RUN_BA = False  # Barabási-Albertグラフを実行
RUN_ER = False  # Erdős-Rényiグラフを実行
RUN_COMPLETE = True  # 完全グラフを実行

# ========== グラフパラメータ ==========
# 格子グラフのサイズ（N×N）
GRID_SIZES = [10, 20, 30, 50, 70, 100]

# コームグラフのパラメータ（背骨の長さ, 歯の長さ）
COMB_PARAMS = [(50, 5), (100, 10), (150, 15), (200, 20), (250, 25), (300, 30)]

# Barabási-Albertグラフのパラメータ
BA_NODE_COUNTS = [50, 100, 150]
BA_M = 3  # 各頂点が追加されるときに接続する既存頂点数

# Erdős-Rényiグラフのパラメータ (n, p)
ER_PARAMS = [
    (100, 0.05),
    # (1000, 0.005),
    # (2000, 0.0025),
    # (5000, 0.001),
    # (10000, 0.0005),
]

# 完全グラフのパラメータ
COMPLETE_NODE_COUNTS = [10, 12, 14, 16, 20, 25]

# ========== 計算パラメータ ==========
# Lmaxの範囲
L_MAX_VALUES = [2, 3, 4, 5, 6]

# 影響定数
INFLUENCE_CONSTANT = 1.0

# タイムアウト時間（秒）
TIMEOUT_SECONDS = 300  # 5分

# 結果出力ディレクトリ
RESULTS_DIR = project_root / "results"
JSON_DIR = RESULTS_DIR / "json"
FIGURES_DIR = RESULTS_DIR / "figures"

# ディレクトリ作成
JSON_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ========== テストグラフ生成 ==========


def generate_test_graphs() -> List[Tuple[str, nx.Graph, Dict[str, Any]]]:
    """
    テスト用グラフを生成（設定に基づく）

    Returns:
        List of (graph_name, graph, metadata)
    """
    graphs = []

    # 1. 格子グラフ (Grid Graph)
    if RUN_GRID:
        for size in GRID_SIZES:
            G = create_grid_graph(size, size)
            metadata = {
                "type": "grid",
                "size": size,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
            }
            graphs.append((f"Grid_{size}x{size}", G, metadata))

    # 2. コームグラフ (Comb Graph)
    if RUN_COMB:
        for spine_len, tooth_len in COMB_PARAMS:
            G = create_comb_graph(spine_len, tooth_len)
            metadata = {
                "type": "comb",
                "spine_length": spine_len,
                "tooth_length": tooth_len,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
            }
            graphs.append((f"Comb_s{spine_len}_t{tooth_len}", G, metadata))

    # 3. Barabási-Albertスケールフリーネットワーク
    if RUN_BA:
        for n in BA_NODE_COUNTS:
            G = create_barabasi_albert(n, BA_M, seed=42)
            metadata = {
                "type": "barabasi_albert",
                "n": n,
                "m": BA_M,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
            }
            graphs.append((f"BA_n{n}_m{BA_M}", G, metadata))

    # 4. Erdős-Rényiランダムグラフ
    if RUN_ER:
        for n, p in ER_PARAMS:
            G = create_erdos_renyi(n, p, seed=42)
            metadata = {
                "type": "erdos_renyi",
                "n": n,
                "p": p,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
            }
            graphs.append((f"ER_n{n}_p{p}", G, metadata))

    # 5. 完全グラフ (Complete Graph)
    if RUN_COMPLETE:
        for n in COMPLETE_NODE_COUNTS:
            G = create_complete(n)
            metadata = {
                "type": "complete",
                "n": n,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
            }
            graphs.append((f"Complete_n{n}", G, metadata))

    return graphs


# ========== 計算時間測定 ==========


def measure_computation_time(
    func, G: nx.Graph, timeout: int = TIMEOUT_SECONDS, **kwargs
) -> Tuple[bool, float, Any]:
    """
    計算時間を測定（タイムアウト付き）

    Args:
        func: 実行する関数
        G: NetworkXグラフ
        timeout: タイムアウト時間（秒）
        **kwargs: 関数に渡す追加引数

    Returns:
        Tuple[成功フラグ, 実行時間, 結果]
    """
    import signal

    def handler(signum, frame):
        raise TimeoutError("計算時間がタイムアウトしました")

    # シグナルハンドラを設定
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    start_time = time.time()
    try:
        result = func(G, **kwargs)
        elapsed_time = time.time() - start_time
        signal.alarm(0)  # タイマーをキャンセル
        return True, elapsed_time, result
    except TimeoutError:
        elapsed_time = time.time() - start_time
        signal.alarm(0)  # タイマーをキャンセル
        return False, elapsed_time, None
    except Exception as e:
        elapsed_time = time.time() - start_time
        signal.alarm(0)  # タイマーをキャンセル
        print(f"エラー: {e}")
        return False, elapsed_time, None


# ========== 実験実行 ==========


def run_time_comparison_experiment() -> Dict[str, Any]:
    """
    計算時間比較実験を実行

    Returns:
        実験結果の辞書
    """
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "timeout_seconds": TIMEOUT_SECONDS,
        "influence_constant": INFLUENCE_CONSTANT,
        "L_max_values": L_MAX_VALUES,
        "experiments": [],
    }

    print("=" * 80)
    print("中心性計算アルゴリズムの計算時間比較実験")
    print("=" * 80)
    print(f"タイムアウト: {TIMEOUT_SECONDS}秒")
    print(f"Lmax範囲: {L_MAX_VALUES}")
    print(f"影響定数 r: {INFLUENCE_CONSTANT}")
    print("=" * 80)
    print()

    # テストグラフ生成
    graphs = generate_test_graphs()
    print(f"テストグラフ数: {len(graphs)}")
    print()

    # タイムアウトしたメソッドとグラフタイプの組み合わせを記録
    # (graph_type, method_key) -> True
    failed_configs = set()

    for graph_idx, (graph_name, G, metadata) in enumerate(graphs, 1):
        print(
            f"[{graph_idx}/{len(graphs)}] {graph_name} (N={metadata['nodes']}, E={metadata['edges']})"
        )
        graph_type = metadata["type"]

        experiment_result = {
            "graph_name": graph_name,
            "graph_metadata": metadata,
            "methods": {},
        }

        # 1. Myerson中心性（オリジナル）
        if RUN_MYERSON:
            method_key = "myerson"
            if (graph_type, method_key) in failed_configs:
                print("  Myerson: Skipped (Previous failure)")
            else:
                success, elapsed, centralities = measure_computation_time(
                    all_myerson_centralities,
                    G,
                    r=INFLUENCE_CONSTANT,
                    verbose=False,
                )
                experiment_result["methods"]["myerson"] = {
                    "success": success,
                    "elapsed_time": elapsed,
                    "L_max": None,
                }
                status = "✓" if success else "✗"
                print(f"  Myerson: {status} {elapsed:.4f}s")
                if not success:
                    failed_configs.add((graph_type, method_key))

        # 2. Modified Myerson中心性（各Lmax）
        if RUN_MODIFIED:
            for L_max in L_MAX_VALUES:
                method_key = f"modified_L{L_max}"
                if (graph_type, method_key) in failed_configs:
                    print(f"  Modified(L={L_max}): Skipped", end=" ")
                    continue

                success, elapsed, centralities = measure_computation_time(
                    all_modified_myerson_centralities,
                    G,
                    walk_length=L_max,
                    verbose=False,
                )
                experiment_result["methods"][f"modified_L{L_max}"] = {
                    "success": success,
                    "elapsed_time": elapsed,
                    "L_max": L_max,
                }
                status = "✓" if success else "✗"
                print(f"  Modified(L={L_max}): {status} {elapsed:.4f}s", end=" ")

                if not success:
                    # この設定で失敗したら、このグラフタイプのこのL_maxは今後スキップ
                    failed_configs.add((graph_type, method_key))
                    # さらに、現在のグラフでより大きいL_maxも無駄なのでスキップ（インナーループ脱出）
                    # ただし、今回は「より大きいパラメータ（グラフサイズ）」でのスキップが主眼なので
                    # ここでbreakしても良いが、L_maxループの残りも失敗とみなすのが自然
                    pass
            print()

        # 3. Path-Limited Myerson中心性（各Lmax）
        if RUN_PATH_LIMITED:
            for L_max in L_MAX_VALUES:
                method_key = f"path_limited_L{L_max}"
                if (graph_type, method_key) in failed_configs:
                    print(f"  PathLimited(L={L_max}): Skipped", end=" ")
                    continue

                success, elapsed, centralities = measure_computation_time(
                    all_path_limited_myerson_centralities,
                    G,
                    L_max=L_max,
                    r=INFLUENCE_CONSTANT,
                    verbose=False,
                    use_cache=False,
                )
                experiment_result["methods"][f"path_limited_L{L_max}"] = {
                    "success": success,
                    "elapsed_time": elapsed,
                    "L_max": L_max,
                }
                status = "✓" if success else "✗"
                print(f"  PathLimited(L={L_max}): {status} {elapsed:.4f}s", end=" ")

                if not success:
                    failed_configs.add((graph_type, method_key))
            print()

        # 4. Revised Myerson中心性（L_maxなし、内部でn-1を使用）
        # Note: Revised Myerson は常に L_max = n-1 を使用するため、
        # L_max パラメータは不要（比較のため1回のみ計算）
        if RUN_REVISED:
            method_key = "revised"
            if (graph_type, method_key) in failed_configs:
                print("  Revised: Skipped (Previous failure)")
            else:
                success, elapsed, centralities = measure_computation_time(
                    all_revised_myerson_centralities,
                    G,
                    r=INFLUENCE_CONSTANT,
                    verbose=True,
                    use_cache=False,
                )
                # Revised は全パスを見るため、L_max = n-1 として記録
                L_max_revised = G.number_of_nodes() - 1
                experiment_result["methods"]["revised"] = {
                    "success": success,
                    "elapsed_time": elapsed,
                    "L_max": L_max_revised,
                }
                status = "✓" if success else "✗"
                print(f"  Revised(L={L_max_revised}): {status} {elapsed:.4f}s")
                if not success:
                    failed_configs.add((graph_type, method_key))

        print()

        results["experiments"].append(experiment_result)

    return results


# ========== 結果の可視化 ==========


def visualize_results(results: Dict[str, Any]) -> None:
    """
    実験結果を可視化

    Args:
        results: 実験結果の辞書
    """
    # データフレームに変換
    data = []
    for exp in results["experiments"]:
        graph_name = exp["graph_name"]
        graph_type = exp["graph_metadata"]["type"]
        nodes = exp["graph_metadata"]["nodes"]

        for method_name, method_data in exp["methods"].items():
            # メソッド名を整形
            if method_name == "myerson":
                display_name = "Myerson"
                category = "Myerson"
                L_max = None
            elif method_name.startswith("modified_"):
                L_max = method_data["L_max"]
                display_name = f"Modified (L={L_max})"
                category = "Modified"
            elif method_name.startswith("path_limited_"):
                L_max = method_data["L_max"]
                display_name = f"Path-Limited (L={L_max})"
                category = "Path-Limited"
            elif method_name.startswith("revised_"):
                L_max = method_data["L_max"]
                display_name = f"Revised (L={L_max})"
                category = "Revised"
            else:
                continue

            data.append(
                {
                    "graph_name": graph_name,
                    "graph_type": graph_type,
                    "nodes": nodes,
                    "method": display_name,
                    "category": category,
                    "L_max": L_max,
                    "success": method_data["success"],
                    "elapsed_time": method_data["elapsed_time"],
                }
            )

    df = pd.DataFrame(data)

    # タイムスタンプ
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 図1: グラフタイプごとの計算時間比較（Lmax別）
    for graph_type in df["graph_type"].unique():
        df_type = df[df["graph_type"] == graph_type]

        # グラフごとにプロット
        unique_graphs = df_type["graph_name"].unique()
        n_graphs = len(unique_graphs)

        fig, axes = plt.subplots(1, n_graphs, figsize=(6 * n_graphs, 6))
        if n_graphs == 1:
            axes = [axes]

        for idx, graph_name in enumerate(unique_graphs):
            ax = axes[idx]
            df_graph = df_type[df_type["graph_name"] == graph_name]

            # カテゴリごとにグループ化
            categories = ["Myerson", "Modified", "Path-Limited", "Revised"]
            x_positions = []
            x_labels = []
            colors_list = []
            heights = []

            color_map = {
                "Myerson": "#1f77b4",
                "Modified": "#ff7f0e",
                "Path-Limited": "#2ca02c",
                "Revised": "#d62728",
            }

            pos = 0
            for category in categories:
                df_cat = df_graph[df_graph["category"] == category]
                if len(df_cat) == 0:
                    continue

                # Lmaxでソート
                df_cat = df_cat.sort_values("L_max")

                for _, row in df_cat.iterrows():
                    if row["success"]:
                        x_positions.append(pos)
                        if row["L_max"] is None:
                            x_labels.append(category)
                        else:
                            x_labels.append(f"L={row['L_max']}")
                        colors_list.append(color_map[category])
                        heights.append(row["elapsed_time"])
                        pos += 1

                pos += 0.5  # カテゴリ間のスペース

            # 棒グラフ
            bars = ax.bar(x_positions, heights, color=colors_list, alpha=0.8)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            ax.set_ylabel("計算時間（秒）", fontsize=12)
            ax.set_title(f"{graph_name}\n(N={df_graph['nodes'].iloc[0]})", fontsize=12)
            ax.set_yscale("log")
            ax.grid(True, axis="y", alpha=0.3)

            # 棒の上に数値を表示
            for bar, height in zip(bars, heights):
                if height < 1:
                    text = f"{height:.3f}"
                elif height < 10:
                    text = f"{height:.2f}"
                else:
                    text = f"{height:.1f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.suptitle(
            f"{graph_type.replace('_', ' ').title()}グラフにおける計算時間比較",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # 保存
        filename = f"time_comparison_{graph_type}_{timestamp}.png"
        filepath = FIGURES_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"図を保存: {filepath}")

    # 図2: 全体的な比較（カテゴリ別、Lmax別）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # サブプロット1: カテゴリ別の平均計算時間
    ax = axes[0, 0]
    df_success = df[df["success"]].copy()
    category_times = df_success.groupby("category")["elapsed_time"].mean()
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
    ]  # Myerson, Modified, Path-Limited, Revised
    bars = ax.bar(range(len(category_times)), category_times.values, color=colors)
    ax.set_xticks(range(len(category_times)))
    ax.set_xticklabels(category_times.index)
    ax.set_ylabel("平均計算時間（秒）", fontsize=12)
    ax.set_title("カテゴリ別平均計算時間", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, category_times.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # サブプロット2: Lmax別の平均計算時間（Modified, Path-Limited, Revised）
    ax = axes[0, 1]
    df_with_lmax = df_success[df_success["L_max"].notna()].copy()
    for category, color in zip(
        ["Modified", "Path-Limited", "Revised"], ["#ff7f0e", "#2ca02c", "#d62728"]
    ):
        df_cat = df_with_lmax[df_with_lmax["category"] == category]
        lmax_times = df_cat.groupby("L_max")["elapsed_time"].mean()
        ax.plot(
            lmax_times.index,
            lmax_times.values,
            marker="o",
            label=category,
            linewidth=2,
            markersize=8,
            color=color,
        )
    ax.set_xlabel("L_max", fontsize=12)
    ax.set_ylabel("平均計算時間（秒）", fontsize=12)
    ax.set_title("Lmax別平均計算時間", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # サブプロット3: グラフサイズ別の計算時間
    ax = axes[1, 0]
    for category, color in zip(
        ["Myerson", "Modified", "Path-Limited", "Revised"],
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    ):
        df_cat = df_success[df_success["category"] == category]
        # Lmaxごとに平均を取る
        size_times = df_cat.groupby("nodes")["elapsed_time"].mean()
        ax.plot(
            size_times.index,
            size_times.values,
            marker="o",
            label=category,
            linewidth=2,
            markersize=8,
            color=color,
        )
    ax.set_xlabel("グラフサイズ（頂点数）", fontsize=12)
    ax.set_ylabel("平均計算時間（秒）", fontsize=12)
    ax.set_title("グラフサイズ別平均計算時間", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # サブプロット4: 成功率
    ax = axes[1, 1]
    success_rates = df.groupby("category")["success"].mean() * 100
    bars = ax.bar(range(len(success_rates)), success_rates.values, color=colors)
    ax.set_xticks(range(len(success_rates)))
    ax.set_xticklabels(success_rates.index)
    ax.set_ylabel("成功率（%）", fontsize=12)
    ax.set_title("カテゴリ別成功率", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, success_rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.suptitle("計算時間比較の総合分析", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # 保存
    filename = f"time_comparison_summary_{timestamp}.png"
    filepath = FIGURES_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"総合図を保存: {filepath}")


def save_results_to_json(results: Dict[str, Any]) -> None:
    """
    実験結果をJSONファイルに保存

    Args:
        results: 実験結果の辞書
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"time_comparison_results_{timestamp}.json"
    filepath = JSON_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果保存: {filepath}")


def print_summary(results: Dict[str, Any]) -> None:
    """
    実験結果のサマリーを表示

    Args:
        results: 実験結果の辞書
    """
    print("\n" + "=" * 80)
    print("実験結果サマリー")
    print("=" * 80)

    # データフレームに変換
    data = []
    for exp in results["experiments"]:
        graph_name = exp["graph_name"]
        for method_name, method_data in exp["methods"].items():
            data.append(
                {
                    "graph": graph_name,
                    "method": method_name,
                    "success": method_data["success"],
                    "time": method_data["elapsed_time"],
                }
            )

    df = pd.DataFrame(data)

    # 各アルゴリズムの平均計算時間
    print("\n平均計算時間（秒）:")
    print("-" * 80)
    for method in sorted(df["method"].unique()):
        df_method = df[df["method"] == method]
        success_count = df_method["success"].sum()
        total_count = len(df_method)
        if success_count > 0:
            avg_time = df_method[df_method["success"]]["time"].mean()
            print(
                f"{method:30s}: {avg_time:8.4f} 秒 (成功: {success_count}/{total_count})"
            )
        else:
            print(f"{method:30s}: タイムアウト (成功: 0/{total_count})")

    print("\n" + "=" * 80)


# ========== メイン処理 ==========


def main():
    """メイン処理"""
    print(f"\n実験開始: {datetime.datetime.now()}")

    # 実験実行
    results = run_time_comparison_experiment()

    # 結果を保存
    save_results_to_json(results)

    print(f"\n実験終了: {datetime.datetime.now()}")
    print("=" * 80)
    print("\n結果はJSONファイルに保存されました。")


if __name__ == "__main__":
    main()
