"""
中心性タイプの切り替えとグラフ選択のテスト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiments.information_loss_analysis import (  # noqa: E402
    GRAPH_TYPE_MAP,
    get_centrality_function,
    get_selected_graphs,
)


def test_centrality_function():
    """中心性関数の取得テスト"""
    print("=== 中心性関数の取得テスト ===")

    for i in range(3):
        func, name = get_centrality_function(i)
        print(f"タイプ {i}: {name}")
        print(f"  関数名: {func.__name__}")

    # 無効なタイプ
    try:
        get_centrality_function(99)
        print("エラー: 無効なタイプが受け入れられました")
    except ValueError as e:
        print(f"正しくエラーが発生: {e}")

    print()


def test_graph_selection():
    """グラフ選択のテスト"""
    print("=== グラフ選択のテスト ===")

    # 元のSELECT_GRAPHSを保存
    from experiments import information_loss_analysis

    original_select = information_loss_analysis.SELECT_GRAPHS

    # 'all' の場合
    information_loss_analysis.SELECT_GRAPHS = "all"
    graphs = get_selected_graphs()
    print("SELECT_GRAPHS = 'all':")
    print(f"  選択されたグラフ: {[k for k, v in graphs.items() if v]}")

    # 単一グラフの場合
    information_loss_analysis.SELECT_GRAPHS = 2
    graphs = get_selected_graphs()
    print("\nSELECT_GRAPHS = 2:")
    print(f"  選択されたグラフ: {[k for k, v in graphs.items() if v]}")

    # 複数グラフの場合
    information_loss_analysis.SELECT_GRAPHS = [0, 2, 4]
    graphs = get_selected_graphs()
    print("\nSELECT_GRAPHS = [0, 2, 4]:")
    print(f"  選択されたグラフ: {[k for k, v in graphs.items() if v]}")

    # 復元
    information_loss_analysis.SELECT_GRAPHS = original_select

    print()


def test_graph_type_map():
    """グラフタイプマッピングの確認"""
    print("=== グラフタイプマッピング ===")
    for idx, name in GRAPH_TYPE_MAP.items():
        print(f"{idx}: {name}")
    print()


if __name__ == "__main__":
    test_centrality_function()
    test_graph_selection()
    test_graph_type_map()

    print("=== 全てのテスト完了 ===")
