"""
情報損失分析の簡易テスト
異なるL_max設定で動作を確認
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 元のinformation_loss_analysis.pyをインポート
import networkx as nx  # noqa: E402

import experiments.information_loss_analysis as analysis_module  # noqa: E402

# グラフの直径を確認するためのグラフ生成
from graph_utils.generator import create_barbell_graph, create_path_graph  # noqa: E402

print("=== L_max設定の動作確認 ===\n")

# テスト1: バーベルグラフ（小さい）
print("1. バーベルグラフ (n=3, l=2):")
G = create_barbell_graph(3, 2)
diameter = nx.diameter(G)
print(f"   直径: {diameter}")

# 現在の設定で計算される範囲を確認
L_max_range = analysis_module.get_L_max_range(diameter)
print(f"   設定モード: {analysis_module.L_max_MODE}")
print(f"   計算されるL_max: {L_max_range}")
print(f"   計算回数: {len(L_max_range)}回\n")

# テスト2: パスグラフ（中規模）
print("2. パスグラフ (n=15):")
G = create_path_graph(15)
diameter = nx.diameter(G)
print(f"   直径: {diameter}")

L_max_range = analysis_module.get_L_max_range(diameter)
print(f"   設定モード: {analysis_module.L_max_MODE}")
print(f"   計算されるL_max: {L_max_range}")
print(f"   計算回数: {len(L_max_range)}回\n")

# 設定変更の例を表示
print("=== 設定変更の例 ===\n")

print("【例1】特定のL_max値だけを比較したい場合:")
print("  L_max_MODE = 'fixed'")
print("  L_max_VALUES = [1, 3, 5]")
print("  → L_max=1, 3, 5の3回だけ計算\n")

print("【例2】直径の半分までだけ調べたい場合:")
print("  L_max_MODE = 'ratio'")
print("  L_max_RATIOS = [0.5]")
print("  → 直径10なら L_max=5 を計算\n")

print("【例3】大きいグラフでも計算量を抑えたい場合:")
print("  L_max_MODE = 'auto'")
print("  L_max_AUTO_LIMIT = 10")
print("  → 直径が30でもL_max=1〜10まで\n")

print("【例4】粗く間引いて計算したい場合:")
print("  L_max_MODE = 'fixed'")
print("  L_max_VALUES = [1, 2, 4, 8, 16]")
print("  → 指数的に間隔を広げて計算\n")
