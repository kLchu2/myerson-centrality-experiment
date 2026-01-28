"""
L_max設定のテストスクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# experiments/information_loss_analysis.pyのget_L_max_range関数をテスト
# ただし、設定変数を直接操作するために関数を再定義して確認


def get_L_max_range(diameter, mode="auto", values=None, ratios=None, auto_limit=None):
    """
    設定に基づいてL_maxの範囲を取得

    Args:
        diameter: グラフの直径
        mode: 'auto', 'fixed', 'ratio'
        values: fixedモード用の固定リスト
        ratios: ratioモード用の比率リスト
        auto_limit: autoモードでの上限

    Returns:
        list: L_max値のリスト
    """
    if mode == "auto":
        # 直径まで全て（上限設定があればそれを適用）
        max_val = diameter if auto_limit is None else min(diameter, auto_limit)
        return list(range(1, max_val + 1))
    elif mode == "fixed":
        # 固定リスト（直径を超えるものは除外）
        return [l for l in values if l <= diameter]  # noqa: E741
    elif mode == "ratio":
        # 直径の比率で計算
        result = []
        for ratio in ratios:
            L_max = max(1, int(diameter * ratio))
            if L_max not in result:
                result.append(L_max)
        return sorted(result)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# テストケース
print("=== L_max設定のテスト ===\n")

# 直径10のグラフを想定
diameter = 10

print(f"グラフの直径: {diameter}\n")

# 1. autoモード（デフォルト）
print("1. autoモード（直径まで全て）:")
result = get_L_max_range(diameter, mode="auto")
print(f"   L_max範囲: {result}")
print(f"   計算回数: {len(result)}回\n")

# 2. autoモードwith上限
print("2. autoモード（上限5）:")
result = get_L_max_range(diameter, mode="auto", auto_limit=5)
print(f"   L_max範囲: {result}")
print(f"   計算回数: {len(result)}回\n")

# 3. fixedモード
print("3. fixedモード:")
fixed_values = [1, 2, 3, 5, 8]
result = get_L_max_range(diameter, mode="fixed", values=fixed_values)
print(f"   設定値: {fixed_values}")
print(f"   L_max範囲: {result}")
print(f"   計算回数: {len(result)}回\n")

# 4. ratioモード
print("4. ratioモード:")
ratios = [0.25, 0.5, 0.75, 1.0]
result = get_L_max_range(diameter, mode="ratio", ratios=ratios)
print(f"   比率設定: {ratios}")
print(f"   L_max範囲: {result}")
print("   詳細:")
for ratio in ratios:
    L_max = max(1, int(diameter * ratio))
    print(f"     - {ratio:.2f} × {diameter} = {L_max}")
print(f"   計算回数: {len(result)}回\n")

# 5. 大きな直径の場合
print("5. 大きな直径（30）での各モード:")
large_diameter = 30

print("   autoモード（上限なし）:")
result_auto = get_L_max_range(large_diameter, mode="auto")
print(f"     計算回数: {len(result_auto)}回")

print("   autoモード（上限10）:")
result_auto_limit = get_L_max_range(large_diameter, mode="auto", auto_limit=10)
print(f"     計算回数: {len(result_auto_limit)}回")

print("   ratioモード:")
result_ratio = get_L_max_range(
    large_diameter, mode="ratio", ratios=[0.25, 0.5, 0.75, 1.0]
)
print(f"     L_max範囲: {result_ratio}")
print(f"     計算回数: {len(result_ratio)}回\n")

print("=== 推奨設定 ===")
print("- autoモード: 小さいグラフ（直径 < 10）の詳細分析に最適")
print("- autoモード（上限付き）: 大きいグラフでも計算量を制限")
print("- fixedモード: 特定のL_max値だけを比較したい場合")
print("- ratioモード: 様々なサイズのグラフで一貫した相対的な制限を適用")
