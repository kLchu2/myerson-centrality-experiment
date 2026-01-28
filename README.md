# Myerson中心性の計算機実験

NetworkXを使ったMyerson中心性とその拡張の実装・実験プロジェクト

## 概要

このプロジェクトでは、グラフ理論におけるMyerson中心性に基づく様々な中心性指標を実装しています。
異なるタイプのグラフに対して中心性指標の実験とテストを容易にするように構成されています。

### Myerson中心性とは

協力ゲーム理論におけるシャープレイ値をネットワーク構造に適用したもので、各頂点がネットワーク全体の価値にどれだけ貢献しているかを測る指標です。

このプロジェクトでは3つのバージョンを実装しています：

#### 通常版Myerson中心性

**定義**: $Y_G(v) = \sum_{l=1}^{L} \frac{A_l(v)}{l+1} \cdot r^l$

- $A_l(v)$: 頂点 $v$ を含む長さ $l$ の**最短パス**の本数
- $L$: 最短パスとして存在する最大の長さ
- $r$: 影響定数パラメータ

#### 改訂版Myerson中心性

**定義**: $Y^*_G(v) = \sum_{l=1}^{n-1} \frac{B_l(v)}{l+1} \cdot r^l$

- $B_l(v)$: 頂点 $v$ を含む長さ $l$ の**全ての単純パス**の本数
- $n-1$: 単純パスの理論的上限（n頂点のグラフにおける単純パスの最大長）
- $r$: 影響定数パラメータ

#### パス長制限Myerson中心性

**定義**: $Y^{L_{max}}_G(v) = \sum_{l=1}^{L_{max}} \frac{B_l(v)}{l+1} \cdot r^l$

- $B_l(v)$: 頂点 $v$ を含む長さ $l$ の**全ての単純パス**の本数
- $L_{max}$: ユーザー指定の最大パス長（制限値）
- $r$: 影響定数パラメータ

## プロジェクト構成

```files
myerson-centrality-experiment/
├── centrality/              # 中心性計算モジュール
│   ├── __init__.py
│   ├── base.py              # 共通基盤（キャッシュ管理、パス数計算）
│   ├── myerson.py           # 通常版Myerson中心性
│   ├── revised_myerson.py   # 改訂版Myerson中心性
│   └── path_limited_myerson.py # パス長制限Myerson中心性
├── graph_utils/             # グラフ生成・読み込み
│   ├── __init__.py
│   └── generator.py         # テスト用グラフ生成（特殊グラフ含む）
├── experiments/             # 実験スクリプト
│   ├── __init__.py
│   └── information_loss_analysis.py  # 情報損失分析実験
├── tests/                   # テストコード
│   ├── __init__.py
│   ├── test_myerson.py
│   ├── test_path_limited_myerson.py
│   ├── test_special_graphs.py
│   ├── test_cache.py
│   ├── test_batch_centrality.py
│   ├── test_batch_performance.py
│   └── その他デバッグ・比較スクリプト
├── utils/                   # ユーティリティモジュール
│   ├── analyzer.py          # 分析ツール
│   └── metrics.py           # メトリクス計算
├── cache/                   # パス数キャッシュ（自動生成）
├── results/                 # 結果出力
│   ├── figures/             # グラフ画像
│   └── json/                # 実験結果データ（JSON）
├── requirements.txt         # 依存パッケージ
└── README.md               # このファイル
```

## インストール

### 1. プロジェクトの準備

ZIPをダウンロードして展開するか、プロジェクトフォルダを目的の場所にコピーしてください。

### 2. 仮想環境の作成（推奨）

```bash
cd /path/to/myerson-centrality-experiment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# または
.venv\Scripts\activate  # Windows
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

## 使い方

### 基本的な使用例

```python
import networkx as nx
from centrality import (
    all_myerson_centralities,
    all_revised_myerson_centralities,
    all_path_limited_myerson_centralities
)

# グラフ作成
G = nx.karate_club_graph()

# 通常版Myerson中心性を計算
centralities = all_myerson_centralities(G, r=0.8, verbose=True)
print("通常版:", centralities)

# 改訂版Myerson中心性を計算
revised_centralities = all_revised_myerson_centralities(G, r=0.8, verbose=True)
print("改訂版:", revised_centralities)

# パス長制限Myerson中心性を計算
limited_centralities = all_path_limited_myerson_centralities(G, L_max=5, r=0.8, verbose=True)
print("制限版 (L_max=5):", limited_centralities)
```

### テスト用グラフの生成

```python
from graph_utils import create_triangle, create_path, create_star

# 三角形グラフ
G_triangle = create_triangle()

# パスグラフ（A-B-C）
G_path = create_path(3)

# スターグラフ
G_star = create_star(4)
```

### 特殊グラフ構造の生成

```python
from graph_utils import (
    create_barbell_graph,
    create_lollipop_graph,
    create_kite_graph,
    create_comb_graph,
    create_grid_graph,
    create_path_graph
)

# バーベルグラフ（n=5の2つのクリーク、l=3の橋）
G_barbell = create_barbell_graph(5, 3)

# ロリポップグラフ（n=5のクリーク、l=4の尻尾）
G_lollipop = create_lollipop_graph(5, 4)

# クラックハートの凧グラフ
G_kite = create_kite_graph()

# コームグラフ（n=5の背骨、l=3の歯）
G_comb = create_comb_graph(5, 3)

# 格子グラフ（5×5）
G_grid = create_grid_graph(5, 5)

# パスグラフ（10頂点）
G_path = create_path_graph(10)
```

### パス長制限Myerson中心性の使用例

```python
from centrality import (
    all_revised_myerson_centralities,
    all_path_limited_myerson_centralities
)
import networkx as nx

G = nx.karate_club_graph()

# 改訂版Myerson中心性（制限なし）
centralities_full = all_revised_myerson_centralities(G, r=0.8, verbose=True)

# パス長制限Myerson中心性（L_max = 3）
centralities_limited = all_path_limited_myerson_centralities(G, L_max=3, r=0.8, verbose=True)

# 保持率の計算
for node in G.nodes():
    retention = centralities_limited[node] / centralities_full[node] if centralities_full[node] > 0 else 1.0
    print(f"Node {node}: {retention:.2%} 保持")
```

### キャッシュ機能

パス数の計算結果は自動的にキャッシュされ、同じグラフに対する再計算を高速化します：

```python
from centrality.base import PathCountsCache

# キャッシュは cache/ ディレクトリに自動保存されます
# グラフ構造のハッシュ値をもとにキャッシュファイルが管理されます
```

## 実験スクリプト

### 情報損失分析（information_loss_analysis.py）

パス長制限による情報損失を複数のグラフタイプで分析するスクリプトです。

#### 実行方法

```bash
# デフォルト設定で実行
python experiments/information_loss_analysis.py

# 実験パラメータはファイル内の定数で調整：
# - SELECT_GRAPHS: 分析対象のグラフ（0-5、または 'all'）
# - GRAPH_PARAMS: 各グラフのパラメータ（サイズ等）
# - R_VALUE: 影響定数r
# - L_max_VALUES: 評価するL_max値のリスト
```

#### 出力

- **results/json/<グラフ名>_<パラメータ>.json**: 実験結果データ
- **results/figures/**: 生成されたグラフ画像（自動生成）

#### 実験で評価されるグラフ

0. バーベルグラフ (Barbell Graph)
1. ロリポップグラフ (Lollipop Graph)
2. クラックハートの凧グラフ (Kite Graph)
3. コームグラフ (Comb Graph)
4. 格子グラフ (Grid Graph)
5. パスグラフ (Path Graph)

## テスト

### 全テストを実行

```bash
pytest tests/ -v
```

### 主要なテストファイル

- **test_myerson.py**: 通常版Myerson中心性の基本動作テスト
- **test_path_limited_myerson.py**: パス長制限Myerson中心性のテスト
- **test_special_graphs.py**: 特殊グラフ構造での中心性テスト
- **test_cache.py**: キャッシュ機能のテスト
- **test_batch_centrality.py**: バッチ計算のテスト
- **test_batch_performance.py**: パフォーマンステスト

## 依存パッケージ

- **networkx** (>=3.0): グラフ操作・最短パス計算
- **numpy** (>=1.24.0): 数値計算
- **matplotlib** (>=3.7.0): グラフ可視化
- **pytest** (>=7.0.0): テストフレームワーク
- **pandas** (>=2.0.0): データ処理

## 応用例

### L_max値を変化させた系統的分析

```python
from graph_utils import create_comb_graph
from centrality import (
    all_revised_myerson_centralities,
    all_path_limited_myerson_centralities
)

G = create_comb_graph(n=6, l=3)

# L_maxを変化させて分析
L_max_values = [2, 4, 6, 8, 10]

# 改訂版中心性（制限なし）を計算
full_centralities = all_revised_myerson_centralities(G, r=0.8)

for L_max in L_max_values:
    limited_centralities = all_path_limited_myerson_centralities(G, L_max=L_max, r=0.8)
    
    # 各ノードの保持率を計算
    for node in G.nodes():
        retention = limited_centralities[node] / full_centralities[node] if full_centralities[node] > 0 else 1.0
        print(f"L_max={L_max}, Node {node}: {retention:.2%} 保持")
```

### バーベルグラフで橋の長さの影響を調査

```python
from graph_utils import create_barbell_graph
from centrality import all_revised_myerson_centralities, all_path_limited_myerson_centralities

# クリークサイズを固定して、橋の長さを変化させる
n = 5
for l in range(2, 7):
    G = create_barbell_graph(n, l)
    
    # 制限なしと制限ありを比較
    cent_full = all_revised_myerson_centralities(G, r=0.8)
    cent_limited = all_path_limited_myerson_centralities(G, L_max=5, r=0.8)
    
    # 橋の頂点の中心性を抽出
    bridge_nodes = [v for v in G.nodes() if G.degree(v) == 2]
    
    print(f"橋の長さ l={l}")
    for node in bridge_nodes:
        retention = cent_limited[node] / cent_full[node] if cent_full[node] > 0 else 1.0
        print(f"  橋頂点{node}: {retention:.2%}保持")
```

## 注意点

- 大規模グラフの場合、計算時間が長くなる可能性があります
  - キャッシュ機能により2回目以降の計算は高速化されます
- パス数の計算結果は `cache/` ディレクトリに自動保存されます
- 非連結グラフの場合、各頂点が含まれる連結成分のみで計算されます
- OneDrive等のクラウド同期フォルダでは、キャッシュファイルの同期によりI/Oが遅くなることがあります

## 主要な機能

### キャッシュシステム

- グラフのパス数計算結果を自動的にキャッシュ
- グラフ構造のハッシュ値でキャッシュファイルを管理
- 同一グラフに対する再計算を大幅に高速化

### バッチ計算

- 全頂点の中心性を効率的に一括計算
- 進捗表示機能（`verbose=True`）

### 柔軟なパラメータ設定

- 影響定数 `r` の調整が可能
- パス長制限 `L_max` の設定が可能

## 分析対象のグラフ構造

このプロジェクトでは、以下の特殊なグラフ構造を用いて中心性指標を分析できます：

1. **バーベルグラフ BG(n, l)**: 2つの完全グラフ（サイズn）をパス（長さl）で連結
2. **ロリポップグラフ LG(n, l)**: 完全グラフ（サイズn）にパス（長さl）が接続
3. **クラックハートの凧グラフ**: 古典的な橋渡し役の評価例
4. **コームグラフ CG(n, l)**: パス（長さn）の各頂点からパス（長さl）が分岐
5. **格子グラフ GG(m, n)**: m×n の格子状に連結
6. **パスグラフ PG(n)**: n頂点が直線状に連結

これらのグラフは`graph_utils.generator`モジュールから利用できます。

```python
from graph_utils import (
    create_barbell_graph,    # BG(n, l)
    create_lollipop_graph,   # LG(n, l)
    create_kite_graph,       # 凧グラフ
    create_comb_graph,       # CG(n, l)
    create_grid_graph,       # GG(m, n)
    create_path_graph        # PG(n)
)
```

## 実装状況

- [x] 通常版Myerson中心性
- [x] 改訂版Myerson中心性
- [x] パス長制限Myerson中心性
- [x] キャッシュシステム
- [x] バッチ計算機能
- [x] 情報損失分析実験
- [x] テストスイート

## ライセンス

（必要に応じて記載）

## 参考文献

（必要に応じて記載）
