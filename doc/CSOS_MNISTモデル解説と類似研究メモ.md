# CSOS_MNIST モデル解説と類似研究メモ

## 1. モデルの目的と全体像

- **タスク**: MNIST 手書き数字認識（28×28 画像 → 0〜9 クラス分類）
- **特徴**:
  - 一般的な CNN / MLP ではなく、
  - **「分子濃度 + 化学反応ダイナミクス」**で推論を行うニューラルモデル
- **基本アイデア**:
  1. 画像を「分子の初期濃度分布」に変換（Encoder）
  2. 3階テンソル W を用いた**二次相互作用（化学反応）**で数ステップ時間発展
  3. 最終濃度の一部（分子 ID 0〜9）をクラス確率として読み出す

---

## 2. モデル構造の詳細

### 2.1 CSOS_MNIST のパラメータ

- `num_basis = N`  
  分子の種類数（基底の数）。例: N=64
- `steps`  
  反応ステップ数（反復更新回数）
- `decay`  
  濃度の減衰率
- `alpha`  
  反応項の強さ

```python
self.encoder = nn.Linear(28*28, num_basis)     # 画像 -> 分子濃度
self.W = nn.Parameter(torch.randn(N, N, N)*.05) # 触媒テンソル（反応ルール）
```

- `W[k, i, j]`:
  - 「分子 i と j が反応して分子 k を生成/変化させる強さ」を表現するイメージ
  - パラメータ数は `O(N^3)`

---

### 2.2 フォワード処理の3フェーズ

#### Phase 1: Injection（初期濃度の生成）

```python
x = x.view(B, 784)
x_current = torch.relu(self.encoder(x))
x_current = x_current / (x_current.sum(dim=1, keepdim=True) + 1e-8)
```

- 画像 → 線形変換 + ReLU で**非負の濃度ベクトル**を得る
- 和が 1 になるよう正規化 → **確率分布/濃度分布**として扱える

#### Phase 2: Reaction（化学反応による推論）

```python
for _ in range(self.steps):
    x_mixed = (1 - decay) * x_current
    interaction = torch.einsum('ijk,bi,bj->bk', W, x_mixed, x_mixed)
    x_pre = x_mixed + alpha * interaction
    x_activated = torch.relu(x_pre)
    x_current = x_activated / (x_activated.sum(dim=1, keepdim=True) + 1e-8)
```

- `x_mixed`:
  - 前ステップの状態に減衰 `(1 - decay)` をかけたもの
- `interaction`:
  - 二次相互作用  
    \[
      \text{interaction}_k = \sum_{i,j} W_{kij} \, x_i x_j
    \]
  - 「分子同士が反応して新たな分子が生まれる」過程を表現
- ReLU + 正規化を毎ステップ行い、
  - 常に「非負 & 総和=1」を保ったまま時間発展
  - 数ステップ後の**安定したパターン**を「答え」とみなす

#### Phase 3: Readout（観測）

```python
output_concentration = x_current[:, :10]
logits = torch.log(output_concentration + 1e-8)  # NLLLoss 用
```

- 分子 ID 0〜9 の濃度を、そのままクラス 0〜9 の確率と解釈
- `log()` を取って log-probability として返す

---

## 3. 学習の設定

- **データ**: torchvision の MNIST
- **損失関数**: `F.nll_loss`（Negative Log Likelihood Loss）
- **最適化**: Adam (`lr=0.005`)
- **学習ループ**:
  - バッチごとに順伝播 → NLLLoss → 逆伝播 → `optimizer.step()`
  - エポックごとに Train Loss / Accuracy を表示
  - 最後にテストセットで平均損失と正解率を評価

---

## 4. このモデルに対する所見（意見）

### 4.1 良い点・ユニークな特徴

- **ダイナミカルシステムとしての設計**
  - 状態ベクトル `x_current` を反復更新する「離散時間力学系」としてきれい
  - 一発変換ではなく「時間発展で安定パターンに到達する」構図が面白い

- **確率単体（非負 + 総和=1）上のダイナミクス**
  - 各ステップで ReLU + 正規化しており、
  - 常に「濃度分布 / 確率分布」として解釈可能
  - 進化ゲーム理論のレプリケータダイナミクスなどと構造的に近い

- **3階テンソル W による二次相互作用**
  - `x_i x_j` の組み合わせを明示的に扱うことで、
    - 線形層だけでは表現しづらい**高次の非線形相互作用**を表現
  - これを「化学反応」として解釈しているのが独自性

### 4.2 課題・気になりそうな点

- **パラメータ数と計算量のスケーリング**
  - W は `O(N^3)` パラメータで、計算も重くなりやすい
  - 実務レベルのスケールでは、
    - テンソル分解（CP分解/Tucker分解）
    - `W[k, i, j] ≈ A[k, i] B[k, j]` のような低ランク化  
    などの工夫が必要になりそう

- **安定性と勾配伝播**
  - ReLU + 正規化により発散しづらい一方、
    - 勾配が消えやすい可能性もある
  - `decay` と `alpha` の値によって挙動が大きく変わるので、
    - これらを系統的にスイープして「どの領域が安定・高性能か」を調べる余地がある

- **解釈性**
  - コンセプトは直感的だが、学習後の `W[k, i, j]` をどう解釈するかは難しい
  - クラスごとに特徴的な反応パターンがあるかを可視化すると、
    - このモデルならではの解釈が得られる可能性がある

---

## 5. 類似・関連する研究領域

このモデルは、以下のいくつかの研究テーマの「交差点」にあると考えられる。

### 5.1 ダイナミカルシステムとしてのニューラルネット

- **Neural ODE** / Neural Ordinary Differential Equations
  - 連続時間の微分方程式として NN を扱う
- **Deep Equilibrium Models**
  - 反復更新の**固定点**を直接解くアプローチ
- 関連キーワード:
  - `"neural ordinary differential equations"`
  - `"deep equilibrium model"`
  - `"neural network as dynamical system"`

この CSOS モデルは、**離散時間版のダイナミカルシステム**として位置付けられる。

### 5.2 Hopfield ネットワーク / エネルギーベースモデル

- 古典的 Hopfield ネット:
  - エネルギー最小化により記憶パターンに収束する
- Modern Hopfield Network / Dense Associative Memory:
  - 連続ベクトル・高容量な記憶を扱う新しい Hopfield 系
- 関連キーワード:
  - `"Hopfield network"`
  - `"modern Hopfield network"`
  - `"energy-based model for classification"`

CSOS はエネルギー関数は明示されていないが、  
**反復更新 → 安定状態を読む**という点で Hopfield 的な構造を持つ。

### 5.3 高次（テンソル）相互作用をもつニューラルネット

- Polynomial Networks / Higher-Order Neural Networks
- 三次テンソルを使う RNN / Boltzmann Machine など
- 関連キーワード:
  - `"higher-order neural network"`
  - `"polynomial neural network"`
  - `"multiplicative interactions neural network"`
  - `"tensor recurrent neural network"`

3階テンソル `W` による `x_i x_j` の相互作用は、  
これらの**高次相互作用型 NN**と本質的に近い。

### 5.4 確率単体上のダイナミクス

- レプリケータダイナミクス（進化ゲーム理論）
- Mirror Descent などの最適化アルゴリズム
- 関連キーワード:
  - `"replicator dynamics"`
  - `"evolutionary game dynamics"`
  - `"probability simplex dynamical system"`

「非負 + 総和=1」の制約の下で状態を更新するという点で、  
これらの枠組みとも構造的に関連している。

### 5.5 反応拡散系 / ニューラルセルオートマトン

- 反応拡散方程式を NN として利用する研究
- Neural Cellular Automata（局所ルールでパターン生成・維持）
- 関連キーワード:
  - `"reaction-diffusion neural network"`
  - `"neural cellular automata"`
  - `"chemical reaction network model learning"`

本モデルは「拡散（空間方向）」は持たず、  
**抽象的な N 種の分子の相互作用だけに集中**している点が異なるが、  
発想としてはかなり近い領域に位置する。

---

## 6. まとめ

- CSOS_MNIST は、
  - 画像を「分子濃度ベクトル」に変換し、
  - 3階テンソルで定義された「化学反応ダイナミクス」を数ステップ回すことで、
  - 最終的な濃度パターンからクラスを読み出すモデルである。
- これは、
  - ダイナミカルシステムとしてのニューラルネット
  - Hopfield / エネルギーベースモデル
  - 高次（テンソル）相互作用ネットワーク
  - 確率単体上のダイナミクス
  - 反応系・化学メタファーを用いた NN  
  などのアイデアを、**「化学反応」という統一的なメタファー**でまとめた設計とみなせる。

今後の拡張の方向性としては、
- W の低ランク化や構造制約によるスケーラビリティ向上
- 反応ステップ数の増減や連続時間化（Neural ODE 的拡張）
- エネルギー関数の導出と Hopfield 的な理論解析
- 学習済み W の可視化・解釈（クラスごとの反応パターンの分析）

などが考えられる。